package com.saunak.director.director

import android.graphics.Bitmap
import android.net.Uri
import com.saunak.director.ai.VlmRuntime
import com.saunak.director.camera.CameraEngine
import com.saunak.director.camera.DirectorExtension
import com.saunak.director.camera.SierraCaptureBridge
import com.saunak.director.engine.capture.CaptureProgram
import com.saunak.director.engine.capture.SafeBurstExecutor
import com.saunak.director.engine.device.CapturePath
import com.saunak.director.engine.experts.YuNetFaceExpert
import com.saunak.director.engine.reconstruction.ComputationalRawOutcome
import com.saunak.director.engine.reconstruction.ComputationalRawPipeline
import com.saunak.director.engine.reconstruction.ComputationalRawResult
import com.saunak.director.engine.reconstruction.ComputationalRawSafetyOnly
import com.saunak.director.platform.ThermalMonitor
import com.saunak.director.processing.InstantFinishEngine
import com.saunak.director.processing.LookProcessor
import com.saunak.director.processing.LookSpec
import com.saunak.director.processing.PhotoQuality
import com.saunak.director.processing.TechnicalQuality
import com.saunak.director.skills.SkillEngine
import com.saunak.director.storage.PhotoRepository
import com.saunak.director.storage.RecipeMemory
import com.saunak.director.storage.SierraCropPolicy
import com.saunak.director.tools.ToolExecutor
import com.saunak.director.tools.ToolIds
import java.io.File
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.withTimeoutOrNull

class DirectorOrchestrator(
    private val camera: CameraEngine,
    private val vlm: VlmRuntime,
    private val skills: SkillEngine,
    private val tools: ToolExecutor,
    private val storage: PhotoRepository,
    private val recipeMemory: RecipeMemory,
    private val thermal: ThermalMonitor,
    private val onState: (DirectorUiState) -> Unit
) {
    private val generation = AtomicLong(0L)
    private val faceExpert = YuNetFaceExpert()
    private val captureBridge = SierraCaptureBridge(camera)
    private val safeBurstExecutor = SafeBurstExecutor(camera, storage)
    private val computationalRaw = ComputationalRawPipeline(camera, storage)

    private data class ApprovedState(
        val look: LookSpec,
        val plan: PlanOutput,
        val score: Double,
        val mode: SierraMode,
        val captureProgram: CaptureProgram?
    )

    @Volatile private var approved: ApprovedState? = null
    @Volatile private var cloudEnabled: Boolean = true
    @Volatile private var aspectRatioOverride: String? = null
    @Volatile private var keepOriginal: Boolean = false

    fun isPrepared(): Boolean = approved != null
    fun setCloudEnabled(enabled: Boolean) { cloudEnabled = enabled; if (!enabled) vlm.cancel() }
    fun setAspectRatioOverride(value: String?) { aspectRatioOverride = value?.takeIf(SierraCropPolicy::isSupported); approved = null }
    fun setKeepOriginal(enabled: Boolean) { keepOriginal = enabled }
    suspend fun recoverPendingReferences(): List<Uri> = storage.recoverPendingToGallery()

    fun cancelForUserAction(): Long {
        approved = null
        val next = generation.incrementAndGet()
        vlm.cancel()
        return next
    }

    suspend fun onPrimaryPressed(): Uri? {
        if (approved != null) return captureFinal()
        val completed = withTimeoutOrNull(HARD_PREPARE_DEADLINE_MS) { prepareShot(); true } ?: false
        if (!completed && approved == null) {
            vlm.cancel()
            runCatching { tools.restoreUserState() }
            val session = generation.incrementAndGet()
            emit(session, DirectorUiState(stage = DirectorStage.READY, aiAvailable = cloudEnabled && vlm.isReady(), sierraMode = SierraMode.BETA, status = "Sierra beta ready • preparation timed out", camera = camera.uiState(), recipeCount = recipeMemory.count(), aspectRatioOverride = aspectRatioOverride, keepOriginal = keepOriginal))
        }
        return null
    }

    private suspend fun prepareShot(): Uri? {
        val session = cancelForUserAction()
        val created = mutableListOf<File>()
        var planningBitmap: Bitmap? = null
        try {
            tools.restoreUserState()
            val thermalPolicy = thermal.policy()
            val cloudAllowed = cloudEnabled && vlm.isReady() && thermalPolicy.allowAi
            emit(session, DirectorUiState(DirectorStage.ANALYZING, cloudAllowed, if (cloudAllowed) "Sierra is reading the scene…" else "Sierra beta is reading the scene…", camera = camera.uiState(), sierraMode = if (cloudAllowed) SierraMode.AI else SierraMode.BETA, aspectRatioOverride = aspectRatioOverride, keepOriginal = keepOriginal))

            planningBitmap = camera.captureAnalysisFrame(FAST_FRAME_TIMEOUT_MS)
            if (planningBitmap == null) {
                emit(session, DirectorUiState(DirectorStage.CAPTURING, cloudAllowed, "Sampling camera…", camera = camera.uiState(), sierraMode = if (cloudAllowed) SierraMode.AI else SierraMode.BETA, aspectRatioOverride = aspectRatioOverride, keepOriginal = keepOriginal))
                val fallback = storage.newAttempt(session, 0).also { created += it }
                camera.capturePrivate(fallback)
                assertCurrent(session)
                planningBitmap = storage.decodeOriented(fallback, 896)
            }

            val frame = checkNotNull(planningBitmap)
            val referenceQuality = PhotoQuality.analyze(frame)
            if (referenceQuality.catastrophic) {
                emit(session, DirectorUiState(DirectorStage.READY, cloudAllowed, "Scene too dark or blank • reframe", camera = camera.uiState(), sierraMode = SierraMode.BETA, aspectRatioOverride = aspectRatioOverride, keepOriginal = keepOriginal))
                return null
            }

            val planningCaps = camera.capabilities()
            val expertFaces = withTimeoutOrNull(LOCAL_FACE_BUDGET_MS) {
                faceExpert.detect(frame).getOrNull()
            }
            val sceneSignals = SierraSceneAnalyzer.analyze(frame, referenceQuality, planningCaps.lensFacing, expertFaces)

            var mode = SierraMode.BETA
            val localPlan by lazy { SierraLocalPlanner.plan(referenceQuality, frame.width, frame.height, sceneSignals, planningCaps) }
            val basePlan = if (cloudAllowed) {
                emit(session, DirectorUiState(DirectorStage.ANALYZING, true, "Sierra AI is composing…", camera = camera.uiState(), sierraMode = SierraMode.AI, aspectRatioOverride = aspectRatioOverride, keepOriginal = keepOriginal))
                vlm.setMaxImageDimension(thermalPolicy.maxAiDimension)
                val cloudResult = withTimeoutOrNull(CLOUD_PLAN_BUDGET_MS) {
                    vlm.plan(frame, planningCaps, camera.snapshot(), skills.planningContext())
                }
                val cloudPlan = cloudResult?.getOrNull()
                if (cloudPlan != null) {
                    mode = SierraMode.AI
                    val reconciled = SierraPlanReconciler.reconcile(cloudPlan, localPlan, sceneSignals)
                    consumeCompositionIntent(reconciled, localPlan)
                } else {
                    vlm.cancel()
                    emit(session, DirectorUiState(DirectorStage.PLANNING, false, "Sierra beta is taking over…", camera = camera.uiState(), sierraMode = SierraMode.BETA, aspectRatioOverride = aspectRatioOverride, keepOriginal = keepOriginal))
                    localPlan
                }
            } else localPlan
            assertCurrent(session)

            val userConstrainedPlan = aspectRatioOverride?.let { basePlan.copy(aspectRatio = it) } ?: basePlan
            val plan = InstantFinishEngine.enrich(userConstrainedPlan, referenceQuality)
            emit(session, DirectorUiState(DirectorStage.APPLYING, mode == SierraMode.AI, if (mode == SierraMode.AI) "Sierra AI directed • Sierra beta applying" else "Sierra beta preparing the shot", camera = camera.uiState(), sierraMode = mode, aspectRatio = plan.aspectRatio, aspectRatioOverride = aspectRatioOverride, sceneLabel = plan.scene, objective = plan.objective, keepOriginal = keepOriginal))

            val execution = tools.apply(plan.toolCalls, planningCaps)
            assertCurrent(session)
            val look = execution.look
            val appliedPlan = plan.copy(toolCalls = execution.applied)

            if (execution.requiresRecapture) {
                emit(session, DirectorUiState(DirectorStage.CHECKING, mode == SierraMode.AI, "Locking focus & exposure…", camera = camera.uiState(), sierraMode = mode, aspectRatio = plan.aspectRatio, aspectRatioOverride = aspectRatioOverride, sceneLabel = plan.scene, objective = plan.objective, keepOriginal = keepOriginal))
                camera.awaitSettledState(LOCAL_SETTLE_TIMEOUT_MS)
                assertCurrent(session)
            }

            emit(session, DirectorUiState(DirectorStage.CHECKING, mode == SierraMode.AI, "Sierra beta checking quality…", camera = camera.uiState(), sierraMode = mode, aspectRatio = plan.aspectRatio, aspectRatioOverride = aspectRatioOverride, sceneLabel = plan.scene, objective = plan.objective, keepOriginal = keepOriginal))
            val verification = camera.captureAnalysisFrame(FAST_VERIFY_TIMEOUT_MS)
            var finalObservedQuality: TechnicalQuality = referenceQuality
            val score = if (verification != null) {
                try {
                    val cameraQuality = PhotoQuality.analyze(verification)
                    finalObservedQuality = cameraQuality
                    if (cameraQuality.catastrophic) {
                        tools.restoreUserState()
                        emit(session, DirectorUiState(DirectorStage.READY, mode == SierraMode.AI, "Hold steady or add light • try again", camera = camera.uiState(), sierraMode = mode, aspectRatioOverride = aspectRatioOverride, keepOriginal = keepOriginal))
                        return null
                    }
                    val rendered = LookProcessor.apply(verification, look)
                    try {
                        val finishQuality = PhotoQuality.analyze(rendered)
                        combinedReadyScore(plan.referenceScore, cameraQuality.heuristicScore, finishQuality.heuristicScore)
                    } finally { if (rendered !== verification && !rendered.isRecycled) rendered.recycle() }
                } finally { if (!verification.isRecycled) verification.recycle() }
            } else CapturePolicy.combinedFallbackScore(plan.referenceScore, referenceQuality.heuristicScore)

            val captureProgram = if (camera.snapshot().extension == DirectorExtension.NONE) {
                withTimeoutOrNull(CAPTURE_PLAN_BUDGET_MS) {
                    captureBridge.plan(finalObservedQuality, sceneSignals, thermalPolicy).getOrNull()
                }
            } else null

            approve(session, appliedPlan, look, score, mode, captureProgram)
            return null
        } catch (_: StaleSession) {
            runCatching { tools.restoreUserState() }
            return null
        } catch (t: CancellationException) {
            runCatching { tools.restoreUserState() }
            throw t
        } catch (t: Throwable) {
            runCatching { tools.restoreUserState() }
            emit(session, DirectorUiState(DirectorStage.ERROR, false, "Sierra could not prepare this shot", error = t.message, camera = camera.uiState(), sierraMode = SierraMode.BETA, aspectRatioOverride = aspectRatioOverride, keepOriginal = keepOriginal))
            return null
        } finally {
            planningBitmap?.let { if (!it.isRecycled) it.recycle() }
            storage.cleanup(created)
        }
    }

    private fun consumeCompositionIntent(ai: PlanOutput, local: PlanOutput): PlanOutput {
        val composition = ai.toolCalls.lastOrNull { it.id == ToolIds.COMPOSITION }
        val requestedRatio = composition?.stringValue?.trim().orEmpty()
        val ratio = when {
            SierraCropPolicy.isSupported(requestedRatio) -> requestedRatio
            SierraCropPolicy.isSupported(ai.aspectRatio) && ai.aspectRatio != "4:3" -> ai.aspectRatio
            else -> local.aspectRatio
        }
        val centerX = (composition?.x ?: ai.cropCenterX).coerceIn(0.08, 0.92)
        val centerY = (composition?.y ?: ai.cropCenterY).coerceIn(0.08, 0.92)
        return ai.copy(aspectRatio = ratio, cropCenterX = centerX, cropCenterY = centerY, toolCalls = ai.toolCalls.filterNot { it.id == ToolIds.COMPOSITION }, planner = "sierra_ai")
    }

    private fun combinedReadyScore(ai: Double, camera: Double, finish: Double): Double = (0.48 * ai.coerceIn(0.0, 1.0) + 0.34 * camera.coerceIn(0.0, 1.0) + 0.18 * finish.coerceIn(0.0, 1.0)).coerceIn(0.0, 1.0)

    private fun approve(session: Long, plan: PlanOutput, look: LookSpec, score: Double, mode: SierraMode, captureProgram: CaptureProgram?) {
        approved = ApprovedState(look, plan, score, mode, captureProgram)
        val captureSuffix = captureProgram?.let {
            when {
                it.path == CapturePath.RAW_CUSTOM && camera.supportsComputationalRawBurst() -> " • ${it.frames.size}f computational RAW"
                it.frames.size > 1 -> " • ${it.frames.size}f smart burst"
                else -> ""
            }
        }.orEmpty()
        emit(session, DirectorUiState(DirectorStage.PREPARED, mode == SierraMode.AI, if (mode == SierraMode.AI) "Ready • Sierra AI + beta$captureSuffix" else "Ready • Sierra beta$captureSuffix", lastScore = score, camera = camera.uiState(), recipeCount = recipeMemory.count(), sierraMode = mode, aspectRatio = plan.aspectRatio, aspectRatioOverride = aspectRatioOverride, sceneLabel = plan.scene, objective = plan.objective, keepOriginal = keepOriginal))
    }

    private suspend fun captureFinal(): Uri? {
        val state = approved ?: return null
        val session = generation.get()
        var burst: SafeBurstExecutor.Result? = null
        var rawOutcome: ComputationalRawOutcome? = null
        return try {
            val activeExtension = camera.snapshot().extension
            val executableProgram = state.captureProgram.takeIf { activeExtension == DirectorExtension.NONE }
            val rawEligible = executableProgram?.path == CapturePath.RAW_CUSTOM && camera.supportsComputationalRawBurst()
            val plannedFrames = executableProgram?.frames?.size ?: 1
            val captureStatus = when {
                activeExtension != DirectorExtension.NONE -> "Capturing OEM computational frame…"
                rawEligible -> "Capturing Sierra RAW burst…"
                plannedFrames > 1 -> "Capturing smart burst…"
                else -> "Capturing full quality…"
            }
            emit(session, DirectorUiState(DirectorStage.FINAL_CAPTURING, state.mode == SierraMode.AI, captureStatus, lastScore = state.score, camera = camera.uiState(), sierraMode = state.mode, aspectRatio = state.plan.aspectRatio, aspectRatioOverride = aspectRatioOverride, sceneLabel = state.plan.scene, objective = state.plan.objective, keepOriginal = keepOriginal))
            camera.awaitSettledState(550)

            if (rawEligible && executableProgram != null) {
                val attempt = computationalRaw.execute(session, executableProgram).getOrNull()
                if (attempt != null) {
                    rawOutcome = attempt
                    assertCurrent(session)
                    return finalizeRawOutcome(session, state, attempt)
                }
            }

            val burstResult = safeBurstExecutor.execute(
                sessionId = session,
                program = executableProgram,
                maxFrames = if (activeExtension == DirectorExtension.NONE) thermal.policy().maxSierraBurstFrames else 1
            )
            burst = burstResult
            assertCurrent(session)

            val selected = burstResult.selected.file
            val captureScore = combinedReadyScore(state.score, burstResult.selected.quality.heuristicScore, burstResult.selected.score)
            val finishingStatus = when {
                burstResult.allCatastrophic -> "Technical gate failed • preserving untouched capture…"
                burstResult.truncatedByFailure -> "Burst shortened safely • finishing best frame…"
                burstResult.executedCount > 1 -> "Sierra selected best frame • finishing…"
                else -> "Sierra beta finishing…"
            }
            emit(session, DirectorUiState(DirectorStage.SAVING, state.mode == SierraMode.AI, finishingStatus, lastScore = captureScore, camera = camera.uiState(), sierraMode = state.mode, aspectRatio = state.plan.aspectRatio, aspectRatioOverride = aspectRatioOverride, sceneLabel = state.plan.scene, objective = state.plan.objective, keepOriginal = keepOriginal))

            if (keepOriginal) runCatching { storage.saveOriginal(selected) }
            val uri = if (burstResult.allCatastrophic) storage.saveRawJpeg(selected)
            else storage.saveFinal(selected, state.look, state.plan.aspectRatio, state.plan.cropCenterX, state.plan.cropCenterY)
            storage.markFinalized(burstResult.protectedReference.file)
            recipeMemory.remember(state.plan, captureScore)
            approved = null
            tools.restoreUserState()
            val savedStatus = when {
                burstResult.allCatastrophic -> "Sierra saved the untouched safety frame"
                burstResult.truncatedByFailure -> "Sierra photo saved • burst shortened safely"
                burstResult.executedCount > 1 -> "Sierra photo saved • best of ${burstResult.executedCount} frames"
                else -> "Sierra photo saved"
            }
            emit(session, DirectorUiState(DirectorStage.READY, cloudEnabled && vlm.isReady(), savedStatus, lastScore = captureScore, camera = camera.uiState(), lastSavedUri = uri.toString(), recipeCount = recipeMemory.count(), sierraMode = state.mode, aspectRatio = state.plan.aspectRatio, aspectRatioOverride = aspectRatioOverride, sceneLabel = state.plan.scene, objective = state.plan.objective, keepOriginal = keepOriginal))
            uri
        } catch (t: Throwable) {
            approved = null
            val fallbackFile = rawOutcome?.safetyJpeg ?: burst?.selected?.file
            val fallbackUri = fallbackFile?.takeIf { it.isFile && it.length() > 0L }?.let { file ->
                runCatching { storage.saveRawJpeg(file) }.getOrNull()
            }
            if (fallbackUri != null && fallbackFile != null) storage.markFinalized(fallbackFile)
            runCatching { tools.restoreUserState() }
            if (fallbackUri != null) {
                emit(session, DirectorUiState(DirectorStage.READY, cloudEnabled && vlm.isReady(), "Sierra processing failed • untouched capture saved", error = t.message, camera = camera.uiState(), lastSavedUri = fallbackUri.toString(), sierraMode = state.mode, aspectRatioOverride = aspectRatioOverride, keepOriginal = keepOriginal))
                fallbackUri
            } else {
                emit(session, DirectorUiState(DirectorStage.ERROR, cloudEnabled && vlm.isReady(), "Final capture failed", error = t.message, camera = camera.uiState(), sierraMode = state.mode, aspectRatioOverride = aspectRatioOverride, keepOriginal = keepOriginal))
                null
            }
        } finally {
            rawOutcome?.close()
            burst?.let { result -> storage.cleanup(result.frames.map { it.file }) }
        }
    }

    private suspend fun finalizeRawOutcome(
        session: Long,
        state: ApprovedState,
        outcome: ComputationalRawOutcome
    ): Uri {
        return when (outcome) {
            is ComputationalRawResult -> {
                val useComputational = outcome.acceptedOverSafetyJpeg
                val captureScore = combinedReadyScore(
                    state.score,
                    if (useComputational) outcome.computationalQuality.heuristicScore else outcome.safetyQuality.heuristicScore,
                    if (useComputational) outcome.reconstruction.meanConfidence else outcome.safetyQuality.heuristicScore
                )
                val status = if (useComputational) {
                    "RAW fusion passed evidence gate • rendering…"
                } else {
                    "RAW fusion rejected (${outcome.rejectionReason}) • preserving same-capture JPEG…"
                }
                emit(session, DirectorUiState(DirectorStage.SAVING, state.mode == SierraMode.AI, status, lastScore = captureScore, camera = camera.uiState(), sierraMode = state.mode, aspectRatio = state.plan.aspectRatio, aspectRatioOverride = aspectRatioOverride, sceneLabel = state.plan.scene, objective = state.plan.objective, keepOriginal = keepOriginal))
                if (keepOriginal) runCatching { storage.saveOriginal(outcome.safetyJpeg) }
                val uri = if (useComputational) {
                    storage.saveComputational(outcome.bitmap, outcome.safetyJpeg, state.look, state.plan.aspectRatio, state.plan.cropCenterX, state.plan.cropCenterY)
                } else {
                    storage.saveFinal(outcome.safetyJpeg, state.look, state.plan.aspectRatio, state.plan.cropCenterX, state.plan.cropCenterY)
                }
                storage.markFinalized(outcome.safetyJpeg)
                recipeMemory.remember(state.plan, captureScore)
                approved = null
                tools.restoreUserState()
                emit(session, DirectorUiState(DirectorStage.READY, cloudEnabled && vlm.isReady(), if (useComputational) "Sierra computational RAW photo saved" else "Sierra safety JPEG saved", lastScore = captureScore, camera = camera.uiState(), lastSavedUri = uri.toString(), recipeCount = recipeMemory.count(), sierraMode = state.mode, aspectRatio = state.plan.aspectRatio, aspectRatioOverride = aspectRatioOverride, sceneLabel = state.plan.scene, objective = state.plan.objective, keepOriginal = keepOriginal))
                uri
            }
            is ComputationalRawSafetyOnly -> {
                val captureScore = combinedReadyScore(state.score, outcome.safetyQuality?.heuristicScore ?: state.score, outcome.safetyQuality?.heuristicScore ?: state.score)
                emit(session, DirectorUiState(DirectorStage.SAVING, state.mode == SierraMode.AI, "RAW reconstruction unavailable • preserving same-capture JPEG…", lastScore = captureScore, camera = camera.uiState(), sierraMode = state.mode, aspectRatio = state.plan.aspectRatio, aspectRatioOverride = aspectRatioOverride, sceneLabel = state.plan.scene, objective = state.plan.objective, keepOriginal = keepOriginal))
                if (keepOriginal) runCatching { storage.saveOriginal(outcome.safetyJpeg) }
                val uri = storage.saveFinal(outcome.safetyJpeg, state.look, state.plan.aspectRatio, state.plan.cropCenterX, state.plan.cropCenterY)
                storage.markFinalized(outcome.safetyJpeg)
                recipeMemory.remember(state.plan, captureScore)
                approved = null
                tools.restoreUserState()
                emit(session, DirectorUiState(DirectorStage.READY, cloudEnabled && vlm.isReady(), "Sierra photo saved • RAW fallback ${outcome.reason.take(80)}", lastScore = captureScore, camera = camera.uiState(), lastSavedUri = uri.toString(), recipeCount = recipeMemory.count(), sierraMode = state.mode, aspectRatio = state.plan.aspectRatio, aspectRatioOverride = aspectRatioOverride, sceneLabel = state.plan.scene, objective = state.plan.objective, keepOriginal = keepOriginal))
                uri
            }
        }
    }

    private fun emit(session: Long, state: DirectorUiState) { if (isCurrent(session)) onState(state) }
    private fun isCurrent(session: Long): Boolean = generation.get() == session
    private fun assertCurrent(session: Long) { if (!isCurrent(session)) throw StaleSession() }
    private class StaleSession : RuntimeException()

    companion object {
        private const val HARD_PREPARE_DEADLINE_MS = 6_500L
        private const val CLOUD_PLAN_BUDGET_MS = 3_400L
        private const val FAST_FRAME_TIMEOUT_MS = 650L
        private const val FAST_VERIFY_TIMEOUT_MS = 500L
        private const val LOCAL_SETTLE_TIMEOUT_MS = 650L
        private const val LOCAL_FACE_BUDGET_MS = 220L
        private const val CAPTURE_PLAN_BUDGET_MS = 120L
    }
}
