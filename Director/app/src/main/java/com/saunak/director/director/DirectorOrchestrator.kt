package com.saunak.director.director

import android.graphics.Bitmap
import android.graphics.Canvas
import android.net.Uri
import com.saunak.director.ai.VlmRuntime
import com.saunak.director.camera.CameraEngine
import com.saunak.director.camera.DirectorExtension
import com.saunak.director.camera.SierraCaptureBridge
import com.saunak.director.engine.capture.CaptureProgram
import com.saunak.director.engine.capture.SafeBurstExecutor
import com.saunak.director.engine.cognition.CognitionDecision
import com.saunak.director.engine.device.CapturePath
import com.saunak.director.engine.prif.PrifLiveCaptureExecutor
import com.saunak.director.platform.ThermalMonitor
import com.saunak.director.processing.InstantFinishEngine
import com.saunak.director.processing.LookProcessor
import com.saunak.director.processing.LookSpec
import com.saunak.director.processing.PhotoQuality
import com.saunak.director.processing.TechnicalQuality
import com.saunak.director.skills.SkillEngine
import com.saunak.director.storage.AnipPhotoFinish
import com.saunak.director.storage.PhotoRepository
import com.saunak.director.storage.RecipeMemory
import com.saunak.director.storage.SierraCropPolicy
import com.saunak.director.storage.saveAnipFinal
import com.saunak.director.storage.savePrifReconstruction
import com.saunak.director.tools.ToolExecutor
import com.saunak.director.tools.ToolIds
import java.io.File
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.math.max

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
    private val perception = SierraPerceptionFabric()
    private val captureBridge = SierraCaptureBridge(camera)
    private val safeBurstExecutor = SafeBurstExecutor(camera, storage)
    private val prifExecutor = PrifLiveCaptureExecutor(camera)

    private data class ApprovedState(
        val look: LookSpec,
        val plan: PlanOutput,
        val score: Double,
        val mode: SierraMode,
        val captureProgram: CaptureProgram?
    )

    private data class ReviewResult(val adjustment: FinalRenderAdjustment?, val status: String)

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
        val completed = if (cloudEnabled && vlm.isReady() && vlm.waitForSemanticPlan()) {
            prepareShot(); true
        } else {
            withTimeoutOrNull(HARD_PREPARE_DEADLINE_MS) { prepareShot(); true } ?: false
        }
        if (!completed && approved == null) {
            vlm.cancel()
            runCatching { tools.restoreUserState() }
            val session = generation.incrementAndGet()
            emit(session, DirectorUiState(stage = DirectorStage.READY, aiAvailable = cloudEnabled && vlm.isReady(), sierraMode = SierraMode.BETA,
                status = "Sierra ready • preparation deadline reached", camera = camera.uiState(), recipeCount = recipeMemory.count(),
                aspectRatioOverride = aspectRatioOverride, keepOriginal = keepOriginal, aiFallbackReason = "prepare_timeout"))
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
            emit(session, DirectorUiState(DirectorStage.ANALYZING, cloudAllowed,
                if (cloudAllowed) "Sierra reading light, subject and scene…" else "Sierra local photography engine reading the scene…",
                camera = camera.uiState(), sierraMode = if (cloudAllowed) SierraMode.AI else SierraMode.BETA,
                aspectRatioOverride = aspectRatioOverride, keepOriginal = keepOriginal))

            planningBitmap = camera.captureAnalysisFrame(FAST_FRAME_TIMEOUT_MS)
            if (planningBitmap == null) {
                emit(session, DirectorUiState(DirectorStage.CAPTURING, cloudAllowed, "Sierra sampling camera…", camera = camera.uiState(),
                    sierraMode = if (cloudAllowed) SierraMode.AI else SierraMode.BETA, aspectRatioOverride = aspectRatioOverride, keepOriginal = keepOriginal))
                val fallback = storage.newAttempt(session, 0).also { created += it }
                camera.capturePrivate(fallback)
                assertCurrent(session)
                planningBitmap = storage.decodeOriented(fallback, 896)
            }

            val frame = checkNotNull(planningBitmap)
            val referenceQuality = PhotoQuality.analyze(frame)
            if (referenceQuality.catastrophic) {
                emit(session, DirectorUiState(DirectorStage.READY, cloudAllowed, "Scene too dark or blank • reframe", camera = camera.uiState(),
                    sierraMode = SierraMode.BETA, aspectRatioOverride = aspectRatioOverride, keepOriginal = keepOriginal))
                return null
            }

            val planningCaps = camera.capabilities()
            val snapshot = camera.snapshot()
            val perceptionSnapshot = withTimeoutOrNull(LOCAL_PERCEPTION_BUDGET_MS) {
                perception.analyze(frame, referenceQuality, planningCaps.lensFacing, snapshot.zoomRatio, thermalPolicy.thermalPressure)
            }
            val sceneSignals = perceptionSnapshot?.signals ?: SierraSceneAnalyzer.analyze(frame, referenceQuality, planningCaps.lensFacing)
            val localPlan by lazy { SierraLocalPlanner.plan(referenceQuality, frame.width, frame.height, sceneSignals, planningCaps) }

            var mode = SierraMode.BETA
            var aiFallbackReason: String? = null
            val (planned, cognition) = coroutineScope {
                val cognitionDeferred = async(Dispatchers.Default) {
                    withTimeoutOrNull(COGNITION_BUDGET_MS) { captureBridge.evaluateCognition(referenceQuality, sceneSignals).getOrNull() }
                }
                val semanticPlan = if (cloudAllowed) {
                    emit(session, DirectorUiState(DirectorStage.ANALYZING, true, "Sierra composing the photograph…", camera = camera.uiState(),
                        sierraMode = SierraMode.AI, aspectRatioOverride = aspectRatioOverride, keepOriginal = keepOriginal))
                    vlm.setMaxImageDimension(if (vlm.waitForSemanticPlan()) 512 else thermalPolicy.maxAiDimension)
                    val result = if (vlm.waitForSemanticPlan()) {
                        vlm.plan(frame, planningCaps, camera.snapshot(), skills.planningContext())
                    } else {
                        withTimeoutOrNull(CLOUD_PLAN_BUDGET_MS) { vlm.plan(frame, planningCaps, camera.snapshot(), skills.planningContext()) }
                            ?: Result.failure(IllegalStateException("Sierra semantic planning deadline"))
                    }
                    val cloudPlan = result.getOrNull()
                    if (cloudPlan != null) {
                        mode = SierraMode.AI
                        consumeCompositionIntent(SierraPlanReconciler.reconcile(cloudPlan, localPlan, sceneSignals), localPlan)
                    } else {
                        aiFallbackReason = result.exceptionOrNull()?.message?.take(120) ?: "Sierra semantic plan unavailable"
                        emit(session, DirectorUiState(DirectorStage.PLANNING, false, "Sierra continuing locally • no interruption", camera = camera.uiState(),
                            sierraMode = SierraMode.BETA, aspectRatioOverride = aspectRatioOverride, keepOriginal = keepOriginal,
                            aiLatencyMs = vlm.diagnostics().latencyMs, aiFallbackReason = aiFallbackReason))
                        localPlan
                    }
                } else localPlan
                semanticPlan to cognitionDeferred.await()
            }
            assertCurrent(session)

            val cognitionPlan = mergeCognitionDirection(planned, cognition)
            val userConstrainedPlan = aspectRatioOverride?.let { cognitionPlan.copy(aspectRatio = it) } ?: cognitionPlan
            val plan = InstantFinishEngine.enrich(userConstrainedPlan, referenceQuality)
            val aiDiag = vlm.diagnostics()
            emit(session, DirectorUiState(DirectorStage.APPLYING, mode == SierraMode.AI,
                if (mode == SierraMode.AI) "Sierra direction locked • applying safely" else "Sierra preparing the shot",
                camera = camera.uiState(), sierraMode = mode, aspectRatio = plan.aspectRatio, aspectRatioOverride = aspectRatioOverride,
                sceneLabel = plan.scene, objective = plan.objective, keepOriginal = keepOriginal,
                aiProvider = if (mode == SierraMode.AI) "Sierra Intelligence" else null,
                aiLatencyMs = if (mode == SierraMode.AI) aiDiag.latencyMs else null,
                aiTheme = plan.photographicDirection.selectedThemeId.takeIf(String::isNotBlank),
                aiThemeStrength = plan.photographicDirection.themeStrength, aiFallbackReason = aiFallbackReason))

            val execution = tools.apply(plan.toolCalls, planningCaps)
            assertCurrent(session)
            val look = execution.look
            val appliedPlan = plan.copy(toolCalls = execution.applied)

            if (execution.requiresRecapture) {
                emit(session, DirectorUiState(DirectorStage.CHECKING, mode == SierraMode.AI, "Sierra locking focus & exposure…", camera = camera.uiState(),
                    sierraMode = mode, aspectRatio = plan.aspectRatio, aspectRatioOverride = aspectRatioOverride, sceneLabel = plan.scene,
                    objective = plan.objective, keepOriginal = keepOriginal, aiTheme = plan.photographicDirection.selectedThemeId))
                camera.awaitSettledState(LOCAL_SETTLE_TIMEOUT_MS)
                assertCurrent(session)
            }

            emit(session, DirectorUiState(DirectorStage.CHECKING, mode == SierraMode.AI, "Sierra checking capture integrity…", camera = camera.uiState(),
                sierraMode = mode, aspectRatio = plan.aspectRatio, aspectRatioOverride = aspectRatioOverride, sceneLabel = plan.scene,
                objective = plan.objective, keepOriginal = keepOriginal, aiTheme = plan.photographicDirection.selectedThemeId))
            val verification = camera.captureAnalysisFrame(FAST_VERIFY_TIMEOUT_MS)
            var finalObservedQuality: TechnicalQuality = referenceQuality
            val score = if (verification != null) {
                try {
                    val cameraQuality = PhotoQuality.analyze(verification)
                    finalObservedQuality = cameraQuality
                    if (cameraQuality.catastrophic) {
                        tools.restoreUserState()
                        emit(session, DirectorUiState(DirectorStage.READY, mode == SierraMode.AI, "Hold steady or add light • try again", camera = camera.uiState(),
                            sierraMode = mode, aspectRatioOverride = aspectRatioOverride, keepOriginal = keepOriginal))
                        return null
                    }
                    val rendered = LookProcessor.apply(verification, look)
                    try { combinedReadyScore(plan.referenceScore, cameraQuality.heuristicScore, PhotoQuality.analyze(rendered).heuristicScore) }
                    finally { if (rendered !== verification && !rendered.isRecycled) rendered.recycle() }
                } finally { if (!verification.isRecycled) verification.recycle() }
            } else CapturePolicy.combinedFallbackScore(plan.referenceScore, referenceQuality.heuristicScore)

            val captureProgram = if (camera.snapshot().extension == DirectorExtension.NONE) {
                withTimeoutOrNull(CAPTURE_PLAN_BUDGET_MS) { captureBridge.plan(finalObservedQuality, sceneSignals, thermalPolicy).getOrNull() }
            } else null
            approve(session, appliedPlan, look, score, mode, captureProgram, aiFallbackReason)
            return null
        } catch (_: StaleSession) {
            runCatching { tools.restoreUserState() }; return null
        } catch (t: CancellationException) {
            runCatching { tools.restoreUserState() }; throw t
        } catch (t: Throwable) {
            runCatching { tools.restoreUserState() }
            emit(session, DirectorUiState(DirectorStage.ERROR, false, "Sierra could not prepare this shot", error = t.message,
                camera = camera.uiState(), sierraMode = SierraMode.BETA, aspectRatioOverride = aspectRatioOverride,
                keepOriginal = keepOriginal, aiFallbackReason = t.message?.take(120)))
            return null
        } finally {
            planningBitmap?.let { if (!it.isRecycled) it.recycle() }
            storage.cleanup(created)
        }
    }

    private fun mergeCognitionDirection(plan: PlanOutput, cognition: CognitionDecision?): PlanOutput {
        val current = plan.photographicDirection
        if (current.selectedThemeId.isNotBlank() && current.themeConfidence >= 0.20) return plan
        val local = cognition ?: return plan
        val r = local.intent.render; val t = r.theme
        return plan.copy(photographicDirection = PhotographicDirection(
            story = plan.objective, mood = "scene-aware natural",
            primarySubject = if (local.world.primaryHumanSubject) "human subject" else local.world.sceneBranch,
            narrativeRole = if (local.world.primaryHumanSubject) "hero" else "environment",
            selectedThemeId = t.themeId, themeStrength = t.strength.toDouble(), themeConfidence = t.confidence,
            atmospherePriority = r.atmospherePreservation, skinNaturalness = r.skinNaturalness,
            highlightProtection = r.highlightProtection, shadowDepth = (1.0 - r.shadowRecovery * 0.55).coerceIn(0.0, 1.0),
            midtoneContrast = 0.58, localContrast = 0.56, colorIntensity = 0.52,
            subjectRelightEv = r.subjectRelightEv, backgroundSuppressionEv = r.backgroundSuppressionEv,
            abstain = t.abstained, reasonCodes = t.reasonCodes
        ))
    }

    private fun consumeCompositionIntent(ai: PlanOutput, local: PlanOutput): PlanOutput {
        val composition = ai.toolCalls.lastOrNull { it.id == ToolIds.COMPOSITION }
        val requestedRatio = composition?.stringValue?.trim().orEmpty()
        val ratio = when {
            SierraCropPolicy.isSupported(requestedRatio) -> requestedRatio
            SierraCropPolicy.isSupported(ai.aspectRatio) && ai.aspectRatio != "4:3" -> ai.aspectRatio
            else -> local.aspectRatio
        }
        return ai.copy(
            aspectRatio = ratio,
            cropCenterX = (composition?.x ?: ai.cropCenterX).coerceIn(0.08, 0.92),
            cropCenterY = (composition?.y ?: ai.cropCenterY).coerceIn(0.08, 0.92),
            toolCalls = ai.toolCalls.filterNot { it.id == ToolIds.COMPOSITION },
            planner = "sierra_ai"
        )
    }

    private fun combinedReadyScore(ai: Double, camera: Double, finish: Double): Double =
        (0.48 * ai.coerceIn(0.0, 1.0) + 0.34 * camera.coerceIn(0.0, 1.0) + 0.18 * finish.coerceIn(0.0, 1.0)).coerceIn(0.0, 1.0)

    private fun approve(session: Long, plan: PlanOutput, look: LookSpec, score: Double, mode: SierraMode, captureProgram: CaptureProgram?, fallback: String?) {
        approved = ApprovedState(look, plan, score, mode, captureProgram)
        val captureSuffix = captureProgram?.takeIf { it.frames.size > 1 }?.let { " • ${it.frames.size}f smart burst" }.orEmpty()
        val theme = plan.photographicDirection.selectedThemeId.takeIf(String::isNotBlank)
        val diag = vlm.diagnostics()
        emit(session, DirectorUiState(DirectorStage.PREPARED, mode == SierraMode.AI,
            if (mode == SierraMode.AI) "Ready • Sierra ${theme ?: "directed"}$captureSuffix" else "Ready • Sierra β$captureSuffix",
            lastScore = score, camera = camera.uiState(), recipeCount = recipeMemory.count(), sierraMode = mode,
            aspectRatio = plan.aspectRatio, aspectRatioOverride = aspectRatioOverride, sceneLabel = plan.scene,
            objective = plan.objective, keepOriginal = keepOriginal, aiProvider = if (mode == SierraMode.AI) "Sierra Intelligence" else null,
            aiLatencyMs = if (mode == SierraMode.AI) diag.latencyMs else null, aiTheme = theme,
            aiThemeStrength = plan.photographicDirection.themeStrength, aiFallbackReason = fallback))
    }

    private suspend fun reviewFinal(source: Bitmap, state: ApprovedState): ReviewResult {
        if (state.mode != SierraMode.AI || !cloudEnabled || !vlm.isReady() || !vlm.supportsFinalReview()) {
            return ReviewResult(null, "Sierra local critic passed")
        }
        var evidence: Bitmap? = null
        var rendered: Bitmap? = null
        var comparison: Bitmap? = null
        return try {
            val longest = max(source.width, source.height)
            evidence = if (longest > FINAL_REVIEW_SIDE_EDGE) {
                val scale = FINAL_REVIEW_SIDE_EDGE.toFloat() / longest
                Bitmap.createScaledBitmap(source, maxOf(1, (source.width * scale).toInt()), maxOf(1, (source.height * scale).toInt()), true)
            } else source
            rendered = AnipPhotoFinish.render(checkNotNull(evidence), state.look, state.plan.scene, state.plan.objective,
                0.88, state.plan.photographicDirection).bitmap

            // One visual packet, one network request: LEFT is faithful evidence; RIGHT is ANIP output.
            // Both originate from the exact selected capture/P-RIF reconstruction, so Sierra can reject
            // lost blacks, identity, material texture, text, geometry or luminous-source character.
            comparison = buildSameCaptureComparison(checkNotNull(evidence), checkNotNull(rendered))
            vlm.setMaxImageDimension(FINAL_REVIEW_UPLOAD_EDGE)
            val result = withTimeoutOrNull(FINAL_REVIEW_BUDGET_MS) {
                vlm.evaluate(checkNotNull(comparison), state.plan.objective, 1, camera.snapshot(), emptyList(),
                    skills.evaluationContext(state.plan.skillBranch), camera.capabilities())
            }
            if (result == null) return ReviewResult(null, "Sierra local critic passed • semantic review deadline")
            val review = result.getOrNull()
            when {
                review == null -> ReviewResult(null, "Sierra local critic passed")
                review.accept -> ReviewResult(null, "Sierra critic passed ${"%.2f".format(java.util.Locale.US, review.score)}")
                else -> ReviewResult(review.renderAdjustment, "Sierra refined • ${review.issues.firstOrNull() ?: "intent alignment"}")
            }
        } finally {
            comparison?.let { if (!it.isRecycled) it.recycle() }
            rendered?.let { if (it !== evidence && !it.isRecycled) it.recycle() }
            evidence?.let { if (it !== source && !it.isRecycled) it.recycle() }
        }
    }

    private fun buildSameCaptureComparison(evidence: Bitmap, rendered: Bitmap): Bitmap {
        val height = max(evidence.height, rendered.height)
        val output = Bitmap.createBitmap(evidence.width + rendered.width, height, Bitmap.Config.ARGB_8888)
        Canvas(output).apply {
            drawBitmap(evidence, 0f, 0f, null)
            drawBitmap(rendered, evidence.width.toFloat(), 0f, null)
        }
        return output
    }

    private suspend fun reviewFile(file: File, state: ApprovedState): ReviewResult {
        val bitmap = storage.decodeOriented(file, FINAL_REVIEW_SIDE_EDGE)
        return try { reviewFinal(bitmap, state) } finally { if (!bitmap.isRecycled) bitmap.recycle() }
    }

    private suspend fun captureFinal(): Uri? {
        val state = approved ?: return null
        val session = generation.get()
        var burst: SafeBurstExecutor.Result? = null
        var prifSafety: File? = null
        var prifBitmap: Bitmap? = null
        return try {
            val activeExtension = camera.snapshot().extension
            val executableProgram = state.captureProgram.takeIf { activeExtension == DirectorExtension.NONE }
            val plannedFrames = executableProgram?.frames?.size ?: 1
            emit(session, DirectorUiState(DirectorStage.FINAL_CAPTURING, state.mode == SierraMode.AI,
                if (plannedFrames > 1) "Capturing Sierra smart burst…" else "Capturing full quality…",
                lastScore = state.score, camera = camera.uiState(), sierraMode = state.mode, aspectRatio = state.plan.aspectRatio,
                aspectRatioOverride = aspectRatioOverride, sceneLabel = state.plan.scene, objective = state.plan.objective,
                keepOriginal = keepOriginal, aiTheme = state.plan.photographicDirection.selectedThemeId))
            camera.awaitSettledState(550)

            if (executableProgram?.path == CapturePath.RAW_CUSTOM && camera.rawJpegCaptureSupported()) {
                val prif = runCatching { prifExecutor.execute(executableProgram, thermal.policy()) }.getOrNull()
                if (prif != null) {
                    assertCurrent(session)
                    val safetyFile = storage.newPendingReference(session)
                    withContext(Dispatchers.IO) { safetyFile.parentFile?.mkdirs(); safetyFile.writeBytes(prif.safeJpegBytes) }
                    prifSafety = safetyFile
                    if (keepOriginal) runCatching { storage.saveOriginal(safetyFile) }
                    val reconstruction = prif.reconstruction
                    if (prif.accepted && reconstruction != null) {
                        prifBitmap = reconstruction.bitmap
                        val captureScore = (0.55 * state.score + 0.45 * reconstruction.integrity.score).coerceIn(0.0, 1.0)
                        emit(session, DirectorUiState(DirectorStage.SAVING, state.mode == SierraMode.AI,
                            "P-RIF verified • Sierra reviewing the photograph…", lastScore = captureScore, camera = camera.uiState(),
                            sierraMode = state.mode, aiTheme = state.plan.photographicDirection.selectedThemeId))
                        val review = reviewFinal(reconstruction.bitmap, state)
                        val uri = storage.savePrifReconstruction(reconstruction.bitmap, safetyFile, state.look, state.plan.aspectRatio,
                            state.plan.cropCenterX, state.plan.cropCenterY, state.plan.scene, state.plan.objective,
                            reconstruction.integrity.score, state.plan.photographicDirection, review.adjustment)
                        recipeMemory.remember(state.plan, captureScore)
                        approved = null; tools.restoreUserState(); storage.markFinalized(safetyFile); prifSafety = null
                        if (!reconstruction.bitmap.isRecycled) reconstruction.bitmap.recycle(); prifBitmap = null
                        emit(session, DirectorUiState(DirectorStage.READY, cloudEnabled && vlm.isReady(), "Sierra photo saved • ${review.status}",
                            lastScore = captureScore, camera = camera.uiState(), lastSavedUri = uri.toString(), recipeCount = recipeMemory.count(),
                            sierraMode = state.mode, aspectRatio = state.plan.aspectRatio, sceneLabel = state.plan.scene, objective = state.plan.objective,
                            aiTheme = state.plan.photographicDirection.selectedThemeId, finalReviewStatus = review.status))
                        return uri
                    }

                    val captureScore = (0.78 * state.score + 0.22 * 0.55).coerceIn(0.0, 1.0)
                    val review = reviewFile(safetyFile, state)
                    val uri = storage.saveAnipFinal(safetyFile, state.look, state.plan.aspectRatio, state.plan.cropCenterX, state.plan.cropCenterY,
                        state.plan.scene, state.plan.objective, state.plan.photographicDirection, review.adjustment)
                    recipeMemory.remember(state.plan, captureScore)
                    approved = null; tools.restoreUserState(); storage.markFinalized(safetyFile); prifSafety = null
                    emit(session, DirectorUiState(DirectorStage.READY, cloudEnabled && vlm.isReady(), "Sierra photo saved • ${review.status}",
                        lastScore = captureScore, camera = camera.uiState(), lastSavedUri = uri.toString(), recipeCount = recipeMemory.count(),
                        sierraMode = state.mode, aiTheme = state.plan.photographicDirection.selectedThemeId, finalReviewStatus = review.status))
                    return uri
                }
            }

            val burstResult = safeBurstExecutor.execute(session, executableProgram,
                if (activeExtension == DirectorExtension.NONE) thermal.policy().maxSierraBurstFrames else 1)
            burst = burstResult
            assertCurrent(session)
            val selected = burstResult.selected.file
            val captureScore = combinedReadyScore(state.score, burstResult.selected.quality.heuristicScore, burstResult.selected.score)
            emit(session, DirectorUiState(DirectorStage.SAVING, state.mode == SierraMode.AI,
                if (burstResult.allCatastrophic) "Technical gate failed • preserving evidence…" else "Sierra rendering ${state.plan.photographicDirection.selectedThemeId.ifBlank { "photographic intent" }}…",
                lastScore = captureScore, camera = camera.uiState(), sierraMode = state.mode, aiTheme = state.plan.photographicDirection.selectedThemeId))
            if (keepOriginal) runCatching { storage.saveOriginal(selected) }
            val review = if (burstResult.allCatastrophic) ReviewResult(null, "technical safety frame") else reviewFile(selected, state)
            val uri = if (burstResult.allCatastrophic) storage.saveRawJpeg(selected)
                else storage.saveAnipFinal(selected, state.look, state.plan.aspectRatio, state.plan.cropCenterX, state.plan.cropCenterY,
                    state.plan.scene, state.plan.objective, state.plan.photographicDirection, review.adjustment)
            recipeMemory.remember(state.plan, captureScore)
            approved = null; tools.restoreUserState()
            val savedStatus = when {
                burstResult.allCatastrophic -> "Sierra saved untouched safety evidence"
                burstResult.truncatedByFailure -> "Sierra photo saved • burst shortened • ${review.status}"
                burstResult.executedCount > 1 -> "Sierra photo saved • best of ${burstResult.executedCount} • ${review.status}"
                else -> "Sierra photo saved • ${review.status}"
            }
            emit(session, DirectorUiState(DirectorStage.READY, cloudEnabled && vlm.isReady(), savedStatus,
                lastScore = captureScore, camera = camera.uiState(), lastSavedUri = uri.toString(), recipeCount = recipeMemory.count(),
                sierraMode = state.mode, aspectRatio = state.plan.aspectRatio, aspectRatioOverride = aspectRatioOverride,
                sceneLabel = state.plan.scene, objective = state.plan.objective, keepOriginal = keepOriginal,
                aiProvider = if (state.mode == SierraMode.AI) "Sierra Intelligence" else null,
                aiLatencyMs = vlm.diagnostics().latencyMs, aiTheme = state.plan.photographicDirection.selectedThemeId,
                aiThemeStrength = state.plan.photographicDirection.themeStrength, finalReviewStatus = review.status))
            uri
        } catch (t: Throwable) {
            approved = null
            val fallbackUri = prifSafety?.takeIf { it.isFile && it.length() > 0L }?.let { runCatching { storage.saveRawJpeg(it) }.getOrNull() }
                ?: burst?.selected?.file?.takeIf { it.isFile && it.length() > 0L }?.let { runCatching { storage.saveRawJpeg(it) }.getOrNull() }
            runCatching { tools.restoreUserState() }
            if (fallbackUri != null) {
                emit(session, DirectorUiState(DirectorStage.READY, cloudEnabled && vlm.isReady(), "Sierra finishing failed • untouched capture saved",
                    error = t.message, camera = camera.uiState(), lastSavedUri = fallbackUri.toString(), sierraMode = state.mode,
                    aiTheme = state.plan.photographicDirection.selectedThemeId, finalReviewStatus = "failed:${t.message?.take(80)}"))
                fallbackUri
            } else {
                emit(session, DirectorUiState(DirectorStage.ERROR, cloudEnabled && vlm.isReady(), "Final capture failed", error = t.message,
                    camera = camera.uiState(), sierraMode = state.mode, finalReviewStatus = "failed"))
                null
            }
        } finally {
            prifBitmap?.let { if (!it.isRecycled) it.recycle() }
            prifSafety?.let { runCatching { it.delete() } }
            burst?.let { result -> storage.cleanup(result.frames.map { it.file }) }
        }
    }

    private fun emit(session: Long, state: DirectorUiState) { if (isCurrent(session)) onState(state) }
    private fun isCurrent(session: Long): Boolean = generation.get() == session
    private fun assertCurrent(session: Long) { if (!isCurrent(session)) throw StaleSession() }
    private class StaleSession : RuntimeException()

    companion object {
        private const val HARD_PREPARE_DEADLINE_MS = 7_500L
        private const val CLOUD_PLAN_BUDGET_MS = 3_800L
        private const val FAST_FRAME_TIMEOUT_MS = 650L
        private const val FAST_VERIFY_TIMEOUT_MS = 500L
        private const val LOCAL_SETTLE_TIMEOUT_MS = 650L
        private const val LOCAL_PERCEPTION_BUDGET_MS = 520L
        private const val COGNITION_BUDGET_MS = 900L
        private const val CAPTURE_PLAN_BUDGET_MS = 550L
        private const val FINAL_REVIEW_BUDGET_MS = 2_200L
        private const val FINAL_REVIEW_SIDE_EDGE = 448
        private const val FINAL_REVIEW_UPLOAD_EDGE = 896
    }
}
