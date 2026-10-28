package com.saunak.director.director

import android.graphics.Bitmap
import android.net.Uri
import com.saunak.director.ai.DirectorSchemaCompiler
import com.saunak.director.ai.VlmRuntime
import com.saunak.director.camera.CameraEngine
import com.saunak.director.camera.CapturedStillTelemetry
import com.saunak.director.camera.DirectorExtension
import com.saunak.director.camera.SierraCaptureBridge
import com.saunak.director.director.live.LiveDirectorSessionRunner
import com.saunak.director.director.live.LiveDirectorValidator
import com.saunak.director.engine.capture.CaptureProgram
import com.saunak.director.engine.capture.SafeBurstExecutor
import com.saunak.director.engine.cognition.CognitionDecision
import com.saunak.director.engine.device.CapturePath
import com.saunak.director.engine.prif.PrifLiveCaptureExecutor
import com.saunak.director.platform.ThermalMonitor
import com.saunak.director.processing.InstantFinishEngine
import com.saunak.director.processing.PhotoQuality
import com.saunak.director.processing.TechnicalQuality
import com.saunak.director.skills.SkillEngine
import com.saunak.director.storage.PhotoRepository
import com.saunak.director.storage.RecipeMemory
import com.saunak.director.storage.SierraCropPolicy
import com.saunak.director.storage.savePrifSierraRenderFinal
import com.saunak.director.storage.saveSierraRenderFinal
import com.saunak.director.tools.CameraStateVerifier
import com.saunak.director.tools.ToolExecutionReceipt
import com.saunak.director.tools.ToolExecutionStatus
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
    private val liveDirector = LiveDirectorSessionRunner(camera, perception)
    private val cameraVerifier = CameraStateVerifier(camera)
    private val preShutterAuthority = PreShutterAuthority(camera, vlm, tools, liveDirector, cameraVerifier)

    private data class ApprovedState(
        val plan: PlanOutput,
        val score: Double,
        val mode: SierraMode,
        val captureProgram: CaptureProgram?,
        val receipts: List<ToolExecutionReceipt> = emptyList()
    )

    private data class RenderDecision(val spec: SierraRenderSpec?, val status: String)

    @Volatile private var approved: ApprovedState? = null
    @Volatile private var cloudEnabled: Boolean = true
    @Volatile private var aspectRatioOverride: String? = null
    @Volatile private var keepOriginal: Boolean = false
    @Volatile private var directorSessionActive: Boolean = false
    @Volatile private var lastUiState: DirectorUiState = DirectorUiState()

    init {
        vlm.setStreamProgressListener { progress ->
            val state = lastUiState
            if (state.sierraMode == SierraMode.AI && state.stage != DirectorStage.READY && state.stage != DirectorStage.ERROR) {
                emit(generation.get(), state.copy(status = progress.detail, aiStreamPhase = progress.phase))
            }
        }
    }

    fun isPrepared(): Boolean = approved != null && !directorSessionActive
    fun setCloudEnabled(enabled: Boolean) { cloudEnabled = enabled; if (!enabled) vlm.cancel() }
    fun setAspectRatioOverride(value: String?) { aspectRatioOverride = value?.takeIf(SierraCropPolicy::isSupported); approved = null }
    fun setKeepOriginal(enabled: Boolean) { keepOriginal = enabled }
    suspend fun recoverPendingReferences(): List<Uri> = storage.recoverPendingToGallery()

    fun cancelForUserAction(): Long {
        approved = null
        directorSessionActive = false
        val reset = liveDirector.stop()
        lastUiState = lastUiState.copy(liveDirector = reset)
        val next = generation.incrementAndGet()
        vlm.cancel()
        return next
    }

    suspend fun onPrimaryPressed(): Uri? {
        if (directorSessionActive) return null
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
            emit(session, DirectorUiState(
                stage = DirectorStage.READY, aiAvailable = cloudEnabled && vlm.isReady(), sierraMode = SierraMode.BETA,
                status = "Sierra ready • preparation deadline reached", camera = camera.uiState(), recipeCount = recipeMemory.count(),
                aspectRatioOverride = aspectRatioOverride, keepOriginal = keepOriginal, aiFallbackReason = "prepare_timeout"
            ))
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
            val cloudAllowed = cloudEnabled && vlm.isReady() && thermalPolicy.allowAi && vlm.supportsPreShutterValidation()
            emit(session, DirectorUiState(
                stage = DirectorStage.ANALYZING, aiAvailable = cloudAllowed,
                status = if (cloudAllowed) "Sierra reading light, subject and scene…" else "Sierra local photography engine reading the scene…",
                camera = camera.uiState(), sierraMode = if (cloudAllowed) SierraMode.AI else SierraMode.BETA,
                aspectRatioOverride = aspectRatioOverride, keepOriginal = keepOriginal,
                aiFallbackReason = if (!cloudAllowed && cloudEnabled && vlm.isReady()) "pre_shutter_authority_unavailable" else null
            ))

            planningBitmap = camera.captureAnalysisFrame(FAST_FRAME_TIMEOUT_MS)
            if (planningBitmap == null) {
                emit(session, lastUiState.copy(stage = DirectorStage.CAPTURING, status = "Sierra sampling camera…"))
                val fallback = storage.newAttempt(session, 0).also { created += it }
                camera.capturePrivate(fallback)
                assertCurrent(session)
                planningBitmap = storage.decodeOriented(fallback, 896)
            }

            val frame = checkNotNull(planningBitmap)
            val referenceQuality = PhotoQuality.analyze(frame)
            if (referenceQuality.catastrophic) {
                emit(session, lastUiState.copy(stage = DirectorStage.READY, status = "Scene too dark or blank • reframe", sierraMode = SierraMode.BETA))
                return null
            }

            val planningCaps = camera.capabilities()
            val snapshot = camera.snapshot()
            val perceptionSnapshot = withTimeoutOrNull(LOCAL_PERCEPTION_BUDGET_MS) {
                perception.analyze(frame, referenceQuality, planningCaps.lensFacing, snapshot.zoomRatio, thermalPolicy.thermalPressure)
            }
            val sceneSignals = perceptionSnapshot?.signals ?: SierraSceneAnalyzer.analyze(frame, referenceQuality, planningCaps.lensFacing)
            val situation = SierraSituationSignature.from(sceneSignals, referenceQuality, frame.width, frame.height)
            val localPlan by lazy { SierraLocalPlanner.plan(referenceQuality, frame.width, frame.height, sceneSignals, planningCaps) }

            var mode = SierraMode.BETA
            var aiFallbackReason: String? = null
            val (planned, cognition) = coroutineScope {
                val cognitionDeferred = async(Dispatchers.Default) {
                    withTimeoutOrNull(COGNITION_BUDGET_MS) { captureBridge.evaluateCognition(referenceQuality, sceneSignals).getOrNull() }
                }
                val semanticPlan = if (cloudAllowed) {
                    emit(session, lastUiState.copy(stage = DirectorStage.ANALYZING, sierraMode = SierraMode.AI, status = "Sierra composing the photograph…"))
                    vlm.setMaxImageDimension(if (vlm.waitForSemanticPlan()) 512 else thermalPolicy.maxAiDimension)
                    val planningContext = buildString {
                        appendLine(skills.planningContext())
                        appendLine(situation.compactDescription())
                        appendLine(recipeMemory.recentCreativeContext())
                    }
                    val result = if (vlm.waitForSemanticPlan()) {
                        vlm.plan(frame, planningCaps, camera.snapshot(), planningContext)
                    } else {
                        withTimeoutOrNull(CLOUD_PLAN_BUDGET_MS) { vlm.plan(frame, planningCaps, camera.snapshot(), planningContext) }
                            ?: Result.failure(IllegalStateException("Sierra semantic planning deadline"))
                    }
                    result.getOrNull()?.let {
                        mode = SierraMode.AI
                        consumeCompositionIntent(SierraPlanReconciler.reconcile(it, localPlan, sceneSignals))
                    } ?: run {
                        aiFallbackReason = result.exceptionOrNull()?.message?.take(120) ?: "Sierra semantic plan unavailable"
                        emit(session, lastUiState.copy(
                            stage = DirectorStage.PLANNING, aiAvailable = false, sierraMode = SierraMode.BETA,
                            status = "Sierra continuing locally • Gemini direction unavailable", aiLatencyMs = vlm.diagnostics().latencyMs,
                            aiFallbackReason = aiFallbackReason
                        ))
                        localPlan
                    }
                } else localPlan
                semanticPlan to cognitionDeferred.await()
            }
            assertCurrent(session)

            val cognitionPlan = if (mode == SierraMode.AI) planned else mergeCognitionDirection(planned, cognition)
            val userConstrainedPlan = aspectRatioOverride?.let { cognitionPlan.copy(aspectRatio = it) } ?: cognitionPlan
            val enriched = InstantFinishEngine.enrich(userConstrainedPlan, referenceQuality)
            val plan = if (mode == SierraMode.BETA) enriched.copy(
                renderSpec = SierraBetaRenderPlanner.plan(referenceQuality, sceneSignals, situation)
            ) else enriched
            val aiDiag = vlm.diagnostics()
            emit(session, lastUiState.copy(
                stage = DirectorStage.APPLYING, aiAvailable = mode == SierraMode.AI,
                status = if (mode == SierraMode.AI) "Sierra direction locked • applying" else "Sierra β measured render + camera plan locked",
                camera = camera.uiState(), sierraMode = mode, aspectRatio = plan.aspectRatio, sceneLabel = plan.scene,
                objective = plan.objective, aiProvider = if (mode == SierraMode.AI) "Sierra Intelligence" else null,
                aiLatencyMs = if (mode == SierraMode.AI) aiDiag.latencyMs else null,
                aiTheme = plan.photographicDirection.selectedThemeId.takeIf(String::isNotBlank),
                aiThemeStrength = plan.photographicDirection.themeStrength, aiFallbackReason = aiFallbackReason
            ))

            val execution = tools.apply(plan.toolCalls, planningCaps, allowPredictiveRetarget = mode != SierraMode.AI)
            assertCurrent(session)
            var receipts = execution.receipts.toMutableList()
            var appliedCalls = execution.applied
            var cameraReport = cameraVerifier.verify(appliedCalls, camera.capabilities(), INITIAL_VERIFY_SETTLE_MS)
            if (cameraReport.retryCalls.isNotEmpty()) {
                val retry = tools.apply(cameraReport.retryCalls, camera.capabilities(), allowPredictiveRetarget = false)
                receipts += retry.receipts
                appliedCalls = mergeToolCalls(appliedCalls, retry.applied)
                cameraReport = cameraVerifier.verify(appliedCalls, camera.capabilities(), INITIAL_VERIFY_SETTLE_MS)
            }
            val appliedPlan = plan.copy(toolCalls = appliedCalls)
            val failure = receipts.firstOrNull { it.status == ToolExecutionStatus.FAILED || it.status == ToolExecutionStatus.REJECTED }
            if (mode == SierraMode.AI && (failure != null || cameraReport.hardFailure)) {
                runCatching { tools.restoreUserState() }
                val reason = failure?.reason ?: "requested camera state could not be verified"
                emit(session, lastUiState.copy(
                    stage = DirectorStage.READY, status = "Sierra control unavailable • $reason", camera = camera.uiState(),
                    aiFallbackReason = reason.take(120), executionReceipts = receipts,
                    finalReviewStatus = cameraReport.compactDescription()
                ))
                return null
            }

            if (execution.requiresRecapture) {
                emit(session, lastUiState.copy(stage = DirectorStage.CHECKING, status = "Sierra locking focus & exposure…", executionReceipts = receipts, finalReviewStatus = cameraReport.compactDescription()))
                camera.awaitSettledState(LOCAL_SETTLE_TIMEOUT_MS)
                assertCurrent(session)
            }

            emit(session, lastUiState.copy(stage = DirectorStage.CHECKING, status = "Sierra checking capture integrity…", executionReceipts = receipts, finalReviewStatus = cameraReport.compactDescription()))
            val verification = camera.captureAnalysisFrame(FAST_VERIFY_TIMEOUT_MS)
            var finalObservedQuality: TechnicalQuality = referenceQuality
            val score = if (verification != null) {
                try {
                    val cameraQuality = PhotoQuality.analyze(verification)
                    finalObservedQuality = cameraQuality
                    if (cameraQuality.catastrophic) {
                        tools.restoreUserState()
                        emit(session, lastUiState.copy(stage = DirectorStage.READY, status = "Hold steady or add light • try again"))
                        return null
                    }
                    combinedReadyScore(plan.referenceScore, cameraQuality.heuristicScore, cameraQuality.heuristicScore)
                } finally { if (!verification.isRecycled) verification.recycle() }
            } else CapturePolicy.combinedFallbackScore(plan.referenceScore, referenceQuality.heuristicScore)

            val captureProgram = if (camera.snapshot().extension == DirectorExtension.NONE) {
                withTimeoutOrNull(CAPTURE_PLAN_BUDGET_MS) { captureBridge.plan(finalObservedQuality, sceneSignals, thermalPolicy).getOrNull() }
            } else null

            if (mode == SierraMode.AI && appliedPlan.directorSpec != null && appliedPlan.renderSpec != null) {
                val validated = runCatching {
                    LiveDirectorValidator.validate(appliedPlan.directorSpec, withContext(Dispatchers.IO) { DirectorSchemaCompiler().compile() })
                }
                if (validated.isFailure) {
                    val reason = validated.exceptionOrNull()?.message ?: "Invalid Director graph"
                    liveDirector.stop()
                    emit(session, lastUiState.copy(stage = DirectorStage.READY, status = "Sierra Director graph rejected safely", aiFallbackReason = reason.take(120)))
                    return null
                }

                directorSessionActive = true
                val guidance = try {
                    liveDirector.guideUntilReady(frame, validated.getOrThrow(), thermalPolicy.thermalPressure, { isCurrent(session) }) { live ->
                        emit(session, lastUiState.copy(stage = DirectorStage.CHECKING, status = live.status, liveDirector = live, executionReceipts = receipts))
                    }
                } finally { directorSessionActive = false }
                assertCurrent(session)

                if (!guidance.ready && !guidance.sceneConstraintSuspected) {
                    val reason = guidance.reason ?: "Live Director unavailable"
                    liveDirector.stop()
                    emit(session, lastUiState.copy(stage = DirectorStage.READY, status = "Sierra could not validate the live target • re-prepare the shot", aiFallbackReason = reason.take(120), liveDirector = liveDirector.latest()))
                    return null
                }

                emit(session, lastUiState.copy(
                    stage = DirectorStage.CHECKING,
                    status = if (guidance.sceneConstraintSuspected) "Sierra validating closest achievable composition…" else "Sierra performing final semantic validation…",
                    liveDirector = guidance.state, executionReceipts = receipts, finalReviewStatus = cameraReport.compactDescription()
                ))
                directorSessionActive = true
                val authority = try {
                    preShutterAuthority.validateOnce(
                        originalPlan = appliedPlan.copy(directorSpec = validated.getOrThrow()),
                        existingReceipts = receipts,
                        thermalPressure = thermalPolicy.thermalPressure,
                        isCurrent = { isCurrent(session) },
                        sceneConstraintSuspected = guidance.sceneConstraintSuspected
                    ) { live -> emit(session, lastUiState.copy(stage = DirectorStage.CHECKING, status = live.status, liveDirector = live)) }
                } finally { directorSessionActive = false }
                assertCurrent(session)
                if (!authority.ready) {
                    approved = null
                    emit(session, lastUiState.copy(stage = DirectorStage.READY, status = authority.status, aiFallbackReason = authority.reason?.take(120), executionReceipts = authority.receipts, liveDirector = authority.liveState))
                    return null
                }
                receipts = authority.receipts.toMutableList()
                val safeCaptureProgram = if (authority.corrected) null else captureProgram
                approve(session, authority.plan, score, mode, safeCaptureProgram, if (authority.corrected) "one_final_correction_applied" else aiFallbackReason, receipts)
                emit(session, lastUiState.copy(status = authority.status, liveDirector = authority.liveState, finalReviewStatus = "pre_shutter_validated"))
            } else {
                approve(session, appliedPlan, score, mode, captureProgram, aiFallbackReason, receipts)
            }
            return null
        } catch (_: StaleSession) {
            directorSessionActive = false; liveDirector.stop(); runCatching { tools.restoreUserState() }; return null
        } catch (t: CancellationException) {
            directorSessionActive = false; liveDirector.stop(); runCatching { tools.restoreUserState() }; throw t
        } catch (t: Throwable) {
            directorSessionActive = false; liveDirector.stop(); runCatching { tools.restoreUserState() }
            emit(session, lastUiState.copy(stage = DirectorStage.ERROR, aiAvailable = false, status = "Sierra could not prepare this shot", error = t.message, sierraMode = SierraMode.BETA, aiFallbackReason = t.message?.take(120)))
            return null
        } finally {
            planningBitmap?.let { if (!it.isRecycled) it.recycle() }
            storage.cleanup(created)
        }
    }

    private fun mergeToolCalls(original: List<ToolCall>, replacement: List<ToolCall>): List<ToolCall> {
        if (replacement.isEmpty()) return original
        val ids = replacement.map { it.id }.toSet()
        return (original.filterNot { it.id in ids } + replacement).take(8)
    }

    private fun mergeCognitionDirection(plan: PlanOutput, cognition: CognitionDecision?): PlanOutput {
        if (plan.planner.startsWith("sierra_ai")) return plan
        val local = cognition ?: return plan
        val r = local.intent.render
        return plan.copy(photographicDirection = PhotographicDirection(
            story = plan.objective,
            mood = "measured local",
            primarySubject = if (local.world.primaryHumanSubject) "human subject" else local.world.sceneBranch,
            narrativeRole = if (local.world.primaryHumanSubject) "hero" else "environment",
            selectedThemeId = "",
            themeStrength = 0.0,
            themeConfidence = 0.0,
            atmospherePriority = r.atmospherePreservation,
            skinNaturalness = r.skinNaturalness,
            highlightProtection = r.highlightProtection,
            shadowDepth = (1.0 - r.shadowRecovery * 0.55).coerceIn(0.0, 1.0),
            midtoneContrast = 0.58,
            localContrast = 0.56,
            colorIntensity = 0.52,
            subjectRelightEv = r.subjectRelightEv,
            backgroundSuppressionEv = r.backgroundSuppressionEv,
            abstain = true,
            reasonCodes = listOf("beta_measurement_driven")
        ))
    }

    private fun consumeCompositionIntent(ai: PlanOutput): PlanOutput {
        val composition = ai.toolCalls.lastOrNull { it.id == ToolIds.COMPOSITION }
        val requestedRatio = composition?.stringValue?.trim().orEmpty()
        val ratio = when {
            SierraCropPolicy.isSupported(requestedRatio) -> requestedRatio
            SierraCropPolicy.isSupported(ai.aspectRatio) -> ai.aspectRatio
            else -> "4:3"
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

    private fun approve(session: Long, plan: PlanOutput, score: Double, mode: SierraMode, captureProgram: CaptureProgram?, fallback: String?, receipts: List<ToolExecutionReceipt>) {
        approved = ApprovedState(plan, score, mode, captureProgram, receipts)
        val captureSuffix = captureProgram?.takeIf { it.frames.size > 1 }?.let { " • ${it.frames.size}f smart burst" }.orEmpty()
        val diag = vlm.diagnostics()
        val strategy = plan.directorSpec?.strategy?.name?.lowercase()?.replace('_', ' ')
        emit(session, lastUiState.copy(
            stage = DirectorStage.PREPARED, aiAvailable = mode == SierraMode.AI,
            status = if (mode == SierraMode.AI) "Ready • Sierra ${strategy ?: "directed"}$captureSuffix" else "Ready • Sierra β measured$captureSuffix",
            lastScore = score, camera = camera.uiState(), recipeCount = recipeMemory.count(), sierraMode = mode,
            aspectRatio = plan.aspectRatio, sceneLabel = plan.scene, objective = plan.objective,
            aiProvider = if (mode == SierraMode.AI) "Sierra Intelligence" else null,
            aiLatencyMs = if (mode == SierraMode.AI) diag.latencyMs else null,
            aiTheme = plan.photographicDirection.selectedThemeId.takeIf(String::isNotBlank),
            aiThemeStrength = plan.photographicDirection.themeStrength, aiFallbackReason = fallback,
            executionReceipts = receipts, aiStreamPhase = null
        ))
    }

    private suspend fun resolveRenderSpec(source: Bitmap, state: ApprovedState, telemetry: CapturedStillTelemetry, captureKind: String, reconstructionIntegrity: Double? = null): RenderDecision {
        if (state.mode == SierraMode.BETA) {
            return RenderDecision(state.plan.renderSpec ?: SierraRenderSpecFactory.fromDirection(state.plan.photographicDirection), "Sierra β measured deterministic render")
        }
        val spec = state.plan.renderSpec
        return if (spec != null) RenderDecision(spec, "Sierra pre-authorized Render Spec • no post-capture AI call")
        else RenderDecision(null, "Sierra Render Spec missing • captured evidence preserved")
    }

    private suspend fun resolveRenderSpecFile(file: File, state: ApprovedState, telemetry: CapturedStillTelemetry, captureKind: String): RenderDecision {
        val bitmap = storage.decodeOriented(file, FINAL_RENDER_UPLOAD_EDGE)
        return try { resolveRenderSpec(bitmap, state, telemetry, captureKind) } finally { if (!bitmap.isRecycled) bitmap.recycle() }
    }

    private suspend fun captureFinal(): Uri? {
        val state = approved ?: return null
        val session = generation.get()
        if (state.plan.directorSpec != null && liveDirector.latest().running) {
            directorSessionActive = true
            val fresh = try { liveDirector.reverify(thermal.policy().thermalPressure, { isCurrent(session) }) { live -> emit(session, lastUiState.copy(stage = DirectorStage.CHECKING, status = live.status, liveDirector = live)) } }
            finally { directorSessionActive = false }
            if (!fresh.ready) {
                assertCurrent(session)
                directorSessionActive = true
                val resumed = try { liveDirector.resumeUntilReady(thermal.policy().thermalPressure, { isCurrent(session) }) { live -> emit(session, lastUiState.copy(stage = DirectorStage.CHECKING, status = live.status, liveDirector = live)) } }
                finally { directorSessionActive = false }
                assertCurrent(session)
                if (resumed.ready) emit(session, lastUiState.copy(stage = DirectorStage.PREPARED, status = "Ready • hold there • press shutter", liveDirector = resumed.state))
                else { approved = null; emit(session, lastUiState.copy(stage = DirectorStage.READY, status = "Live Director unavailable • re-prepare the shot", aiFallbackReason = resumed.reason?.take(120), liveDirector = liveDirector.stop())) }
                return null
            }
        }

        val clearedDirector = liveDirector.stop()
        directorSessionActive = false
        var burst: SafeBurstExecutor.Result? = null
        var prifSafety: File? = null
        var prifBitmap: Bitmap? = null
        return try {
            val activeExtension = camera.snapshot().extension
            val executableProgram = state.captureProgram.takeIf { activeExtension == DirectorExtension.NONE }
            val plannedFrames = executableProgram?.frames?.size ?: 1
            emit(session, lastUiState.copy(stage = DirectorStage.FINAL_CAPTURING, status = if (plannedFrames > 1) "Capturing Sierra smart burst…" else "Capturing full quality…", lastScore = state.score, camera = camera.uiState(), sierraMode = state.mode, executionReceipts = state.receipts, liveDirector = clearedDirector))
            camera.awaitSettledState(550)

            if (executableProgram?.path == CapturePath.RAW_CUSTOM && camera.rawJpegCaptureSupported()) {
                val shutterState = camera.snapshot()
                val prif = runCatching { prifExecutor.execute(executableProgram, thermal.policy()) }.getOrNull()
                if (prif != null) {
                    assertCurrent(session)
                    val safetyFile = storage.newPendingReference(session)
                    withContext(Dispatchers.IO) { safetyFile.parentFile?.mkdirs(); safetyFile.writeBytes(prif.safeJpegBytes) }
                    prifSafety = safetyFile
                    if (keepOriginal) runCatching { storage.saveOriginal(safetyFile) }
                    val telemetry = CapturedStillTelemetry.fromJpeg(safetyFile, "$session:prif", shutterState, camera.snapshot(), camera.sensorTelemetry())
                    val reconstruction = prif.reconstruction
                    if (prif.accepted && reconstruction != null) {
                        prifBitmap = reconstruction.bitmap
                        val captureScore = (0.55 * state.score + 0.45 * reconstruction.integrity.score).coerceIn(0.0, 1.0)
                        emit(session, lastUiState.copy(stage = DirectorStage.SAVING, status = "P-RIF verified • applying Sierra render…", lastScore = captureScore))
                        val render = resolveRenderSpec(reconstruction.bitmap, state, telemetry, "prif_reconstruction", reconstruction.integrity.score)
                        val uri = if (render.spec != null) storage.savePrifSierraRenderFinal(reconstruction.bitmap, safetyFile, render.spec, state.plan.aspectRatio, state.plan.cropCenterX, state.plan.cropCenterY) else storage.saveProcessed(reconstruction.bitmap, safetyFile)
                        recipeMemory.remember(state.plan, captureScore)
                        approved = null; tools.restoreUserState(); storage.markFinalized(safetyFile); prifSafety = null
                        if (!reconstruction.bitmap.isRecycled) reconstruction.bitmap.recycle(); prifBitmap = null
                        emit(session, lastUiState.copy(stage = DirectorStage.READY, status = "Sierra photo saved • ${render.status}", lastScore = captureScore, camera = camera.uiState(), lastSavedUri = uri.toString(), recipeCount = recipeMemory.count(), finalReviewStatus = render.status, aiStreamPhase = null))
                        return uri
                    }
                    val captureScore = (0.78 * state.score + 0.22 * 0.55).coerceIn(0.0, 1.0)
                    val render = resolveRenderSpecFile(safetyFile, state, telemetry, "prif_safety_jpeg")
                    val uri = if (render.spec != null) storage.saveSierraRenderFinal(safetyFile, render.spec, state.plan.aspectRatio, state.plan.cropCenterX, state.plan.cropCenterY) else storage.saveRawJpeg(safetyFile)
                    recipeMemory.remember(state.plan, captureScore)
                    approved = null; tools.restoreUserState(); storage.markFinalized(safetyFile); prifSafety = null
                    emit(session, lastUiState.copy(stage = DirectorStage.READY, status = "Sierra photo saved • ${render.status}", lastScore = captureScore, camera = camera.uiState(), lastSavedUri = uri.toString(), recipeCount = recipeMemory.count(), finalReviewStatus = render.status, aiStreamPhase = null))
                    return uri
                }
            }

            val burstResult = safeBurstExecutor.execute(session, executableProgram, if (activeExtension == DirectorExtension.NONE) thermal.policy().maxSierraBurstFrames else 1)
            burst = burstResult
            assertCurrent(session)
            val selected = burstResult.selected.file
            val captureScore = combinedReadyScore(state.score, burstResult.selected.quality.heuristicScore, burstResult.selected.score)
            emit(session, lastUiState.copy(stage = DirectorStage.SAVING, status = if (burstResult.allCatastrophic) "Technical gate failed • preserving evidence…" else "Applying Sierra pre-authorized render…", lastScore = captureScore, camera = camera.uiState()))
            if (keepOriginal) runCatching { storage.saveOriginal(selected) }
            val render = if (burstResult.allCatastrophic) RenderDecision(null, "technical safety frame") else resolveRenderSpecFile(selected, state, burstResult.selected.telemetry, "camera_jpeg")
            val uri = if (render.spec == null) storage.saveRawJpeg(selected) else storage.saveSierraRenderFinal(selected, render.spec, state.plan.aspectRatio, state.plan.cropCenterX, state.plan.cropCenterY)
            recipeMemory.remember(state.plan, captureScore)
            approved = null; tools.restoreUserState()
            val savedStatus = when {
                burstResult.allCatastrophic -> "Sierra saved untouched safety evidence"
                burstResult.truncatedByFailure -> "Sierra photo saved • burst shortened • ${render.status}"
                burstResult.executedCount > 1 -> "Sierra photo saved • best of ${burstResult.executedCount} • ${render.status}"
                else -> "Sierra photo saved • ${render.status}"
            }
            emit(session, lastUiState.copy(stage = DirectorStage.READY, status = savedStatus, lastScore = captureScore, camera = camera.uiState(), lastSavedUri = uri.toString(), recipeCount = recipeMemory.count(), finalReviewStatus = render.status, aiLatencyMs = vlm.diagnostics().latencyMs, aiStreamPhase = null))
            uri
        } catch (t: Throwable) {
            approved = null; directorSessionActive = false; liveDirector.stop()
            val fallbackUri = prifSafety?.takeIf { it.isFile && it.length() > 0L }?.let { runCatching { storage.saveRawJpeg(it) }.getOrNull() }
                ?: burst?.selected?.file?.takeIf { it.isFile && it.length() > 0L }?.let { runCatching { storage.saveRawJpeg(it) }.getOrNull() }
            runCatching { tools.restoreUserState() }
            if (fallbackUri != null) {
                emit(session, lastUiState.copy(stage = DirectorStage.READY, status = "Sierra finishing failed • untouched capture saved", error = t.message, camera = camera.uiState(), lastSavedUri = fallbackUri.toString(), finalReviewStatus = "failed:${t.message?.take(80)}", aiStreamPhase = null))
                fallbackUri
            } else {
                emit(session, lastUiState.copy(stage = DirectorStage.ERROR, status = "Final capture failed", error = t.message, camera = camera.uiState(), finalReviewStatus = "failed", aiStreamPhase = null)); null
            }
        } finally {
            prifBitmap?.let { if (!it.isRecycled) it.recycle() }
            prifSafety?.let { runCatching { it.delete() } }
            burst?.let { result -> storage.cleanup(result.frames.map { it.file }) }
        }
    }

    private fun emit(session: Long, state: DirectorUiState) { if (isCurrent(session)) { lastUiState = state; onState(state) } }
    private fun isCurrent(session: Long): Boolean = generation.get() == session
    private fun assertCurrent(session: Long) { if (!isCurrent(session)) throw StaleSession() }
    private class StaleSession : RuntimeException()

    companion object {
        private const val HARD_PREPARE_DEADLINE_MS = 7_500L
        private const val CLOUD_PLAN_BUDGET_MS = 3_800L
        private const val FAST_FRAME_TIMEOUT_MS = 650L
        private const val FAST_VERIFY_TIMEOUT_MS = 500L
        private const val LOCAL_SETTLE_TIMEOUT_MS = 650L
        private const val INITIAL_VERIFY_SETTLE_MS = 360L
        private const val LOCAL_PERCEPTION_BUDGET_MS = 520L
        private const val COGNITION_BUDGET_MS = 900L
        private const val CAPTURE_PLAN_BUDGET_MS = 550L
        private const val FINAL_RENDER_UPLOAD_EDGE = 896
    }
}
