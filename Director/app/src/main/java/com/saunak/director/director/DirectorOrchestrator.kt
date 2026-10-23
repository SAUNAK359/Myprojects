package com.saunak.director.director

import android.graphics.Bitmap
import android.net.Uri
import com.saunak.director.ai.VlmRuntime
import com.saunak.director.camera.CameraEngine
import com.saunak.director.platform.ThermalMonitor
import com.saunak.director.processing.InstantFinishEngine
import com.saunak.director.processing.LookProcessor
import com.saunak.director.processing.LookSpec
import com.saunak.director.processing.PhotoQuality
import com.saunak.director.skills.SkillEngine
import com.saunak.director.storage.PhotoRepository
import com.saunak.director.storage.RecipeMemory
import com.saunak.director.tools.ToolExecutor
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File
import java.util.concurrent.atomic.AtomicLong

/**
 * Director V2 pipeline: one semantic VLM upload per shutter preparation.
 *
 * The cloud model performs the expensive semantic task once (scene/intent/composition/style).
 * Everything after that is deterministic/local: camera controls, finish selection and technical
 * verification. This keeps latency predictable and avoids repeatedly uploading user photos.
 */
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
    private data class ApprovedState(val look: LookSpec, val plan: PlanOutput, val score: Double)
    @Volatile private var approved: ApprovedState? = null

    fun isPrepared(): Boolean = approved != null
    suspend fun recoverPendingReferences(): List<Uri> = storage.recoverPendingToGallery()

    fun cancelForUserAction(): Long {
        approved = null
        val next = generation.incrementAndGet()
        vlm.cancel()
        return next
    }

    suspend fun onPrimaryPressed(): Uri? {
        if (approved != null) return captureFinal()
        val completed = withTimeoutOrNull(HARD_PREPARE_DEADLINE_MS) {
            prepareShot()
            true
        } ?: false
        if (!completed && approved == null) {
            vlm.cancel()
            runCatching { tools.restoreUserState() }
            val session = generation.incrementAndGet()
            emit(
                session,
                DirectorUiState(
                    stage = DirectorStage.READY,
                    aiAvailable = vlm.isReady(),
                    status = "Director took too long • switch AI route and try again",
                    camera = camera.uiState(),
                    recipeCount = recipeMemory.count()
                )
            )
        }
        return null
    }

    private suspend fun prepareShot(): Uri? {
        val session = cancelForUserAction()
        val created = mutableListOf<File>()
        var planningBitmap: Bitmap? = null
        try {
            tools.restoreUserState()
            if (!vlm.isReady()) {
                emit(session, DirectorUiState(DirectorStage.CAMERA_ONLY, false, "Configure an AI provider in Settings", camera = camera.uiState()))
                return null
            }

            val thermalPolicy = thermal.policy()
            if (!thermalPolicy.allowAi) {
                emit(session, DirectorUiState(DirectorStage.CAMERA_ONLY, true, "Director paused for device temperature", camera = camera.uiState()))
                return null
            }

            emit(session, DirectorUiState(DirectorStage.ANALYZING, true, "Reading live scene…", camera = camera.uiState()))

            // Fast path: a fresh CameraX ImageAnalysis frame. Nothing is written to disk and the
            // high-resolution still pipeline remains untouched until the user takes the final shot.
            planningBitmap = camera.captureAnalysisFrame(FAST_FRAME_TIMEOUT_MS)

            // Compatibility fallback for extension modes / limited devices that cannot bind
            // Preview + ImageCapture + ImageAnalysis together.
            if (planningBitmap == null) {
                emit(session, DirectorUiState(DirectorStage.CAPTURING, true, "Sampling camera…", camera = camera.uiState()))
                val fallback = storage.newAttempt(session, 0).also { created += it }
                camera.capturePrivate(fallback)
                assertCurrent(session)
                planningBitmap = storage.decodeOriented(fallback, 896)
            }

            val frame = checkNotNull(planningBitmap)
            val referenceQuality = PhotoQuality.analyze(frame)
            if (referenceQuality.catastrophic) {
                emit(session, DirectorUiState(DirectorStage.READY, true, "Scene too dark/blank • reframe and try again", camera = camera.uiState()))
                return null
            }

            emit(session, DirectorUiState(DirectorStage.ANALYZING, true, "Designing the shot…", camera = camera.uiState()))
            vlm.setMaxImageDimension(thermalPolicy.maxAiDimension)
            var caps = camera.capabilities()
            val cloudPlan = vlm.plan(frame, caps, camera.snapshot(), skills.planningContext()).getOrElse {
                val reason = aiFailureMessage(it)
                emit(session, DirectorUiState(DirectorStage.ERROR, true, reason, error = it.message, camera = camera.uiState()))
                return null
            }
            assertCurrent(session)

            // Custom local intelligence: guarantee a tasteful post-ready finish even when the VLM
            // omits a look, and temper model-selected strength based on real image statistics.
            val plan = InstantFinishEngine.enrich(cloudPlan, referenceQuality)

            emit(session, DirectorUiState(DirectorStage.APPLYING, true, "Building your look…", camera = camera.uiState()))
            val execution = tools.apply(plan.toolCalls, caps)
            assertCurrent(session)
            caps = camera.capabilities()
            val look = execution.look
            val appliedPlan = plan.copy(toolCalls = execution.applied)

            if (execution.requiresRecapture) {
                emit(session, DirectorUiState(DirectorStage.CHECKING, true, "Locking focus & exposure…", camera = camera.uiState()))
                camera.awaitSettledState(LOCAL_SETTLE_TIMEOUT_MS)
                assertCurrent(session)
            }

            // Local QA only: no second cloud/VLM upload. We inspect a fresh preview frame after
            // camera changes and the rendered finish for exposure, clipping and usable detail.
            emit(session, DirectorUiState(DirectorStage.CHECKING, true, "Polishing locally…", camera = camera.uiState()))
            val verification = camera.captureAnalysisFrame(FAST_VERIFY_TIMEOUT_MS)
            val score = if (verification != null) {
                try {
                    val cameraQuality = PhotoQuality.analyze(verification)
                    if (cameraQuality.catastrophic) {
                        tools.restoreUserState()
                        emit(session, DirectorUiState(DirectorStage.READY, true, "Hold steady or add light • try again", camera = camera.uiState()))
                        return null
                    }
                    val rendered = LookProcessor.apply(verification, look)
                    try {
                        val finishQuality = PhotoQuality.analyze(rendered)
                        combinedReadyScore(plan.referenceScore, cameraQuality.heuristicScore, finishQuality.heuristicScore)
                    } finally {
                        if (rendered !== verification && !rendered.isRecycled) rendered.recycle()
                    }
                } finally {
                    if (!verification.isRecycled) verification.recycle()
                }
            } else {
                CapturePolicy.combinedFallbackScore(plan.referenceScore, referenceQuality.heuristicScore)
            }

            approve(
                session,
                appliedPlan,
                look,
                score,
                if (execution.rejected.isEmpty()) "Ready • post-ready finish locked" else "Ready • best safe finish locked"
            )
            return null
        } catch (_: StaleSession) {
            return null
        } catch (t: CancellationException) {
            throw t
        } catch (t: Throwable) {
            runCatching { tools.restoreUserState() }
            emit(session, DirectorUiState(DirectorStage.ERROR, vlm.isReady(), aiFailureMessage(t), error = t.message, camera = camera.uiState()))
            return null
        } finally {
            planningBitmap?.let { if (!it.isRecycled) it.recycle() }
            storage.cleanup(created)
        }
    }

    private fun combinedReadyScore(ai: Double, camera: Double, finish: Double): Double =
        (0.48 * ai.coerceIn(0.0, 1.0) + 0.34 * camera.coerceIn(0.0, 1.0) + 0.18 * finish.coerceIn(0.0, 1.0))
            .coerceIn(0.0, 1.0)

    private fun aiFailureMessage(t: Throwable): String {
        val message = t.message.orEmpty().trim()
        val lower = message.lowercase()
        return when {
            "401" in lower || "invalid" in lower && ("key" in lower || "credential" in lower) -> "AI key invalid • update it in Settings"
            "402" in lower || "credits" in lower || "billing" in lower -> "AI endpoint requires credits or billing"
            "403" in lower || "blocked" in lower || "access denied" in lower -> "AI access blocked • check API key/project"
            "404" in lower || "model" in lower && "unavailable" in lower || "endpoint unavailable" in lower -> "AI model endpoint unavailable"
            "429" in lower || "quota" in lower || "rate limit" in lower || "throttled" in lower -> "AI rate limit reached • switch route or retry"
            "timeout" in lower || "timed out" in lower -> "AI response timed out • switch route or retry"
            "no candidate" in lower || "no choices" in lower || "no content" in lower || "empty output" in lower -> "AI returned no usable analysis • try another route"
            "non-json" in lower || "unreadable" in lower || "invalid analysis" in lower -> "AI returned an invalid analysis • try another route"
            message.isNotBlank() -> "Director AI error • ${message.take(96)}"
            else -> "Director analysis failed • try again"
        }
    }

    private fun approve(session: Long, plan: PlanOutput, look: LookSpec, score: Double, message: String = "Ready — take the shot") {
        approved = ApprovedState(look, plan, score)
        emit(
            session,
            DirectorUiState(
                DirectorStage.PREPARED,
                true,
                message,
                lastScore = score,
                camera = camera.uiState(),
                recipeCount = recipeMemory.count()
            )
        )
    }

    private suspend fun captureFinal(): Uri? {
        val state = approved ?: return null
        val session = generation.get()
        var file: File? = null
        return try {
            emit(session, DirectorUiState(DirectorStage.FINAL_CAPTURING, true, "Capturing full quality…", lastScore = state.score, camera = camera.uiState()))
            camera.awaitSettledState(550)
            file = storage.newAttempt(session, 99)
            camera.capturePrivate(file)
            emit(session, DirectorUiState(DirectorStage.SAVING, true, "Applying final finish…", lastScore = state.score, camera = camera.uiState()))
            val uri = storage.saveFinal(file, state.look)
            recipeMemory.remember(state.plan, state.score)
            approved = null
            tools.restoreUserState()
            emit(
                session,
                DirectorUiState(
                    DirectorStage.READY,
                    true,
                    "Post-ready photo saved",
                    lastScore = state.score,
                    camera = camera.uiState(),
                    lastSavedUri = uri.toString(),
                    recipeCount = recipeMemory.count()
                )
            )
            uri
        } catch (t: Throwable) {
            approved = null
            runCatching { tools.restoreUserState() }
            emit(session, DirectorUiState(DirectorStage.ERROR, vlm.isReady(), "Final capture failed", error = t.message, camera = camera.uiState()))
            null
        } finally {
            file?.let { storage.cleanup(listOf(it)) }
        }
    }

    private fun emit(session: Long, state: DirectorUiState) {
        if (isCurrent(session)) onState(state)
    }

    private fun isCurrent(session: Long): Boolean = generation.get() == session
    private fun assertCurrent(session: Long) {
        if (!isCurrent(session)) throw StaleSession()
    }

    private class StaleSession : RuntimeException()

    companion object {
        private const val HARD_PREPARE_DEADLINE_MS = 14_000L
        private const val FAST_FRAME_TIMEOUT_MS = 700L
        private const val FAST_VERIFY_TIMEOUT_MS = 550L
        private const val LOCAL_SETTLE_TIMEOUT_MS = 750L
    }
}
