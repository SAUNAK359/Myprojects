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
 * Sierra pipeline: Sierra β always owns execution and technical verification. Sierra AI may
 * provide the semantic plan, but any unavailable/failed/disabled cloud route falls back to the
 * deterministic Sierra β planner using the same validated tool surface.
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
    private data class ApprovedState(val look: LookSpec, val plan: PlanOutput, val score: Double, val mode: SierraMode)
    @Volatile private var approved: ApprovedState? = null
    @Volatile private var cloudEnabled: Boolean = true

    fun isPrepared(): Boolean = approved != null
    fun setCloudEnabled(enabled: Boolean) {
        cloudEnabled = enabled
        if (!enabled) vlm.cancel()
    }
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
                    aiAvailable = cloudEnabled && vlm.isReady(),
                    sierraMode = SierraMode.BETA,
                    status = "Sierra β ready • cloud preparation timed out",
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
            val thermalPolicy = thermal.policy()
            val cloudAllowed = cloudEnabled && vlm.isReady() && thermalPolicy.allowAi

            emit(
                session,
                DirectorUiState(
                    DirectorStage.ANALYZING,
                    cloudAllowed,
                    if (cloudAllowed) "Sierra is reading the scene…" else "Sierra β is reading the scene…",
                    camera = camera.uiState(),
                    sierraMode = if (cloudAllowed) SierraMode.AI else SierraMode.BETA
                )
            )

            planningBitmap = camera.captureAnalysisFrame(FAST_FRAME_TIMEOUT_MS)
            if (planningBitmap == null) {
                emit(session, DirectorUiState(DirectorStage.CAPTURING, cloudAllowed, "Sampling camera…", camera = camera.uiState(), sierraMode = if (cloudAllowed) SierraMode.AI else SierraMode.BETA))
                val fallback = storage.newAttempt(session, 0).also { created += it }
                camera.capturePrivate(fallback)
                assertCurrent(session)
                planningBitmap = storage.decodeOriented(fallback, 896)
            }

            val frame = checkNotNull(planningBitmap)
            val referenceQuality = PhotoQuality.analyze(frame)
            if (referenceQuality.catastrophic) {
                emit(session, DirectorUiState(DirectorStage.READY, cloudAllowed, "Scene too dark or blank • reframe", camera = camera.uiState(), sierraMode = SierraMode.BETA))
                return null
            }

            var mode = SierraMode.BETA
            val basePlan = if (cloudAllowed) {
                emit(session, DirectorUiState(DirectorStage.ANALYZING, true, "Sierra AI is composing…", camera = camera.uiState(), sierraMode = SierraMode.AI))
                vlm.setMaxImageDimension(thermalPolicy.maxAiDimension)
                val cloud = vlm.plan(frame, camera.capabilities(), camera.snapshot(), skills.planningContext())
                cloud.getOrNull()?.also { mode = SierraMode.AI } ?: run {
                    emit(session, DirectorUiState(DirectorStage.PLANNING, false, "Sierra β is taking over…", camera = camera.uiState(), sierraMode = SierraMode.BETA))
                    SierraLocalPlanner.plan(referenceQuality, frame.width, frame.height)
                }
            } else {
                SierraLocalPlanner.plan(referenceQuality, frame.width, frame.height)
            }
            assertCurrent(session)

            val plan = InstantFinishEngine.enrich(basePlan, referenceQuality)
            emit(
                session,
                DirectorUiState(
                    DirectorStage.APPLYING,
                    mode == SierraMode.AI,
                    if (mode == SierraMode.AI) "Sierra AI directed • Sierra β applying" else "Sierra β preparing the shot",
                    camera = camera.uiState(),
                    sierraMode = mode,
                    aspectRatio = plan.aspectRatio,
                    sceneLabel = plan.scene,
                    objective = plan.objective
                )
            )

            var caps = camera.capabilities()
            val execution = tools.apply(plan.toolCalls, caps)
            assertCurrent(session)
            caps = camera.capabilities()
            val look = execution.look
            val appliedPlan = plan.copy(toolCalls = execution.applied)

            if (execution.requiresRecapture) {
                emit(session, DirectorUiState(DirectorStage.CHECKING, mode == SierraMode.AI, "Locking focus & exposure…", camera = camera.uiState(), sierraMode = mode, aspectRatio = plan.aspectRatio, sceneLabel = plan.scene, objective = plan.objective))
                camera.awaitSettledState(LOCAL_SETTLE_TIMEOUT_MS)
                assertCurrent(session)
            }

            emit(session, DirectorUiState(DirectorStage.CHECKING, mode == SierraMode.AI, "Sierra β checking quality…", camera = camera.uiState(), sierraMode = mode, aspectRatio = plan.aspectRatio, sceneLabel = plan.scene, objective = plan.objective))
            val verification = camera.captureAnalysisFrame(FAST_VERIFY_TIMEOUT_MS)
            val score = if (verification != null) {
                try {
                    val cameraQuality = PhotoQuality.analyze(verification)
                    if (cameraQuality.catastrophic) {
                        tools.restoreUserState()
                        emit(session, DirectorUiState(DirectorStage.READY, mode == SierraMode.AI, "Hold steady or add light • try again", camera = camera.uiState(), sierraMode = mode))
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

            approve(session, appliedPlan, look, score, mode)
            return null
        } catch (_: StaleSession) {
            return null
        } catch (t: CancellationException) {
            throw t
        } catch (t: Throwable) {
            runCatching { tools.restoreUserState() }
            emit(session, DirectorUiState(DirectorStage.ERROR, false, "Sierra could not prepare this shot", error = t.message, camera = camera.uiState(), sierraMode = SierraMode.BETA))
            return null
        } finally {
            planningBitmap?.let { if (!it.isRecycled) it.recycle() }
            storage.cleanup(created)
        }
    }

    private fun combinedReadyScore(ai: Double, camera: Double, finish: Double): Double =
        (0.48 * ai.coerceIn(0.0, 1.0) + 0.34 * camera.coerceIn(0.0, 1.0) + 0.18 * finish.coerceIn(0.0, 1.0))
            .coerceIn(0.0, 1.0)

    private fun approve(session: Long, plan: PlanOutput, look: LookSpec, score: Double, mode: SierraMode) {
        approved = ApprovedState(look, plan, score, mode)
        emit(
            session,
            DirectorUiState(
                DirectorStage.PREPARED,
                mode == SierraMode.AI,
                if (mode == SierraMode.AI) "Ready • Sierra AI + β" else "Ready • Sierra β",
                lastScore = score,
                camera = camera.uiState(),
                recipeCount = recipeMemory.count(),
                sierraMode = mode,
                aspectRatio = plan.aspectRatio,
                sceneLabel = plan.scene,
                objective = plan.objective
            )
        )
    }

    private suspend fun captureFinal(): Uri? {
        val state = approved ?: return null
        val session = generation.get()
        var file: File? = null
        return try {
            emit(session, DirectorUiState(DirectorStage.FINAL_CAPTURING, state.mode == SierraMode.AI, "Capturing full quality…", lastScore = state.score, camera = camera.uiState(), sierraMode = state.mode, aspectRatio = state.plan.aspectRatio, sceneLabel = state.plan.scene, objective = state.plan.objective))
            camera.awaitSettledState(550)
            file = storage.newAttempt(session, 99)
            camera.capturePrivate(file)
            emit(session, DirectorUiState(DirectorStage.SAVING, state.mode == SierraMode.AI, "Sierra β finishing…", lastScore = state.score, camera = camera.uiState(), sierraMode = state.mode, aspectRatio = state.plan.aspectRatio, sceneLabel = state.plan.scene, objective = state.plan.objective))
            val uri = storage.saveFinal(file, state.look, state.plan.aspectRatio, state.plan.cropCenterX, state.plan.cropCenterY)
            recipeMemory.remember(state.plan, state.score)
            approved = null
            tools.restoreUserState()
            emit(
                session,
                DirectorUiState(
                    DirectorStage.READY,
                    cloudEnabled && vlm.isReady(),
                    "Sierra photo saved",
                    lastScore = state.score,
                    camera = camera.uiState(),
                    lastSavedUri = uri.toString(),
                    recipeCount = recipeMemory.count(),
                    sierraMode = state.mode,
                    aspectRatio = state.plan.aspectRatio,
                    sceneLabel = state.plan.scene,
                    objective = state.plan.objective
                )
            )
            uri
        } catch (t: Throwable) {
            approved = null
            runCatching { tools.restoreUserState() }
            emit(session, DirectorUiState(DirectorStage.ERROR, cloudEnabled && vlm.isReady(), "Final capture failed", error = t.message, camera = camera.uiState(), sierraMode = state.mode))
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
        private const val HARD_PREPARE_DEADLINE_MS = 8_000L
        private const val FAST_FRAME_TIMEOUT_MS = 650L
        private const val FAST_VERIFY_TIMEOUT_MS = 500L
        private const val LOCAL_SETTLE_TIMEOUT_MS = 650L
    }
}
