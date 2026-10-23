package com.saunak.director.director

import android.net.Uri
import com.saunak.director.ai.VlmRuntime
import com.saunak.director.camera.CameraEngine
import com.saunak.director.platform.ThermalMonitor
import com.saunak.director.processing.LookProcessor
import com.saunak.director.processing.LookSpec
import com.saunak.director.processing.PhotoQuality
import com.saunak.director.skills.SkillEngine
import com.saunak.director.storage.PhotoRepository
import com.saunak.director.storage.RecipeMemory
import com.saunak.director.tools.ToolExecutor
import com.saunak.director.tools.ToolIds
import kotlinx.coroutines.CancellationException
import java.io.File
import java.util.concurrent.atomic.AtomicLong

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

    suspend fun onPrimaryPressed(): Uri? = if (approved != null) captureFinal() else prepareShot()

    private suspend fun prepareShot(): Uri? {
        val session = cancelForUserAction()
        val created = mutableListOf<File>()
        try {
            tools.restoreUserState()
            if (!vlm.isReady()) {
                emit(session, DirectorUiState(DirectorStage.CAMERA_ONLY, false, "Add Gemini API key in Settings", camera = camera.uiState()))
                return null
            }
            val thermalPolicy = thermal.policy()
            if (!thermalPolicy.allowAi) {
                emit(session, DirectorUiState(DirectorStage.CAMERA_ONLY, true, "Director paused for device temperature", camera = camera.uiState()))
                return null
            }

            emit(session, DirectorUiState(DirectorStage.CAPTURING, true, "Reading scene…", camera = camera.uiState()))
            val reference = storage.newAttempt(session, 0).also { created += it }
            camera.capturePrivate(reference)
            assertCurrent(session)
            val referenceBitmap = storage.decodeOriented(reference)
            try {
                val referenceQuality = PhotoQuality.analyze(referenceBitmap)
                if (referenceQuality.catastrophic) {
                    emit(session, DirectorUiState(DirectorStage.READY, true, "Hold steady and try again", camera = camera.uiState()))
                    return null
                }

                emit(session, DirectorUiState(DirectorStage.ANALYZING, true, "Understanding scene & light…", camera = camera.uiState()))
                vlm.setMaxImageDimension(thermalPolicy.maxAiDimension)
                var caps = camera.capabilities()
                val plan = vlm.plan(referenceBitmap, caps, camera.snapshot(), skills.planningContext()).getOrElse {
                    emit(session, DirectorUiState(DirectorStage.ERROR, true, "Director analysis failed", error = it.message, camera = camera.uiState()))
                    return null
                }
                assertCurrent(session)
                val branch = skills.resolveBranch(plan.skillBranch)
                val physicalBudget = CapturePolicy.physicalBudget(thermalPolicy.maxPhysicalCaptures)

                emit(session, DirectorUiState(DirectorStage.APPLYING, true, "Composing the shot…", camera = camera.uiState()))
                val initialExecution = tools.apply(plan.toolCalls, caps)
                assertCurrent(session)
                caps = camera.capabilities()
                var look = initialExecution.look
                val finalCalls = initialExecution.applied.toMutableList()
                var issues = initialExecution.rejected.map { "tool_${it.call.id}:${it.reason}" }.take(8)
                var physicalCaptures = 1
                var currentFile = reference
                var lastScore = CapturePolicy.combinedFallbackScore(plan.referenceScore, referenceQuality.heuristicScore)

                if (initialExecution.requiresRecapture && physicalCaptures < physicalBudget) {
                    camera.awaitSettledState()
                    assertCurrent(session)
                    physicalCaptures++
                    currentFile = storage.newAttempt(session, physicalCaptures).also { created += it }
                    emit(session, DirectorUiState(DirectorStage.RECAPTURING, true, "Testing Director setup…", attempt = physicalCaptures, camera = camera.uiState()))
                    camera.capturePrivate(currentFile)
                }

                var evaluationCount = 0
                while (evaluationCount < CapturePolicy.MAX_EVALUATIONS) {
                    assertCurrent(session)
                    evaluationCount++
                    val decoded = storage.decodeOriented(currentFile)
                    val quality = PhotoQuality.analyze(decoded)
                    val rendered = LookProcessor.apply(decoded, look)
                    try {
                        emit(session, DirectorUiState(DirectorStage.CHECKING, true, "Critiquing result…", attempt = physicalCaptures, camera = camera.uiState()))
                        val eval = vlm.evaluate(
                            rendered,
                            plan.objective,
                            physicalCaptures,
                            camera.snapshot(),
                            issues,
                            skills.evaluationContext(branch),
                            caps
                        ).getOrElse { break }
                        assertCurrent(session)
                        lastScore = CapturePolicy.combinedFallbackScore(eval.score, quality.heuristicScore)
                        if (CapturePolicy.accepted(eval, quality.catastrophic)) {
                            approve(session, plan.copy(toolCalls = finalCalls.toList()), look, lastScore)
                            return null
                        }

                        issues = (eval.issues + issues).distinct().take(8)
                        if (evaluationCount >= CapturePolicy.MAX_EVALUATIONS) break
                        val deltaInput = if (physicalCaptures < physicalBudget) eval.deltaCalls else eval.deltaCalls.filter { it.id in ToolIds.LOOK_IDS }
                        if (deltaInput.isEmpty()) break

                        emit(session, DirectorUiState(DirectorStage.CORRECTING, true, "Refining camera…", attempt = physicalCaptures, lastScore = lastScore, camera = camera.uiState()))
                        val delta = tools.apply(deltaInput, caps, look)
                        assertCurrent(session)
                        if (delta.applied.isEmpty()) break
                        mergeCalls(finalCalls, delta.applied)
                        caps = camera.capabilities()
                        look = delta.look
                        issues = (delta.rejected.map { "tool_${it.call.id}:${it.reason}" } + issues).distinct().take(8)
                        if (delta.requiresRecapture) {
                            if (physicalCaptures >= physicalBudget) break
                            camera.awaitSettledState()
                            assertCurrent(session)
                            physicalCaptures++
                            currentFile = storage.newAttempt(session, physicalCaptures).also { created += it }
                            emit(session, DirectorUiState(DirectorStage.RECAPTURING, true, "Testing refinement…", attempt = physicalCaptures, camera = camera.uiState()))
                            camera.capturePrivate(currentFile)
                        }
                    } finally {
                        if (rendered !== decoded && !rendered.isRecycled) rendered.recycle()
                        if (!decoded.isRecycled) decoded.recycle()
                    }
                }

                if (lastScore >= 0.82) {
                    approve(session, plan.copy(toolCalls = finalCalls.toList()), look, lastScore, "Best achievable setup ready")
                } else {
                    tools.restoreUserState()
                    emit(session, DirectorUiState(DirectorStage.READY, true, "Couldn’t improve enough — try reframing", lastScore = lastScore, camera = camera.uiState()))
                }
                return null
            } finally {
                if (!referenceBitmap.isRecycled) referenceBitmap.recycle()
            }
        } catch (_: StaleSession) {
            return null
        } catch (_: CancellationException) {
            return null
        } catch (t: Throwable) {
            runCatching { tools.restoreUserState() }
            emit(session, DirectorUiState(DirectorStage.ERROR, vlm.isReady(), "Director preparation failed", error = t.message, camera = camera.uiState()))
            return null
        } finally {
            storage.cleanup(created)
        }
    }

    private fun mergeCalls(target: MutableList<ToolCall>, updates: List<ToolCall>) {
        updates.forEach { call ->
            target.removeAll { it.id == call.id }
            target += call
        }
    }

    private fun approve(session: Long, plan: PlanOutput, look: LookSpec, score: Double, message: String = "Ready — take the shot") {
        approved = ApprovedState(look, plan, score)
        emit(session, DirectorUiState(DirectorStage.PREPARED, true, message, lastScore = score, camera = camera.uiState(), recipeCount = recipeMemory.count()))
    }

    private suspend fun captureFinal(): Uri? {
        val state = approved ?: return null
        val session = generation.get()
        var file: File? = null
        return try {
            emit(session, DirectorUiState(DirectorStage.FINAL_CAPTURING, true, "Capturing…", lastScore = state.score, camera = camera.uiState()))
            camera.awaitSettledState(650)
            file = storage.newAttempt(session, 99)
            camera.capturePrivate(file)
            emit(session, DirectorUiState(DirectorStage.SAVING, true, "Saving masterpiece…", lastScore = state.score, camera = camera.uiState()))
            val uri = storage.saveFinal(file, state.look)
            recipeMemory.remember(state.plan, state.score)
            approved = null
            tools.restoreUserState()
            emit(session, DirectorUiState(DirectorStage.READY, true, "Saved to Gallery", lastScore = state.score, camera = camera.uiState(), lastSavedUri = uri.toString(), recipeCount = recipeMemory.count()))
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

    private fun emit(session: Long, state: DirectorUiState) { if (isCurrent(session)) onState(state) }
    private fun isCurrent(session: Long): Boolean = generation.get() == session
    private fun assertCurrent(session: Long) { if (!isCurrent(session)) throw StaleSession() }
    private class StaleSession : RuntimeException()
}
