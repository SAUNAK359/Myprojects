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
    private val thermal: ThermalMonitor,
    private val onState: (DirectorUiState) -> Unit
) {
    private val generation = AtomicLong(0L)
    private data class Candidate(val file: File, val look: LookSpec, val score: Double, val valid: Boolean, val accepted: Boolean, val captureNumber: Int)

    suspend fun recoverPendingReferences(): List<Uri> = storage.recoverPendingToGallery()
    fun cancelForUserAction(): Long { val next = generation.incrementAndGet(); vlm.cancel(); return next }

    suspend fun onShutterPressed(): Uri? {
        val session = cancelForUserAction()
        var reference: File? = null
        val attempts = mutableListOf<File>()
        var finalized = false
        try {
            tools.restoreUserState()
            emit(session, DirectorUiState(DirectorStage.CAPTURING, vlm.isReady(), "Capturing reference…", camera = camera.uiState()))
            if (!storage.hasWorkingSpace()) emit(session, DirectorUiState(DirectorStage.CAPTURING, vlm.isReady(), "Low storage • protecting capture", camera = camera.uiState()))
            val ref = storage.newPendingReference(session); reference = ref
            camera.capturePrivate(ref); assertCurrent(session)

            val thermalPolicy = thermal.policy()
            if (!vlm.isReady() || !thermalPolicy.allowAi) {
                val uri = storage.saveRawJpeg(ref); storage.markFinalized(ref); finalized = true
                emit(session, DirectorUiState(DirectorStage.READY, vlm.isReady(), if (thermalPolicy.allowAi) "Saved • AI unavailable" else "Saved • AI paused for temperature", camera = camera.uiState(), lastSavedUri = uri.toString()))
                return uri
            }

            val referenceBitmap = runCatching { storage.decodeOriented(ref) }.getOrElse {
                val uri = storage.saveRawJpeg(ref); storage.markFinalized(ref); finalized = true
                emit(session, DirectorUiState(DirectorStage.READY, true, "Saved reference", camera = camera.uiState(), lastSavedUri = uri.toString()))
                return uri
            }
            try {
                val referenceQuality = PhotoQuality.analyze(referenceBitmap)
                if (referenceQuality.catastrophic) {
                    val uri = storage.saveRawJpeg(ref); storage.markFinalized(ref); finalized = true
                    emit(session, DirectorUiState(DirectorStage.READY, true, "Saved reference", camera = camera.uiState(), lastSavedUri = uri.toString()))
                    return uri
                }

                emit(session, DirectorUiState(DirectorStage.ANALYZING, true, "Analyzing captured photo…", camera = camera.uiState()))
                vlm.setMaxImageDimension(thermalPolicy.maxAiDimension)
                var caps = camera.capabilities()
                val plan = vlm.plan(referenceBitmap, caps, camera.snapshot(), skills.planningContext()).getOrElse {
                    return fallbackReference(session, ref, "Saved • AI analysis unavailable").also { finalized = it != null }
                }
                assertCurrent(session)
                val branch = skills.resolveBranch(plan.skillBranch)
                val candidates = mutableListOf(Candidate(ref, LookSpec(), CapturePolicy.combinedFallbackScore(plan.referenceScore, referenceQuality.heuristicScore), true, plan.referenceAccept, 1))
                val physicalBudget = CapturePolicy.physicalBudget(thermal.policy().maxPhysicalCaptures)

                val initialCalls = if (physicalBudget > 1 && plan.recapture) plan.toolCalls else plan.toolCalls.filter { it.id in ToolIds.LOOK_IDS }
                emit(session, DirectorUiState(DirectorStage.APPLYING, true, "Preparing improvement…", camera = camera.uiState()))
                val initialExecution = tools.apply(initialCalls, caps); assertCurrent(session)
                caps = camera.capabilities()
                var look = initialExecution.look
                var issues = initialExecution.rejected.map { "tool_${it.call.id}:${it.reason}" }.take(8)
                var physicalCaptures = 1
                var evaluationCount = 0
                var currentFile = ref

                if (initialExecution.requiresRecapture && physicalCaptures < physicalBudget) {
                    assertCurrent(session); camera.awaitSettledState(); assertCurrent(session)
                    physicalCaptures++
                    emit(session, DirectorUiState(DirectorStage.RECAPTURING, true, "Capturing improvement…", attempt = physicalCaptures, camera = camera.uiState()))
                    currentFile = storage.newAttempt(session, physicalCaptures).also { attempts += it }
                    camera.capturePrivate(currentFile); assertCurrent(session)
                } else if (plan.referenceAccept && look.isIdentity() && initialExecution.applied.none { it.id in ToolIds.LOOK_IDS }) {
                    val uri = finalizeCandidate(session, candidates.first(), ref, attempts); finalized = uri != null; return uri
                }

                while (evaluationCount < CapturePolicy.MAX_EVALUATIONS) {
                    assertCurrent(session); evaluationCount++
                    val decoded = storage.decodeOriented(currentFile)
                    val quality = PhotoQuality.analyze(decoded)
                    val rendered = LookProcessor.apply(decoded, look)
                    try {
                        emit(session, DirectorUiState(DirectorStage.CHECKING, true, "Checking result…", attempt = physicalCaptures, camera = camera.uiState()))
                        val eval = vlm.evaluate(rendered, plan.objective, physicalCaptures, camera.snapshot(), issues, skills.evaluationContext(branch), caps).getOrElse { break }
                        assertCurrent(session)
                        val score = CapturePolicy.combinedFallbackScore(eval.score, quality.heuristicScore)
                        val accepted = CapturePolicy.accepted(eval, quality.catastrophic)
                        candidates += Candidate(currentFile, look, score, !quality.catastrophic, accepted, physicalCaptures)
                        if (accepted) { val uri = finalizeCandidate(session, candidates.last(), ref, attempts); finalized = uri != null; return uri }

                        issues = (eval.issues + issues).distinct().take(8)
                        if (evaluationCount >= CapturePolicy.MAX_EVALUATIONS) break
                        val deltaInput = if (physicalCaptures < physicalBudget) eval.deltaCalls else eval.deltaCalls.filter { it.id in ToolIds.LOOK_IDS }
                        if (deltaInput.isEmpty()) break
                        emit(session, DirectorUiState(DirectorStage.CORRECTING, true, "Refining…", attempt = physicalCaptures, lastScore = eval.score, camera = camera.uiState()))
                        val delta = tools.apply(deltaInput, caps, look); assertCurrent(session)
                        if (delta.applied.isEmpty()) break
                        caps = camera.capabilities()
                        look = delta.look
                        issues = (delta.rejected.map { "tool_${it.call.id}:${it.reason}" } + issues).distinct().take(8)
                        if (delta.requiresRecapture) {
                            if (physicalCaptures >= physicalBudget) break
                            camera.awaitSettledState(); assertCurrent(session)
                            physicalCaptures++
                            emit(session, DirectorUiState(DirectorStage.RECAPTURING, true, "Capturing refinement…", attempt = physicalCaptures, camera = camera.uiState()))
                            currentFile = storage.newAttempt(session, physicalCaptures).also { attempts += it }
                            camera.capturePrivate(currentFile); assertCurrent(session)
                        }
                    } finally {
                        if (rendered !== decoded && !rendered.isRecycled) rendered.recycle()
                        if (!decoded.isRecycled) decoded.recycle()
                    }
                }

                assertCurrent(session)
                val best = candidates.filter { it.valid }.maxByOrNull { it.score } ?: candidates.first()
                val uri = finalizeCandidate(session, best, ref, attempts); finalized = uri != null; return uri
            } finally { if (!referenceBitmap.isRecycled) referenceBitmap.recycle() }
        } catch (_: StaleSession) {
            val ref = reference
            if (!finalized && ref != null && ref.exists() && ref.length() > 0L) runCatching { storage.saveRawJpeg(ref) }.onSuccess { storage.markFinalized(ref); finalized = true }
            return null
        } catch (_: CancellationException) {
            val ref = reference
            if (!finalized && ref != null && ref.exists() && ref.length() > 0L) runCatching { storage.saveRawJpeg(ref) }.onSuccess { storage.markFinalized(ref); finalized = true }
            return null
        } catch (t: Throwable) {
            val ref = reference
            if (!finalized && ref != null && ref.exists() && ref.length() > 0L) {
                val fallback = runCatching { storage.saveRawJpeg(ref) }.getOrNull()
                if (fallback != null) {
                    storage.markFinalized(ref); finalized = true
                    emit(session, DirectorUiState(DirectorStage.ERROR, vlm.isReady(), "AI failed • reference saved", error = t.message, camera = camera.uiState(), lastSavedUri = fallback.toString()))
                    return fallback
                }
            }
            emit(session, DirectorUiState(DirectorStage.ERROR, vlm.isReady(), "Capture retained privately • storage action needed", error = t.message, camera = camera.uiState()))
            return null
        } finally {
            storage.cleanup(attempts)
            if (isCurrent(session)) runCatching { tools.restoreUserState() }
        }
    }

    private suspend fun finalizeCandidate(session: Long, candidate: Candidate, reference: File, attempts: List<File>): Uri? {
        assertCurrent(session)
        emit(session, DirectorUiState(DirectorStage.SAVING, true, "Saving best photo…", attempt = candidate.captureNumber, lastScore = candidate.score, camera = camera.uiState()))
        val uri = storage.saveFinal(candidate.file, candidate.look)
        storage.markFinalized(reference); storage.cleanup(attempts)
        emit(session, DirectorUiState(DirectorStage.READY, vlm.isReady(), "Saved", lastScore = candidate.score, camera = camera.uiState(), lastSavedUri = uri.toString()))
        return uri
    }

    private suspend fun fallbackReference(session: Long, reference: File, status: String): Uri? {
        assertCurrent(session)
        val uri = storage.saveRawJpeg(reference); storage.markFinalized(reference)
        emit(session, DirectorUiState(DirectorStage.READY, vlm.isReady(), status, camera = camera.uiState(), lastSavedUri = uri.toString()))
        return uri
    }

    private fun emit(session: Long, state: DirectorUiState) { if (isCurrent(session)) onState(state) }
    private fun isCurrent(session: Long): Boolean = generation.get() == session
    private fun assertCurrent(session: Long) { if (!isCurrent(session)) throw StaleSession() }
    private class StaleSession : RuntimeException()
}
