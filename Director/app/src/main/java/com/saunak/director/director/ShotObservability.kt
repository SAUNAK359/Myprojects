package com.saunak.director.director

import android.os.SystemClock
import com.saunak.director.tools.ToolExecutionStatus


enum class PipelineActor { GEMINI, SIERRA_BETA, CAMERA, LIVE_DIRECTOR, CAPTURE_ENGINE, PRIF, RENDER_EXECUTOR, TECHNICAL_GATE, STORAGE, SYSTEM }
enum class PipelineStage {
    SCENE_ANALYSIS,
    AI_PLANNING,
    PLAN_VALIDATION,
    CAMERA_RESOLUTION,
    CAMERA_APPLICATION,
    CAMERA_VERIFICATION,
    LIVE_DIRECTING,
    PRE_SHUTTER_VALIDATION,
    FINAL_CORRECTION,
    CAPTURE,
    PRIF_RECONSTRUCTION,
    RENDER,
    TECHNICAL_VALIDATION,
    MEDIASTORE_SAVE,
    COMPLETE,
    FALLBACK,
    ERROR
}
enum class PipelineEventState { ACTIVE, COMPLETE, CLAMPED, REJECTED, FAILED, INFO }

data class PipelineEvent(
    val id: String,
    val timestampMs: Long,
    val actor: PipelineActor,
    val stage: PipelineStage,
    val state: PipelineEventState,
    val title: String,
    val detail: String? = null
)

/**
 * Converts real orchestrator states/receipts into a bounded user-visible and auditable timeline.
 * It never fabricates percentages or progress. Repeated live-coaching text is deduplicated.
 */
class ShotObservability {
    private val events = ArrayDeque<PipelineEvent>()
    private val seenReceiptTimes = mutableSetOf<Long>()
    private var lastSignature: String? = null
    private var lastSavedUri: String? = null
    private var sequence = 0L

    fun reset() {
        events.clear()
        seenReceiptTimes.clear()
        lastSignature = null
        lastSavedUri = null
        sequence = 0L
    }

    fun observe(state: DirectorUiState, previous: DirectorUiState?): List<PipelineEvent> {
        if (state.stage == DirectorStage.ANALYZING && previous?.stage !in setOf(DirectorStage.ANALYZING, DirectorStage.PLANNING)) reset()
        state.executionReceipts.forEach { receipt ->
            if (seenReceiptTimes.add(receipt.timestampMs)) {
                val eventState = when (receipt.status) {
                    ToolExecutionStatus.APPLIED -> PipelineEventState.COMPLETE
                    ToolExecutionStatus.CLAMPED -> PipelineEventState.CLAMPED
                    ToolExecutionStatus.REJECTED -> PipelineEventState.REJECTED
                    ToolExecutionStatus.FAILED -> PipelineEventState.FAILED
                }
                append(
                    actor = PipelineActor.CAMERA,
                    stage = PipelineStage.CAMERA_APPLICATION,
                    state = eventState,
                    title = when (receipt.status) {
                        ToolExecutionStatus.APPLIED -> "Camera action applied"
                        ToolExecutionStatus.CLAMPED -> "Camera action resolved to device limit"
                        ToolExecutionStatus.REJECTED -> "Camera action rejected"
                        ToolExecutionStatus.FAILED -> "Camera action failed"
                    },
                    detail = receipt.compactDescription()
                )
            }
        }

        val mapped = mapState(state)
        val signature = mapped?.let { "${it.first}|${it.second}|${it.third}|${state.status}" }
        if (mapped != null && signature != lastSignature) {
            append(mapped.first, mapped.second, mapped.third, mapped.fourth, state.status.take(220))
            lastSignature = signature
        }
        if (state.aiFallbackReason != null && previous?.aiFallbackReason != state.aiFallbackReason) {
            append(PipelineActor.SYSTEM, PipelineStage.FALLBACK, PipelineEventState.INFO, "Authority path changed", state.aiFallbackReason.take(180))
        }
        if (state.lastSavedUri != null && state.lastSavedUri != lastSavedUri) {
            append(PipelineActor.STORAGE, PipelineStage.COMPLETE, PipelineEventState.COMPLETE, "Saved to Gallery", state.finalReviewStatus ?: "MediaStore write confirmed")
            lastSavedUri = state.lastSavedUri
        }
        return events.toList()
    }

    private fun mapState(state: DirectorUiState): Quadruple? {
        val status = state.status.lowercase()
        return when (state.stage) {
            DirectorStage.ANALYZING -> if (state.sierraMode == SierraMode.AI)
                Quadruple(PipelineActor.GEMINI, PipelineStage.AI_PLANNING, PipelineEventState.ACTIVE, "Gemini planning")
            else Quadruple(PipelineActor.SIERRA_BETA, PipelineStage.SCENE_ANALYSIS, PipelineEventState.ACTIVE, "Local scene analysis")
            DirectorStage.PLANNING -> Quadruple(PipelineActor.SIERRA_BETA, PipelineStage.PLAN_VALIDATION, PipelineEventState.ACTIVE, "Plan validation")
            DirectorStage.APPLYING -> Quadruple(PipelineActor.CAMERA, PipelineStage.CAMERA_APPLICATION, PipelineEventState.ACTIVE, "Applying camera plan")
            DirectorStage.CHECKING -> when {
                "semantic" in status || "closest achievable" in status -> Quadruple(PipelineActor.GEMINI, PipelineStage.PRE_SHUTTER_VALIDATION, PipelineEventState.ACTIVE, "Pre-shutter semantic validation")
                state.liveDirector.running -> Quadruple(PipelineActor.LIVE_DIRECTOR, PipelineStage.LIVE_DIRECTING, PipelineEventState.ACTIVE, "Live Director")
                else -> Quadruple(PipelineActor.CAMERA, PipelineStage.CAMERA_VERIFICATION, PipelineEventState.ACTIVE, "Camera-state verification")
            }
            DirectorStage.CORRECTING -> Quadruple(PipelineActor.SIERRA_BETA, PipelineStage.FINAL_CORRECTION, PipelineEventState.ACTIVE, "One final correction")
            DirectorStage.PREPARED -> Quadruple(PipelineActor.SYSTEM, PipelineStage.PRE_SHUTTER_VALIDATION, PipelineEventState.COMPLETE, "Capture ready")
            DirectorStage.FINAL_CAPTURING, DirectorStage.CAPTURING, DirectorStage.RECAPTURING -> Quadruple(PipelineActor.CAPTURE_ENGINE, PipelineStage.CAPTURE, PipelineEventState.ACTIVE, "Full-quality capture")
            DirectorStage.SAVING -> when {
                "p-rif" in status -> Quadruple(PipelineActor.PRIF, PipelineStage.PRIF_RECONSTRUCTION, PipelineEventState.ACTIVE, "P-RIF reconstruction")
                "render" in status -> Quadruple(PipelineActor.RENDER_EXECUTOR, PipelineStage.RENDER, PipelineEventState.ACTIVE, "Sierra Render Executor")
                "technical" in status || "evidence" in status -> Quadruple(PipelineActor.TECHNICAL_GATE, PipelineStage.TECHNICAL_VALIDATION, PipelineEventState.ACTIVE, "Technical safety gate")
                else -> Quadruple(PipelineActor.STORAGE, PipelineStage.MEDIASTORE_SAVE, PipelineEventState.ACTIVE, "Saving to Gallery")
            }
            DirectorStage.ERROR -> Quadruple(PipelineActor.SYSTEM, PipelineStage.ERROR, PipelineEventState.FAILED, "Shot pipeline failed")
            else -> null
        }
    }

    private fun append(actor: PipelineActor, stage: PipelineStage, state: PipelineEventState, title: String, detail: String?) {
        sequence++
        events.addLast(PipelineEvent("evt-$sequence", SystemClock.elapsedRealtime(), actor, stage, state, title, detail))
        while (events.size > MAX_EVENTS) events.removeFirst()
    }

    private data class Quadruple(
        val first: PipelineActor,
        val second: PipelineStage,
        val third: PipelineEventState,
        val fourth: String
    )

    companion object { private const val MAX_EVENTS = 80 }
}
