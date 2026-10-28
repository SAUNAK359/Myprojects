package com.saunak.director.director

import android.graphics.Bitmap
import com.saunak.director.ai.VlmRuntime
import com.saunak.director.camera.CameraEngine
import com.saunak.director.director.live.LiveDirectorSessionRunner
import com.saunak.director.director.live.LiveDirectorState
import com.saunak.director.tools.CameraStateVerifier
import com.saunak.director.tools.ToolExecutionReceipt
import com.saunak.director.tools.ToolExecutionStatus
import com.saunak.director.tools.ToolExecutor
import java.util.Locale

/**
 * Gemini call #2 authority. Exactly one model validation is allowed. If correction is requested,
 * beta executes it once and all remaining convergence/verification is local.
 */
class PreShutterAuthority(
    private val camera: CameraEngine,
    private val vlm: VlmRuntime,
    private val tools: ToolExecutor,
    private val liveDirector: LiveDirectorSessionRunner,
    private val cameraVerifier: CameraStateVerifier = CameraStateVerifier(camera)
) {
    data class Result(
        val ready: Boolean,
        val plan: PlanOutput,
        val receipts: List<ToolExecutionReceipt>,
        val liveState: LiveDirectorState,
        val corrected: Boolean,
        val status: String,
        val reason: String? = null
    )

    suspend fun validateOnce(
        originalPlan: PlanOutput,
        existingReceipts: List<ToolExecutionReceipt>,
        thermalPressure: Double,
        isCurrent: () -> Boolean,
        sceneConstraintSuspected: Boolean,
        onLiveUpdate: (LiveDirectorState) -> Unit
    ): Result {
        if (!vlm.supportsPreShutterValidation()) {
            return Result(false, originalPlan, existingReceipts, liveDirector.latest(), false,
                "Sierra pre-shutter authority unavailable", "pre_shutter_validation_unavailable")
        }
        if (!isCurrent()) return Result(false, originalPlan, existingReceipts, liveDirector.latest(), false, "Shot cancelled", "stale_session")

        val caps = camera.capabilities()
        var receipts = existingReceipts.toMutableList()
        var cameraReport = cameraVerifier.verify(originalPlan.toolCalls, caps)

        // One fast local repair pass for controls that did not settle at the resolved target.
        if (cameraReport.retryCalls.isNotEmpty() || !cameraReport.settled) {
            if (cameraReport.retryCalls.isNotEmpty()) {
                val retry = tools.apply(cameraReport.retryCalls, caps, allowPredictiveRetarget = false)
                receipts += retry.receipts
            }
            cameraReport = cameraVerifier.verify(originalPlan.toolCalls, camera.capabilities(), RETRY_SETTLE_MS)
        }
        if (cameraReport.hardFailure) {
            return Result(false, originalPlan, receipts, liveDirector.latest(), false,
                "Camera state could not reach Sierra target", "camera_state_verification_failed")
        }

        val frame = camera.captureAnalysisFrame(VALIDATION_FRAME_TIMEOUT_MS)
            ?: return Result(false, originalPlan, receipts, liveDirector.latest(), false,
                "Fresh validation frame unavailable", "validation_frame_unavailable")
        return try {
            val context = buildContext(originalPlan, liveDirector.latest(), cameraReport, sceneConstraintSuspected)
            vlm.setMaxImageDimension(VALIDATION_UPLOAD_EDGE)
            val validation = vlm.validatePreShutter(frame, originalPlan, camera.snapshot(), context, camera.capabilities()).getOrElse { error ->
                return Result(false, originalPlan, receipts, liveDirector.latest(), false,
                    "Sierra pre-shutter validation unavailable", error.message?.take(160))
            }

            if (validation.accept) {
                // Gemini has accepted semantics. The camera still must be physically settled now.
                val finalPhysical = cameraVerifier.verify(originalPlan.toolCalls, camera.capabilities(), FINAL_SETTLE_MS)
                if (!finalPhysical.verified || !finalPhysical.settled) {
                    return Result(false, originalPlan, receipts, liveDirector.latest(), false,
                        "Camera state is still settling", "three_a_not_settled")
                }
                return Result(
                    true, originalPlan, receipts, liveDirector.latest(), false,
                    "Sierra validated the final scene • ${fmt(validation.intentMatch)} intent match"
                )
            }

            // This is the one and only correction package. No further Gemini request follows.
            val correctionExecution = if (validation.correctionCalls.isNotEmpty()) {
                tools.apply(validation.correctionCalls, camera.capabilities(), allowPredictiveRetarget = false)
            } else null
            correctionExecution?.let { receipts += it.receipts }
            val correctionFailure = correctionExecution?.receipts?.firstOrNull {
                it.status == ToolExecutionStatus.FAILED || it.status == ToolExecutionStatus.REJECTED
            }
            if (correctionFailure != null) {
                return Result(false, originalPlan, receipts, liveDirector.latest(), true,
                    "Final correction could not be executed", correctionFailure.reason)
            }

            val correctedPlan = originalPlan.copy(
                toolCalls = if (correctionExecution != null) mergeCameraCalls(originalPlan.toolCalls, correctionExecution.applied) else originalPlan.toolCalls,
                directorSpec = validation.correctedDirectorSpec ?: originalPlan.directorSpec
            )

            var postCorrectionReport = cameraVerifier.verify(correctedPlan.toolCalls, camera.capabilities(), RETRY_SETTLE_MS)
            if (postCorrectionReport.retryCalls.isNotEmpty()) {
                val repair = tools.apply(postCorrectionReport.retryCalls, camera.capabilities(), allowPredictiveRetarget = false)
                receipts += repair.receipts
                postCorrectionReport = cameraVerifier.verify(correctedPlan.toolCalls, camera.capabilities(), RETRY_SETTLE_MS)
            }
            if (postCorrectionReport.hardFailure) {
                return Result(false, correctedPlan, receipts, liveDirector.latest(), true,
                    "Corrected camera state could not be verified", "corrected_camera_state_failed")
            }

            val correctedDirector = validation.correctedDirectorSpec
            val guidance = if (correctedDirector != null) {
                val correctionFrame = camera.captureAnalysisFrame(CORRECTION_FRAME_TIMEOUT_MS)
                    ?: return Result(false, correctedPlan, receipts, liveDirector.latest(), true,
                        "Correction frame unavailable", "correction_frame_unavailable")
                try {
                    liveDirector.guideUntilReady(correctionFrame, correctedDirector, thermalPressure, isCurrent, onLiveUpdate)
                } finally {
                    if (!correctionFrame.isRecycled) correctionFrame.recycle()
                }
            } else {
                liveDirector.resumeUntilReady(thermalPressure, isCurrent, onLiveUpdate)
            }

            if (!guidance.ready) {
                return Result(false, correctedPlan, receipts, guidance.state, true,
                    "One final correction did not reach capture tolerance", guidance.reason)
            }

            // Final local freshness/camera check only. Never call Gemini again.
            val fresh = liveDirector.reverify(thermalPressure, isCurrent, onLiveUpdate)
            val finalReport = cameraVerifier.verify(correctedPlan.toolCalls, camera.capabilities(), FINAL_SETTLE_MS)
            if (!fresh.ready || !finalReport.verified || !finalReport.settled) {
                return Result(false, correctedPlan, receipts, fresh.state, true,
                    "Final corrected scene moved outside capture tolerance",
                    fresh.reason ?: "final_camera_or_three_a_verification_failed")
            }

            Result(
                true, correctedPlan, receipts, fresh.state, true,
                "Sierra final correction locked • local verification passed"
            )
        } finally {
            if (!frame.isRecycled) frame.recycle()
        }
    }

    private fun buildContext(
        plan: PlanOutput,
        live: LiveDirectorState,
        cameraReport: CameraStateVerifier.Report,
        sceneConstraintSuspected: Boolean
    ): String = buildString {
        appendLine("scene_constraint_suspected=$sceneConstraintSuspected")
        appendLine("live_capture_ready=${live.captureReady};live_stable=${live.stable};live_score=${fmt(live.readinessScore)};entity_locked=${live.entityLocked}")
        appendLine("live_status=${live.status.take(160)}")
        appendLine(cameraReport.compactDescription())
        camera.sensorTelemetry()?.let { sensor ->
            appendLine("camera2={sensor_ts_ns=${sensor.sensorTimestampNs};exposure_ns=${sensor.exposureTimeNs};iso=${sensor.sensitivityIso};frame_ns=${sensor.frameDurationNs};focus_diopters=${sensor.focusDistanceDiopters ?: "unknown"};af=${sensor.afState ?: "unknown"};ae=${sensor.aeState ?: "unknown"};awb=${sensor.awbState ?: "unknown"}}")
        }
        live.evaluations.take(16).forEach { e ->
            append("goal=").append(e.node.id).append(";type=").append(e.node.type)
            append(";state=").append(e.measurement.state)
            append(";satisfaction=").append(fmt(e.satisfaction))
            append(";confidence=").append(fmt(e.measurement.confidence))
            append(";value=").append(e.measurement.value?.let(::fmt) ?: "unknown")
            append(";source=").append(e.measurement.source.take(64)).appendLine()
        }
        appendLine("original_render_intent=${plan.renderSpec?.intentSummary?.take(180) ?: "missing"}")
    }

    private fun mergeCameraCalls(original: List<ToolCall>, correction: List<ToolCall>): List<ToolCall> {
        if (correction.isEmpty()) return original
        val replacementIds = correction.map { it.id }.toSet()
        return (original.filterNot { it.id in replacementIds } + correction).take(8)
    }

    private fun fmt(value: Double): String = String.format(Locale.US, "%.3f", value)

    companion object {
        private const val VALIDATION_FRAME_TIMEOUT_MS = 620L
        private const val CORRECTION_FRAME_TIMEOUT_MS = 620L
        private const val VALIDATION_UPLOAD_EDGE = 704
        private const val RETRY_SETTLE_MS = 520L
        private const val FINAL_SETTLE_MS = 520L
    }
}
