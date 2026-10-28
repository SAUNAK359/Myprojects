package com.saunak.director.tools

import android.hardware.camera2.CaptureResult
import com.saunak.director.camera.CameraEngine
import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.camera.DirectorExtension
import com.saunak.director.camera.StillFlashMode
import com.saunak.director.director.ToolCall
import java.util.Locale
import kotlin.math.abs

/**
 * Fast local verification layer. Creative intent never changes here: requested values are normalized
 * to the closest physical value by ToolValidator, executed, then compared with actual CameraX/Camera2 state.
 */
class CameraStateVerifier(private val camera: CameraEngine) {
    enum class Status { VERIFIED, VERIFIED_CLOSEST, SEARCHING, UNKNOWN, FAILED }

    data class Item(
        val toolId: Int,
        val requested: String,
        val actual: String,
        val status: Status,
        val reason: String? = null
    )

    data class Report(
        val items: List<Item>,
        val af: String,
        val ae: String,
        val awb: String,
        val settled: Boolean,
        val retryCalls: List<ToolCall>
    ) {
        val hardFailure: Boolean get() = items.any { it.status == Status.FAILED }
        val verified: Boolean get() = !hardFailure && items.none { it.status == Status.SEARCHING }
        fun compactDescription(): String = buildString {
            append("camera_state_verified=").append(verified)
            append(";settled=").append(settled)
            append(";af=").append(af).append(";ae=").append(ae).append(";awb=").append(awb)
            items.forEach { item ->
                append(";tool_").append(item.toolId).append("={requested=").append(item.requested)
                    .append(",actual=").append(item.actual).append(",status=").append(item.status)
                item.reason?.let { append(",reason=").append(it.take(80)) }
                append('}')
            }
        }
    }

    suspend fun verify(calls: List<ToolCall>, caps: CapabilityRegistry, settleMs: Long = 360L): Report {
        camera.awaitSettledState(settleMs)
        val snapshot = camera.snapshot()
        val sensor = camera.sensorTelemetry()
        val afState = afName(sensor?.afState)
        val aeState = aeName(sensor?.aeState)
        val awbState = awbName(sensor?.awbState)
        val afSettled = sensor?.afState == null || sensor.afState in AF_SETTLED
        val aeSettled = sensor?.aeState == null || sensor.aeState in AE_SETTLED
        val awbSettled = sensor?.awbState == null || sensor.awbState in AWB_SETTLED
        val settled = afSettled && aeSettled && awbSettled
        val retry = mutableListOf<ToolCall>()

        val items = calls.filter { it.id in ToolIds.CAMERA_IDS }.map { call ->
            when (call.id) {
                ToolIds.ZOOM -> {
                    val target = call.value ?: snapshot.zoomRatio.toDouble()
                    val delta = abs(snapshot.zoomRatio - target)
                    val ok = delta <= maxOf(0.02, target * 0.015)
                    if (!ok) retry += call
                    Item(call.id, fmt(target) + "x", fmt(snapshot.zoomRatio.toDouble()) + "x", if (ok) Status.VERIFIED else Status.FAILED,
                        if (ok) null else "zoom state differs after execution")
                }
                ToolIds.EXPOSURE_EV -> {
                    val target = call.value ?: 0.0
                    val actual = snapshot.exposureCompensationIndex * caps.exposureStepEv.toDouble()
                    val tolerance = maxOf(abs(caps.exposureStepEv.toDouble()) * 0.55, 0.05)
                    val ok = abs(actual - target) <= tolerance
                    if (!ok || !aeSettled) retry += call
                    val closest = abs(actual - target) > 1e-6 && ok
                    Item(call.id, fmt(target) + "EV", fmt(actual) + "EV",
                        when { !ok -> Status.FAILED; !aeSettled -> Status.SEARCHING; closest -> Status.VERIFIED_CLOSEST; else -> Status.VERIFIED },
                        if (!aeSettled) "AE not settled" else null)
                }
                ToolIds.FOCUS -> {
                    val status = when {
                        sensor?.afState == null -> Status.UNKNOWN
                        sensor.afState in AF_GOOD -> Status.VERIFIED
                        sensor.afState in AF_SEARCHING -> Status.SEARCHING
                        sensor.afState in AF_SETTLED -> Status.VERIFIED_CLOSEST
                        else -> Status.UNKNOWN
                    }
                    if (status == Status.SEARCHING) retry += call
                    Item(call.id, "point(${fmt(call.x ?: 0.5)},${fmt(call.y ?: 0.5)})", afState, status,
                        if (status == Status.VERIFIED_CLOSEST) "AF settled without confirmed focus lock" else null)
                }
                ToolIds.TORCH -> {
                    val target = (call.value ?: 0.0) >= 0.5
                    val ok = snapshot.torchEnabled == target
                    if (!ok) retry += call
                    Item(call.id, target.toString(), snapshot.torchEnabled.toString(), if (ok) Status.VERIFIED else Status.FAILED)
                }
                ToolIds.FLASH -> {
                    val target = StillFlashMode.fromWire(call.stringValue) ?: StillFlashMode.OFF
                    val ok = snapshot.flashMode == target
                    if (!ok) retry += call
                    Item(call.id, target.wireName, snapshot.flashMode.wireName, if (ok) Status.VERIFIED else Status.FAILED)
                }
                ToolIds.EXTENSION -> {
                    val target = DirectorExtension.fromWire(call.stringValue) ?: DirectorExtension.NONE
                    val ok = snapshot.extension == target
                    if (!ok) retry += call
                    Item(call.id, target.wireName, snapshot.extension.wireName, if (ok) Status.VERIFIED else Status.FAILED)
                }
                else -> Item(call.id, "n/a", "n/a", Status.UNKNOWN)
            }
        }

        return Report(items, afState, aeState, awbState, settled, retry.distinctBy { it.id })
    }

    private fun fmt(v: Double): String = String.format(Locale.US, "%.3f", v)
    private fun afName(v: Int?): String = when (v) {
        null -> "unknown"
        CaptureResult.CONTROL_AF_STATE_INACTIVE -> "inactive"
        CaptureResult.CONTROL_AF_STATE_PASSIVE_SCAN -> "passive_scan"
        CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED -> "passive_focused"
        CaptureResult.CONTROL_AF_STATE_ACTIVE_SCAN -> "active_scan"
        CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED -> "focused_locked"
        CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED -> "not_focused_locked"
        CaptureResult.CONTROL_AF_STATE_PASSIVE_UNFOCUSED -> "passive_unfocused"
        else -> "state_$v"
    }
    private fun aeName(v: Int?): String = when (v) {
        null -> "unknown"
        CaptureResult.CONTROL_AE_STATE_INACTIVE -> "inactive"
        CaptureResult.CONTROL_AE_STATE_SEARCHING -> "searching"
        CaptureResult.CONTROL_AE_STATE_CONVERGED -> "converged"
        CaptureResult.CONTROL_AE_STATE_LOCKED -> "locked"
        CaptureResult.CONTROL_AE_STATE_FLASH_REQUIRED -> "flash_required"
        CaptureResult.CONTROL_AE_STATE_PRECAPTURE -> "precapture"
        else -> "state_$v"
    }
    private fun awbName(v: Int?): String = when (v) {
        null -> "unknown"
        CaptureResult.CONTROL_AWB_STATE_INACTIVE -> "inactive"
        CaptureResult.CONTROL_AWB_STATE_SEARCHING -> "searching"
        CaptureResult.CONTROL_AWB_STATE_CONVERGED -> "converged"
        CaptureResult.CONTROL_AWB_STATE_LOCKED -> "locked"
        else -> "state_$v"
    }

    companion object {
        private val AF_GOOD = setOf(CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED, CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED)
        private val AF_SEARCHING = setOf(CaptureResult.CONTROL_AF_STATE_ACTIVE_SCAN, CaptureResult.CONTROL_AF_STATE_PASSIVE_SCAN)
        private val AF_SETTLED = AF_GOOD + setOf(CaptureResult.CONTROL_AF_STATE_INACTIVE, CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED, CaptureResult.CONTROL_AF_STATE_PASSIVE_UNFOCUSED)
        private val AE_SETTLED = setOf(CaptureResult.CONTROL_AE_STATE_INACTIVE, CaptureResult.CONTROL_AE_STATE_CONVERGED, CaptureResult.CONTROL_AE_STATE_LOCKED, CaptureResult.CONTROL_AE_STATE_FLASH_REQUIRED)
        private val AWB_SETTLED = setOf(CaptureResult.CONTROL_AWB_STATE_INACTIVE, CaptureResult.CONTROL_AWB_STATE_CONVERGED, CaptureResult.CONTROL_AWB_STATE_LOCKED)
    }
}
