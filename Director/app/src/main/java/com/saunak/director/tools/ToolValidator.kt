package com.saunak.director.tools

import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.camera.DirectorExtension
import com.saunak.director.camera.StillFlashMode
import com.saunak.director.director.ToolCall
import kotlin.math.roundToInt

/**
 * Physical/API feasibility validator. Sierra v11 MVP deliberately rejects OEM extension execution:
 * CameraX extension sessions do not expose the analysis stream required by the deterministic Live
 * Director. Extensions stay discoverable as device capability, but they cannot own the active v11
 * capture session until a device-qualified dual-session/analysis-safe path exists.
 */
class ToolValidator {
    data class Rejection(val call: ToolCall, val reason: String)
    data class Result(val accepted: List<ToolCall>, val rejected: List<Rejection>)

    fun validate(input: List<ToolCall>, caps: CapabilityRegistry): Result {
        val accepted = mutableListOf<ToolCall>()
        val rejected = mutableListOf<Rejection>()
        val calls = input.take(8)
        if (input.size > 8) input.drop(8).forEach { rejected += Rejection(it, "too_many_calls") }

        val duplicateCameraIds = calls.filter { it.id in ToolIds.CAMERA_IDS }.groupBy { it.id }.filterValues { it.size > 1 }.keys
        val looks = calls.filter { it.id in ToolIds.LOOK_IDS }
        val flashOn = calls.any { it.id == ToolIds.FLASH && it.stringValue != StillFlashMode.OFF.wireName }
        val torchOn = calls.any { it.id == ToolIds.TORCH && (it.value ?: 0.0) >= .5 }

        if (duplicateCameraIds.isNotEmpty() || looks.size > 1 || (flashOn && torchOn)) {
            calls.forEach { call ->
                val ambiguous = call.id in duplicateCameraIds ||
                    (looks.size > 1 && call.id in ToolIds.LOOK_IDS) ||
                    ((flashOn && torchOn) && call.id in setOf(ToolIds.FLASH, ToolIds.TORCH))
                if (ambiguous) rejected += Rejection(call, "ambiguous_plan")
            }
        }

        for (call in calls) {
            if (rejected.any { it.call === call || it.call == call }) continue
            if (call.id == ToolIds.EXTENSION) {
                rejected += Rejection(call, "oem_extension_disabled_for_v11_live_director")
                continue
            }
            if (call.id !in caps.allowedToolIds && call.id !in ToolIds.LOOK_IDS) {
                rejected += Rejection(call, "unsupported_tool")
                continue
            }
            val normalized = when (call.id) {
                ToolIds.FOCUS -> {
                    val x = call.x; val y = call.y
                    if (!finite(x) || !finite(y) || !caps.focusMeteringSupported) null
                    else call.copy(x = x!!.coerceIn(0.0, 1.0), y = y!!.coerceIn(0.0, 1.0))
                }
                ToolIds.ZOOM -> if (!finite(call.value)) null
                    else call.copy(value = call.value!!.coerceIn(caps.minZoom.toDouble(), caps.maxZoom.toDouble()))
                ToolIds.EXPOSURE_EV -> if (!finite(call.value) || caps.exposureStepEv <= 0f) null
                    else {
                        val step = caps.exposureStepEv.toDouble()
                        val desired = call.value!!.coerceIn(caps.minExposureEv.toDouble(), caps.maxExposureEv.toDouble())
                        val index = (desired / step).roundToInt().coerceIn(caps.exposureMinIndex, caps.exposureMaxIndex)
                        call.copy(value = index * step)
                    }
                ToolIds.TORCH -> if (!finite(call.value) || !caps.torchSupported) null
                    else call.copy(value = if (call.value!! >= 0.5) 1.0 else 0.0)
                ToolIds.FLASH -> if (!caps.hasFlash) null
                    else StillFlashMode.fromWire(call.stringValue)?.let { call.copy(stringValue = it.wireName) }
                ToolIds.EXTENSION -> null
                in ToolIds.LOOK_IDS -> if (call.value == null || finite(call.value)) call.copy(value = call.value?.coerceIn(0.0, 1.0)) else null
                else -> null
            }
            if (normalized == null) rejected += Rejection(call, "invalid_or_unsupported_parameters")
            else accepted += normalized
        }
        return Result(accepted, rejected)
    }

    private fun finite(value: Double?): Boolean = value != null && value.isFinite()
}
