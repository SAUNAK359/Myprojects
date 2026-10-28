package com.saunak.director.tools

import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.camera.DirectorExtension
import com.saunak.director.camera.StillFlashMode
import com.saunak.director.director.ToolCall

/**
 * Physical/API feasibility validator. It normalizes only to real device bounds; it does not make
 * creative choices or reject direct Sierra controls merely because an OEM extension is active.
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
        val extensionCall = calls.firstOrNull { it.id == ToolIds.EXTENSION }
        val nonNoneExtension = DirectorExtension.fromWire(extensionCall?.stringValue)?.takeIf { it != DirectorExtension.NONE }
        val directControls = calls.filter { it.id in ToolIds.CAMERA_IDS && it.id != ToolIds.EXTENSION }
        val flashOn = calls.any { it.id == ToolIds.FLASH && it.stringValue != StillFlashMode.OFF.wireName }
        val torchOn = calls.any { it.id == ToolIds.TORCH && (it.value ?: 0.0) >= .5 }

        if (duplicateCameraIds.isNotEmpty() || looks.size > 1 || (nonNoneExtension != null && directControls.isNotEmpty()) || (flashOn && torchOn)) {
            calls.forEach { call ->
                val ambiguous = call.id in duplicateCameraIds ||
                    (looks.size > 1 && call.id in ToolIds.LOOK_IDS) ||
                    (nonNoneExtension != null && call.id in ToolIds.CAMERA_IDS) ||
                    ((flashOn && torchOn) && call.id in setOf(ToolIds.FLASH, ToolIds.TORCH))
                if (ambiguous) rejected += Rejection(call, "ambiguous_plan")
            }
        }

        for (call in calls) {
            if (rejected.any { it.call === call || it.call == call }) continue
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
                    else call.copy(value = call.value!!.coerceIn(caps.minExposureEv.toDouble(), caps.maxExposureEv.toDouble()))
                ToolIds.TORCH -> if (!finite(call.value) || !caps.torchSupported) null
                    else call.copy(value = if (call.value!! >= 0.5) 1.0 else 0.0)
                ToolIds.FLASH -> if (!caps.hasFlash) null
                    else StillFlashMode.fromWire(call.stringValue)?.let { call.copy(stringValue = it.wireName) }
                ToolIds.EXTENSION -> {
                    val extension = DirectorExtension.fromWire(call.stringValue)
                    val supported = extension != null && (extension == DirectorExtension.NONE || extension in caps.extensions)
                    if (!supported) null else call.copy(stringValue = extension!!.wireName)
                }
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
