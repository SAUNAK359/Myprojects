package com.saunak.director.tools

import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.camera.DirectorExtension
import com.saunak.director.camera.StillFlashMode
import com.saunak.director.director.ToolCall

class ToolValidator {
    data class Rejection(val call: ToolCall, val reason: String)
    data class Result(val accepted: List<ToolCall>, val rejected: List<Rejection>)

    fun validate(input: List<ToolCall>, caps: CapabilityRegistry): Result {
        val accepted = mutableListOf<ToolCall>()
        val rejected = mutableListOf<Rejection>()
        val calls = input.take(8)
        if (input.size > 8) input.drop(8).forEach { rejected += Rejection(it, "too_many_calls") }
        var chosenLook: ToolCall? = null
        var requestedExtension: DirectorExtension? = null

        for (call in calls) {
            if (call.id !in caps.allowedToolIds && call.id !in ToolIds.LOOK_IDS) { rejected += Rejection(call, "unsupported_tool"); continue }
            if (caps.activeExtension != DirectorExtension.NONE && call.id in ToolIds.CAMERA_IDS && call.id != ToolIds.EXTENSION) {
                rejected += Rejection(call, "extension_active")
                continue
            }
            val normalized = when (call.id) {
                ToolIds.FOCUS -> {
                    val x = call.x; val y = call.y
                    if (!finite(x) || !finite(y)) null else call.copy(x = x!!.coerceIn(0.0, 1.0), y = y!!.coerceIn(0.0, 1.0))
                }
                ToolIds.ZOOM -> if (!finite(call.value)) null else call.copy(value = call.value!!.coerceIn(caps.minZoom.toDouble(), caps.maxZoom.toDouble()))
                ToolIds.EXPOSURE_EV -> if (!finite(call.value) || caps.exposureStepEv <= 0f) null else call.copy(value = call.value!!.coerceIn(caps.minExposureEv.toDouble(), caps.maxExposureEv.toDouble()))
                ToolIds.TORCH -> if (!finite(call.value)) null else call.copy(value = if (call.value!! >= 0.5) 1.0 else 0.0)
                ToolIds.FLASH -> StillFlashMode.fromWire(call.stringValue)?.let { call.copy(stringValue = it.wireName) }
                ToolIds.EXTENSION -> {
                    val extension = DirectorExtension.fromWire(call.stringValue)
                    val supported = extension != null && (extension in caps.extensions || (extension == DirectorExtension.NONE && caps.activeExtension != DirectorExtension.NONE))
                    if (!supported) null else call.copy(stringValue = extension!!.wireName).also { requestedExtension = extension }
                }
                in ToolIds.LOOK_IDS -> if (call.value == null || finite(call.value)) call.copy(value = call.value?.coerceIn(0.0, 1.0)) else null
                else -> null
            }
            if (normalized == null) { rejected += Rejection(call, "invalid_parameters"); continue }
            if (normalized.id in ToolIds.LOOK_IDS) {
                chosenLook?.let { rejected += Rejection(it, "superseded_look") }
                chosenLook = normalized
            } else accepted += normalized
        }
        chosenLook?.let { accepted += it }

        if (requestedExtension != null && requestedExtension != DirectorExtension.NONE) {
            val conflicting = accepted.filter { it.id in ToolIds.CAMERA_IDS && it.id != ToolIds.EXTENSION }
            if (conflicting.isNotEmpty()) {
                accepted.removeAll(conflicting.toSet())
                conflicting.forEach { rejected += Rejection(it, "extension_conflict") }
            }
        }

        val flash = accepted.firstOrNull { it.id == ToolIds.FLASH && it.stringValue != StillFlashMode.OFF.wireName }
        val torchOn = accepted.firstOrNull { it.id == ToolIds.TORCH && (it.value ?: 0.0) >= 0.5 }
        if (flash != null && torchOn != null) {
            accepted.remove(torchOn)
            rejected += Rejection(torchOn, "flash_torch_conflict")
        }
        return Result(accepted, rejected)
    }

    private fun finite(value: Double?): Boolean = value != null && value.isFinite()
}
