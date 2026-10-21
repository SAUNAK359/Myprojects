package com.saunak.director.tools

import com.saunak.director.camera.CameraEngine
import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.camera.DirectorExtension
import com.saunak.director.camera.StillFlashMode
import com.saunak.director.director.ToolCall
import com.saunak.director.processing.Look
import com.saunak.director.processing.LookSpec
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class ToolExecutor(private val camera: CameraEngine, private val validator: ToolValidator = ToolValidator()) {
    data class Rejected(val call: ToolCall, val reason: String)
    data class Result(val applied: List<ToolCall>, val rejected: List<Rejected>, val look: LookSpec, val requiresRecapture: Boolean)
    private val mutationMutex = Mutex()

    suspend fun apply(calls: List<ToolCall>, capabilities: CapabilityRegistry, currentLook: LookSpec = LookSpec()): Result = mutationMutex.withLock {
        val validated = validator.validate(calls, capabilities)
        val applied = mutableListOf<ToolCall>()
        val rejected = validated.rejected.mapTo(mutableListOf()) { Rejected(it.call, it.reason) }
        var look = currentLook
        val ordered = validated.accepted.sortedBy { if (it.id == ToolIds.EXTENSION) 0 else if (it.id in ToolIds.LOOK_IDS) 2 else 1 }
        for (call in ordered) {
            runCatching {
                when (call.id) {
                    ToolIds.FOCUS -> camera.aiFocusNormalized(call.x!!.toFloat(), call.y!!.toFloat())
                    ToolIds.ZOOM -> camera.aiSetZoom(call.value!!.toFloat())
                    ToolIds.EXPOSURE_EV -> camera.aiSetExposureEv(call.value!!.toFloat())
                    ToolIds.TORCH -> camera.aiSetTorch(call.value!! >= 0.5)
                    ToolIds.FLASH -> camera.aiSetFlash(checkNotNull(StillFlashMode.fromWire(call.stringValue)))
                    ToolIds.EXTENSION -> camera.aiSetExtension(checkNotNull(DirectorExtension.fromWire(call.stringValue)))
                    ToolIds.LOOK_NATURAL -> look = LookSpec(Look.NATURAL, call.value?.toFloat() ?: 0f)
                    ToolIds.LOOK_WARM -> look = LookSpec(Look.WARM, call.value?.toFloat() ?: .65f)
                    ToolIds.LOOK_COOL -> look = LookSpec(Look.COOL, call.value?.toFloat() ?: .65f)
                    ToolIds.LOOK_CINEMATIC_WARM -> look = LookSpec(Look.CINEMATIC_WARM, call.value?.toFloat() ?: .65f)
                    ToolIds.LOOK_CINEMATIC_COOL -> look = LookSpec(Look.CINEMATIC_COOL, call.value?.toFloat() ?: .65f)
                    ToolIds.LOOK_MONOCHROME -> look = LookSpec(Look.MONOCHROME, call.value?.toFloat() ?: 1f)
                    ToolIds.LOOK_SOFT_PORTRAIT -> look = LookSpec(Look.SOFT_PORTRAIT, call.value?.toFloat() ?: .6f)
                    ToolIds.LOOK_LANDSCAPE_VIVID -> look = LookSpec(Look.LANDSCAPE_VIVID, call.value?.toFloat() ?: .55f)
                    else -> error("Unknown tool ${call.id}")
                }
            }.onSuccess { applied += call }.onFailure { rejected += Rejected(call, it.message ?: "execution_failed") }
        }
        val requiresRecapture = applied.any { it.id in setOf(ToolIds.FOCUS, ToolIds.ZOOM, ToolIds.EXPOSURE_EV, ToolIds.TORCH, ToolIds.FLASH, ToolIds.EXTENSION) }
        Result(applied, rejected, look, requiresRecapture)
    }

    suspend fun restoreUserState() = mutationMutex.withLock { camera.restoreUserState() }
}
