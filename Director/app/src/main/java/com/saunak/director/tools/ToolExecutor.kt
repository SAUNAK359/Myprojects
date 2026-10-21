package com.saunak.director.tools

import com.saunak.director.camera.CameraEngine
import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.director.ToolCall
import com.saunak.director.processing.Look
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class ToolExecutor(private val camera: CameraEngine) {
    data class Result(val rejected: List<ToolCall>, val look: Look = Look.NATURAL)
    private val mutationMutex = Mutex()
    suspend fun apply(calls: List<ToolCall>, capabilities: CapabilityRegistry, currentLook: Look = Look.NATURAL): Result = mutationMutex.withLock {
        var look = currentLook; val rejected = mutableListOf<ToolCall>()
        calls.distinctBy { it.id }.forEach { call ->
            if (call.id !in capabilities.allowedToolIds) { rejected += call; return@forEach }
            runCatching {
                when (call.id) {
                    1 -> Unit
                    2 -> camera.setZoomAwait((call.value ?: 1.0).toFloat())
                    3 -> camera.setExposureIndexAwait((call.value ?: 0.0).toInt())
                    4 -> camera.setTorchAwait((call.value ?: 0.0) >= 0.5)
                    30 -> look = Look.NATURAL; 31 -> look = Look.WARM; 32 -> look = Look.COOL; 33 -> look = Look.CINEMATIC_WARM
                    34 -> look = Look.CINEMATIC_COOL; 35 -> look = Look.MONOCHROME; 36 -> look = Look.SOFT_PORTRAIT; 37 -> look = Look.LANDSCAPE_VIVID
                    100 -> Unit
                    else -> error("Unknown tool ${call.id}")
                }
            }.onFailure { rejected += call }
        }
        Result(rejected, look)
    }
}
