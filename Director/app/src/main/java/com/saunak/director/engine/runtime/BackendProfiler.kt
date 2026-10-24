package com.saunak.director.engine.runtime

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.os.PowerManager
import com.saunak.director.engine.device.BackendExecutionProfile
import com.saunak.director.engine.device.ComputeBackend
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/**
 * Measures the executable graph that actually runs on the device. This never assumes NNAPI/NPU/GPU
 * is faster merely because session creation succeeded.
 */
class BackendProfiler(
    private val context: Context,
    private val store: RuntimeProfileStore,
    private val runtimeVersion: String
) {
    private val power = context.getSystemService(PowerManager::class.java)

    suspend fun profile(
        deviceFingerprint: String,
        modelId: String,
        inputSignature: String,
        backend: ComputeBackend,
        graphCoverage: Double,
        transferBoundaries: Int,
        warmups: Int = 2,
        repeats: Int = 7,
        execute: suspend () -> Unit
    ): MeasuredExecutionProfile {
        require(deviceFingerprint.isNotBlank())
        require(modelId.isNotBlank() && inputSignature.isNotBlank())
        require(graphCoverage in 0.0..1.0)
        require(transferBoundaries >= 0)
        require(warmups in 0..10)
        require(repeats in 3..30)

        repeat(warmups) { execute() }
        val thermalBefore = thermalStatus()
        val memoryBefore = processMemoryMb()
        val latencies = DoubleArray(repeats)
        var peakMemory = memoryBefore

        for (i in 0 until repeats) {
            val start = System.nanoTime()
            execute()
            latencies[i] = (System.nanoTime() - start) / 1_000_000.0
            peakMemory = max(peakMemory, processMemoryMb())
        }

        latencies.sort()
        val p50 = percentile(latencies, 0.50)
        val p95 = percentile(latencies, 0.95)
        val thermalAfter = thermalStatus()
        val thermalEscalation = max(0, thermalAfter - thermalBefore)
        // Proxy only: combines sustained latency with observed Android thermal escalation.
        val thermalPowerProxy = (p95 / 80.0 + thermalEscalation * 2.0).coerceIn(0.0, 10.0)
        val peakDelta = max(0.0, peakMemory - memoryBefore)

        val profile = BackendExecutionProfile(
            modelId = modelId,
            backend = backend,
            inputSignature = inputSignature,
            graphCoverage = graphCoverage,
            transferBoundaries = transferBoundaries,
            p50LatencyMs = p50,
            p95LatencyMs = max(p50, p95),
            peakMemoryMb = peakDelta,
            sustainedPowerScore = thermalPowerProxy
        )
        return MeasuredExecutionProfile(
            key = ProfileKey(deviceFingerprint, modelId, inputSignature, backend),
            profile = profile,
            samples = repeats,
            measuredAtEpochMs = System.currentTimeMillis(),
            runtimeVersion = runtimeVersion,
            valid = true
        ).also(store::upsert)
    }

    private fun thermalStatus(): Int = runCatching { power.currentThermalStatus }.getOrDefault(0)

    private fun processMemoryMb(): Double {
        val pssKb = runCatching { Debug.getPss() }.getOrDefault(0L)
        if (pssKb > 0L) return pssKb / 1024.0
        val runtime = Runtime.getRuntime()
        return (runtime.totalMemory() - runtime.freeMemory()).toDouble() / (1024.0 * 1024.0)
    }

    private fun percentile(sorted: DoubleArray, q: Double): Double {
        require(sorted.isNotEmpty())
        val index = ceil((sorted.size - 1) * q).toInt().coerceIn(0, sorted.lastIndex)
        return sorted[index]
    }
}
