package com.saunak.director.engine.prif

import com.saunak.director.engine.reconstruction.RawImageFormation
import com.saunak.director.engine.reconstruction.RawObservation
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Estimates visibility/occlusion support directly from exposure-normalized RAW disagreement. This is
 * intentionally model-free: semantic masks may further restrict support, but cannot manufacture it.
 */
object PrifOcclusionEstimator {
    data class Result(
        val visibility: List<FloatArray>,
        val meanOcclusionProbability: Double
    ) {
        init {
            require(visibility.isNotEmpty())
            val size = visibility.first().size
            require(visibility.all { it.size == size && it.all { v -> v in 0f..1f } })
            require(meanOcclusionProbability in 0.0..1.0)
        }
    }

    fun estimate(observations: List<RawObservation>): Result {
        require(observations.isNotEmpty())
        val reference = observations.first()
        val normalized = observations.map { RawImageFormation.normalize(it.raw) }
        val referenceEnergy = reference.exposureTimeNs.toDouble() * reference.sensitivityIso.toDouble()
        val output = observations.map { FloatArray(reference.raw.values.size) { 1f } }
        var occlusionSum = 0.0
        var occlusionCount = 0L

        for (j in 1 until observations.size) {
            val obs = observations[j]
            val energy = obs.exposureTimeNs.toDouble() * obs.sensitivityIso.toDouble()
            val exposureScale = (referenceEnergy / energy).coerceIn(0.0625, 16.0)
            val dst = output[j]
            for (i in dst.indices) {
                val ref = normalized[0][i].toDouble()
                val value = normalized[j][i].toDouble() * exposureScale
                val vRef = reference.noise.variance(ref.toFloat()).toDouble()
                val vObs = obs.noise.variance(normalized[j][i]).toDouble() * exposureScale * exposureScale
                val sigma = sqrt(max(vRef + vObs, 1e-8))
                val z = abs(value - ref) / max(sigma, 1e-5)
                // Smooth visibility posterior: near-noise residuals remain fully supported; large
                // residuals collapse toward zero without a brittle binary ghost mask.
                val visibility = (1.0 / (1.0 + exp((z - CENTER_Z) * SLOPE))).coerceIn(MIN_VISIBILITY, 1.0)
                dst[i] = visibility.toFloat()
                occlusionSum += 1.0 - visibility
                occlusionCount++
            }
        }
        val mean = if (occlusionCount > 0L) occlusionSum / occlusionCount else 0.0
        return Result(output, mean.coerceIn(0.0, 1.0))
    }

    private const val CENTER_Z = 3.2
    private const val SLOPE = 1.35
    private const val MIN_VISIBILITY = 0.04
}

object RollingShutterEvidenceModel {
    /**
     * Confidence that one frame can be treated as globally aligned. Until per-row gyro integration is
     * available, rolling-shutter skew is represented honestly as uncertainty instead of fake correction.
     */
    fun globalAlignmentConfidence(
        rollingShutterSkewNs: Long?,
        motionRisk: Double
    ): Double {
        if (rollingShutterSkewNs == null) return (0.88 - 0.28 * motionRisk).coerceIn(0.35, 0.88)
        val skewMs = rollingShutterSkewNs / 1_000_000.0
        val temporalRisk = (skewMs / 33.0).coerceIn(0.0, 1.5)
        return (1.0 / (1.0 + 1.15 * temporalRisk * motionRisk)).coerceIn(0.20, 1.0)
    }
}
