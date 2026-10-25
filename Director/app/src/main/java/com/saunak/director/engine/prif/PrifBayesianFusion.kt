package com.saunak.director.engine.prif

import com.saunak.director.engine.reconstruction.RawImageFormation
import com.saunak.director.engine.reconstruction.RawObservation
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

object PrifBayesianFusion {
    fun fuse(
        observations: List<RawObservation>,
        degreesOfFreedom: Double,
        maxNormalizedResidual: Double,
        conservativeAnchor: Boolean,
        spatialSupport: ((frameIndex: Int, x: Int, y: Int) -> Double)? = null
    ): RadiancePosterior {
        require(observations.isNotEmpty())
        require(degreesOfFreedom in 1.0..30.0)
        require(maxNormalizedResidual in 1.0..12.0)
        val reference = observations.first().raw
        require(observations.size <= 255)
        require(observations.all {
            it.raw.width == reference.width && it.raw.height == reference.height && it.raw.pattern == reference.pattern
        })
        val normalized = observations.map { RawImageFormation.normalize(it.raw) }
        val referenceEnergy = observations.first().exposureTimeNs.toDouble() * observations.first().sensitivityIso.toDouble()
        val n = reference.values.size
        val mean = FloatArray(n)
        val sensorVariance = FloatArray(n)
        val reconstructionVariance = FloatArray(n)
        val support = FloatArray(n)
        val dominantFrame = ByteArray(n)
        val supportCount = ByteArray(n)
        for (i in 0 until n) {
            val x = i % reference.width
            val y = i / reference.width
            val anchor = normalized.first()[i].toDouble()
            var weighted = 0.0
            var weightSum = 0.0
            var precisionSum = 0.0
            var dominant = 0
            var dominantWeight = -1.0
            var accepted = 0
            val values = DoubleArray(observations.size)
            val weights = DoubleArray(observations.size)
            for (j in observations.indices) {
                val obs = observations[j]
                val energy = obs.exposureTimeNs.toDouble() * obs.sensitivityIso.toDouble()
                val exposureScale = (referenceEnergy / energy).coerceIn(0.0625, 16.0)
                val value = (normalized[j][i].toDouble() * exposureScale).coerceAtLeast(0.0)
                values[j] = value
                val nativeVariance = obs.noise.variance(normalized[j][i]).toDouble()
                val scaledVariance = max(nativeVariance * exposureScale * exposureScale, 1e-9)
                val anchorVariance = max(obs.noise.variance(anchor.toFloat()).toDouble(), 1e-9)
                val sigma = sqrt(scaledVariance + anchorVariance)
                val z = abs(value - anchor) / max(sigma, 1e-5)
                if (z > maxNormalizedResidual && j != 0) continue
                val student = (degreesOfFreedom + 1.0) / (degreesOfFreedom + z * z)
                var physicalSupport = obs.alignmentConfidence.toDouble() *
                    obs.motionConfidence.toDouble() * obs.saturationConfidence.toDouble()
                if (conservativeAnchor && j != 0) physicalSupport *= 0.72
                physicalSupport *= spatialSupport?.invoke(j, x, y)?.coerceIn(0.0, 1.0) ?: 1.0
                val precision = 1.0 / scaledVariance
                val weight = physicalSupport * student * precision
                if (weight <= 0.0 || !weight.isFinite()) continue
                weights[j] = weight
                weighted += value * weight
                weightSum += weight
                precisionSum += physicalSupport * student * precision
                accepted++
                if (weight > dominantWeight) {
                    dominantWeight = weight
                    dominant = j
                }
            }
            val fused = if (weightSum > 1e-12) weighted / weightSum else anchor
            mean[i] = fused.toFloat().coerceAtLeast(0f)
            sensorVariance[i] = if (precisionSum > 1e-12) (1.0 / precisionSum).toFloat() else 1f
            var residualVariance = 0.0
            if (weightSum > 1e-12) {
                for (j in observations.indices) {
                    val w = weights[j]
                    if (w <= 0.0) continue
                    val d = values[j] - fused
                    residualVariance += w * d * d
                }
                residualVariance /= weightSum
            } else residualVariance = 1.0
            reconstructionVariance[i] = residualVariance.toFloat().coerceAtLeast(0f)
            val countFactor = (accepted.toDouble() / observations.size.toDouble()).coerceIn(0.0, 1.0)
            val uncertainty = sqrt(sensorVariance[i].toDouble() + reconstructionVariance[i].toDouble())
            val certaintyFactor = (1.0 / (1.0 + 8.0 * uncertainty)).coerceIn(0.0, 1.0)
            support[i] = (countFactor * certaintyFactor).toFloat()
            dominantFrame[i] = dominant.toByte()
            supportCount[i] = accepted.coerceIn(0, 255).toByte()
        }
        return RadiancePosterior(reference.width, reference.height, mean, sensorVariance,
            reconstructionVariance, support, dominantFrame, supportCount, reference.pattern)
    }
}
