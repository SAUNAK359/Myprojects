package com.saunak.director.engine.prif

import com.saunak.director.engine.reconstruction.RawImageFormation
import com.saunak.director.engine.reconstruction.RawObservation
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

object SensorConsistencyVerifier {
    fun verify(
        posterior: RadiancePosterior,
        observations: List<RawObservation>,
        acceptanceThreshold: Double = 0.62
    ): SensorConsistencyReport {
        require(observations.isNotEmpty())
        require(acceptanceThreshold in 0.0..1.0)
        val referenceEnergy = observations.first().exposureTimeNs.toDouble() * observations.first().sensitivityIso.toDouble()
        val frameReports = observations.mapIndexed { frameIndex, obs ->
            require(obs.raw.width == posterior.width && obs.raw.height == posterior.height)
            val actual = RawImageFormation.normalize(obs.raw)
            val energy = obs.exposureTimeNs.toDouble() * obs.sensitivityIso.toDouble()
            val forwardScale = (energy / referenceEnergy).coerceIn(0.0625, 16.0)
            var squared = 0.0
            var inliers = 0
            var compared = 0
            var saturated = 0
            var i = 0
            val pixelCount = posterior.width.toLong() * posterior.height.toLong()
            val stride = when {
                pixelCount > 12_000_000L -> 16
                pixelCount > 4_000_000L -> 8
                else -> 4
            }
            while (i < actual.size) {
                val predicted = (posterior.mean[i].toDouble() * forwardScale).coerceIn(0.0, 1.0)
                val measured = actual[i].toDouble()
                if (measured >= 0.995) saturated++
                val variance = max(obs.noise.variance(measured.toFloat()).toDouble(), 1e-8)
                val z = abs(measured - predicted) / sqrt(variance)
                squared += z * z
                if (z <= 3.0) inliers++
                compared++
                i += stride
            }
            val normalizedRmse = if (compared > 0) sqrt(squared / compared) else Double.POSITIVE_INFINITY
            val inlierFraction = if (compared > 0) inliers.toDouble() / compared else 0.0
            val saturationFraction = if (compared > 0) saturated.toDouble() / compared else 1.0
            val residualScore = exp(-0.18 * normalizedRmse * normalizedRmse).coerceIn(0.0, 1.0)
            val score = (0.62 * residualScore + 0.30 * inlierFraction +
                0.08 * (1.0 - saturationFraction)).coerceIn(0.0, 1.0)
            FrameConsistency(frameIndex, normalizedRmse, inlierFraction, saturationFraction, score)
        }
        val aggregate = frameReports.map { it.score }.average().coerceIn(0.0, 1.0)
        return SensorConsistencyReport(frameReports, aggregate, aggregate >= acceptanceThreshold)
    }
}

object PhotoIntegrityEvaluator {
    fun evaluate(
        posterior: RadiancePosterior,
        consistency: SensorConsistencyReport,
        uncertaintyThreshold: Double = 0.02,
        acceptanceThreshold: Double = 0.58
    ): PhotoIntegrityReport {
        require(uncertaintyThreshold > 0.0)
        require(acceptanceThreshold in 0.0..1.0)
        var multi = 0
        var single = 0
        var uncertain = 0
        val n = posterior.mean.size.coerceAtLeast(1)
        for (i in posterior.mean.indices) {
            val count = posterior.supportCount[i].toInt() and 0xff
            if (count >= 2) multi++ else single++
            val variance = posterior.sensorVariance[i] + posterior.reconstructionVariance[i]
            if (variance > uncertaintyThreshold) uncertain++
        }
        val provenance = ProvenanceSummary(
            averagePhysicalSupport = posterior.meanSupport.coerceIn(0.0, 1.0),
            multiFrameFraction = multi.toDouble() / n,
            uncertainFraction = uncertain.toDouble() / n,
            singleFrameFraction = single.toDouble() / n,
            neuralFraction = 0.0
        )
        val uncertaintyPenalty = (provenance.uncertainFraction * 0.72 +
            posterior.meanUncertainty.coerceIn(0.0, 1.0) * 0.28).coerceIn(0.0, 1.0)
        val score = (0.46 * consistency.aggregateScore +
            0.40 * provenance.averagePhysicalSupport +
            0.14 * provenance.multiFrameFraction -
            0.22 * uncertaintyPenalty).coerceIn(0.0, 1.0)
        return PhotoIntegrityReport(
            score,
            consistency.aggregateScore,
            provenance.averagePhysicalSupport,
            uncertaintyPenalty,
            provenance,
            consistency.accepted && score >= acceptanceThreshold
        )
    }
}
