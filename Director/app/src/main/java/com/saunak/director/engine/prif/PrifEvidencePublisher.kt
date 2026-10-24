package com.saunak.director.engine.prif

import com.saunak.director.camera.RawJpegCaptureFrame
import com.saunak.director.engine.reconstruction.EvidenceLedger
import com.saunak.director.engine.reconstruction.EvidenceRecord
import com.saunak.director.engine.reconstruction.NormalizedRect
import com.saunak.director.engine.reconstruction.RegionId
import com.saunak.director.engine.reconstruction.RegionSemantic
import kotlin.math.max
import kotlin.math.min

/** Bridges compact P-RIF pixel provenance into Sierra's existing region-level evidence ledger. */
object PrifEvidencePublisher {
    fun publish(
        ledger: EvidenceLedger,
        result: PrifReconstructionResult,
        captured: List<RawJpegCaptureFrame>,
        tileSize: Int = 256,
        timestampNs: Long = System.nanoTime()
    ) {
        require(tileSize in 64..1024)
        require(timestampNs > 0)
        val posterior = result.posterior
        val frameIds = result.usedCaptureIndices.map { index ->
            captured[index].raw.timestampNs
        }
        if (frameIds.isEmpty()) return

        var y0 = 0
        while (y0 < posterior.height) {
            val h = min(tileSize, posterior.height - y0)
            var x0 = 0
            while (x0 < posterior.width) {
                val w = min(tileSize, posterior.width - x0)
                var supportSum = 0.0
                var sensorVariance = 0.0
                var reconstructionVariance = 0.0
                var multiFrame = 0
                var pixels = 0
                val dominantCounts = HashMap<Long, Double>()

                for (y in y0 until y0 + h) {
                    for (x in x0 until x0 + w) {
                        val i = y * posterior.width + x
                        supportSum += posterior.support[i]
                        sensorVariance += posterior.sensorVariance[i]
                        reconstructionVariance += posterior.reconstructionVariance[i]
                        val supportCount = posterior.supportCount[i].toInt() and 0xff
                        if (supportCount >= 2) multiFrame++
                        val local = (posterior.dominantFrame[i].toInt() and 0xff).coerceIn(frameIds.indices)
                        val frameId = frameIds[local]
                        dominantCounts[frameId] = (dominantCounts[frameId] ?: 0.0) + max(1e-5, posterior.support[i].toDouble())
                        pixels++
                    }
                }

                val count = max(1, pixels).toDouble()
                val meanSupport = (supportSum / count).coerceIn(0.0, 1.0)
                val meanSensorVariance = sensorVariance / count
                val meanReconstructionVariance = reconstructionVariance / count
                val uncertainty = (meanSensorVariance + meanReconstructionVariance).coerceAtLeast(0.0)
                val snrConfidence = (1.0 / (1.0 + 12.0 * meanSensorVariance)).coerceIn(0.0, 1.0)
                val geometryConfidence = (1.0 - result.meanOcclusionProbability).coerceIn(0.0, 1.0)
                val multiFrameFraction = multiFrame / count
                val weights = if (dominantCounts.values.sum() > 0.0) dominantCounts else hashMapOf(frameIds.first() to 1.0)

                ledger.put(
                    EvidenceRecord(
                        regionId = RegionId("prif/tile/${x0}_${y0}_$timestampNs"),
                        bounds = NormalizedRect(
                            left = x0.toDouble() / posterior.width,
                            top = y0.toDouble() / posterior.height,
                            right = (x0 + w).toDouble() / posterior.width,
                            bottom = (y0 + h).toDouble() / posterior.height
                        ),
                        semantic = RegionSemantic.GENERAL,
                        sourceFrameIds = frameIds.toSet(),
                        fusionWeights = weights,
                        alignmentConfidence = result.consistency.aggregateScore,
                        motionConfidence = (1.0 - 0.75 * result.program.subjectProtection).coerceIn(0.0, 1.0),
                        exposureValidity = result.consistency.frames.map { 1.0 - it.saturationFraction }.average().coerceIn(0.0, 1.0),
                        signalToNoiseConfidence = snrConfidence,
                        saturationConfidence = result.consistency.frames.map { 1.0 - it.saturationFraction }.average().coerceIn(0.0, 1.0),
                        geometryConfidence = (0.70 * geometryConfidence + 0.30 * multiFrameFraction).coerceIn(0.0, 1.0),
                        neuralResidual = 0.0,
                        timestampNs = timestampNs
                    )
                )
                x0 += w
            }
            y0 += h
        }
    }
}
