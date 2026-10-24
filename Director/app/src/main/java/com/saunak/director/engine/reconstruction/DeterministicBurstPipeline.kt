package com.saunak.director.engine.reconstruction

import kotlin.math.max

data class RawBurstFrame(
    val frameId: Long,
    val observation: RawObservation
) {
    init { require(frameId > 0) }
}

data class DeterministicBurstResult(
    val display: DisplayRgbImage,
    val linear: LinearRgbImage,
    val fusedRaw: RawPlane,
    val evidence: EvidenceRecord,
    val alignment: CfaAlignedBurst
)

/** End-to-end deterministic evidence-preserving reconstruction baseline for a RAW burst. */
class DeterministicBurstPipeline(
    private val ledger: EvidenceLedger
) {
    fun reconstruct(
        frames: List<RawBurstFrame>,
        calibration: SierraColorCalibration,
        timestampNs: Long
    ): DeterministicBurstResult {
        require(frames.isNotEmpty())
        require(timestampNs > 0)

        val aligned = CfaBurstAlignment.align(frames.map { it.observation })
        val fused = RawImageFormation.fuseAligned(aligned.observations)
        val linear = RawImageFormation.demosaic(fused)
        val display = DeterministicIsp.render(linear, calibration)

        val frameIds = frames.map { it.frameId }.toSet()
        val rawWeights = frames.indices.associate { index ->
            val obs = aligned.observations[index]
            val t = aligned.translations[index]
            frames[index].frameId to max(
                1e-6,
                obs.alignmentConfidence.toDouble() *
                    obs.motionConfidence.toDouble() *
                    obs.saturationConfidence.toDouble() /
                    max(obs.noise.readNoiseVariance.toDouble(), 1e-6)
            ) * max(t.confidence.toDouble(), 1e-3)
        }

        val alignmentConfidence = aligned.translations.map { it.confidence.toDouble() }.average().coerceIn(0.0, 1.0)
        val motionConfidence = aligned.observations.map { it.motionConfidence.toDouble() }.average().coerceIn(0.0, 1.0)
        val saturationConfidence = aligned.observations.map { it.saturationConfidence.toDouble() }.average().coerceIn(0.0, 1.0)
        val snrConfidence = aligned.observations.map {
            (1.0 / (1.0 + 100.0 * it.noise.readNoiseVariance.toDouble())).coerceIn(0.0, 1.0)
        }.average().coerceIn(0.0, 1.0)

        val evidence = EvidenceRecord(
            regionId = RegionId("burst/global/$timestampNs"),
            bounds = NormalizedRect(0.0, 0.0, 1.0, 1.0),
            semantic = RegionSemantic.GENERAL,
            sourceFrameIds = frameIds,
            fusionWeights = rawWeights,
            alignmentConfidence = alignmentConfidence,
            motionConfidence = motionConfidence,
            exposureValidity = 1.0,
            signalToNoiseConfidence = snrConfidence,
            saturationConfidence = saturationConfidence,
            geometryConfidence = alignmentConfidence,
            neuralResidual = 0.0,
            timestampNs = timestampNs
        )
        ledger.put(evidence)

        return DeterministicBurstResult(
            display = display,
            linear = linear,
            fusedRaw = fused,
            evidence = evidence,
            alignment = aligned
        )
    }
}
