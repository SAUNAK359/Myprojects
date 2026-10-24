package com.saunak.director

import com.saunak.director.engine.reconstruction.EvidenceRecord
import com.saunak.director.engine.reconstruction.NormalizedRect
import com.saunak.director.engine.reconstruction.ReconstructionArbiter
import com.saunak.director.engine.reconstruction.ReconstructionCandidate
import com.saunak.director.engine.reconstruction.ReconstructionPath
import com.saunak.director.engine.reconstruction.RegionId
import com.saunak.director.engine.reconstruction.RegionSemantic
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ReconstructionArbiterTest {
    private fun evidence(semantic: RegionSemantic, neuralResidual: Double, support: Double): EvidenceRecord = EvidenceRecord(
        regionId = RegionId("r1"),
        bounds = NormalizedRect(0.1, 0.1, 0.5, 0.5),
        semantic = semantic,
        sourceFrameIds = setOf(1, 2, 3),
        fusionWeights = mapOf(1L to 0.4, 2L to 0.3, 3L to 0.3),
        alignmentConfidence = support,
        motionConfidence = support,
        exposureValidity = support,
        signalToNoiseConfidence = support,
        saturationConfidence = support,
        geometryConfidence = support,
        neuralResidual = neuralResidual,
        timestampNs = 1_000L
    )

    @Test fun unsupportedFaceDetailFallsBackToDeterministic() {
        val e = evidence(RegionSemantic.HUMAN_FACE, neuralResidual = 1.5, support = 0.58)
        val deterministic = ReconstructionCandidate(e.regionId, ReconstructionPath.DETERMINISTIC, 0.74, 0.72, 0.96, 0.05, 0.76)
        val neural = ReconstructionCandidate(e.regionId, ReconstructionPath.NEURAL, 0.96, 0.95, 0.80, 0.08, 0.95)
        val result = ReconstructionArbiter().choose(e, listOf(neural, deterministic))
        assertEquals(ReconstructionPath.DETERMINISTIC, result.chosen.path)
        assertTrue(e.hallucinationRisk > 0.35)
    }

    @Test fun wellSupportedNonProtectedTextureCanUseNeuralPath() {
        val e = evidence(RegionSemantic.FOLIAGE, neuralResidual = 0.04, support = 0.96)
        val deterministic = ReconstructionCandidate(e.regionId, ReconstructionPath.DETERMINISTIC, 0.72, 0.75, 0.96, 0.03, 0.72)
        val neural = ReconstructionCandidate(e.regionId, ReconstructionPath.NEURAL, 0.98, 0.94, 0.94, 0.03, 0.96)
        val result = ReconstructionArbiter().choose(e, listOf(neural, deterministic))
        assertEquals(ReconstructionPath.NEURAL, result.chosen.path)
    }
}
