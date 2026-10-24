package com.saunak.director

import com.saunak.director.engine.reconstruction.BayerPattern
import com.saunak.director.engine.reconstruction.CfaBurstAlignment
import com.saunak.director.engine.reconstruction.DeterministicBurstPipeline
import com.saunak.director.engine.reconstruction.EvidenceLedger
import com.saunak.director.engine.reconstruction.Matrix3x3
import com.saunak.director.engine.reconstruction.NoiseModel
import com.saunak.director.engine.reconstruction.RawBurstFrame
import com.saunak.director.engine.reconstruction.RawObservation
import com.saunak.director.engine.reconstruction.RawPlane
import com.saunak.director.engine.reconstruction.SierraColorCalibration
import com.saunak.director.engine.reconstruction.WhiteBalanceGains
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CfaBurstAlignmentTest {
    private fun pattern(width: Int, height: Int): FloatArray = FloatArray(width * height) { index ->
        val x = index % width
        val y = index / width
        (((x * 13 + y * 7) % 97) / 96f).coerceIn(0f, 1f)
    }

    private fun shifted(source: FloatArray, width: Int, height: Int, dx: Int, dy: Int): FloatArray {
        val out = FloatArray(source.size)
        for (y in 0 until height) for (x in 0 until width) {
            val sx = (x - dx).coerceIn(0, width - 1)
            val sy = (y - dy).coerceIn(0, height - 1)
            out[y * width + x] = source[sy * width + sx]
        }
        return out
    }

    @Test
    fun estimatesEvenTranslationAndProducesCommonCrop() {
        val width = 64
        val height = 64
        val base = pattern(width, height)
        val ref = RawPlane(width, height, base, BayerPattern.RGGB, 0f, 1f)
        val moved = RawPlane(width, height, shifted(base, width, height, 4, -2), BayerPattern.RGGB, 0f, 1f)
        val t = CfaBurstAlignment.estimate(ref, moved, 8)
        assertEquals(0, t.dx % 2)
        assertEquals(0, t.dy % 2)
        assertTrue(t.confidence in 0f..1f)

        val noise = NoiseModel(0.002f, 0.0001f)
        val aligned = CfaBurstAlignment.align(
            listOf(
                RawObservation(ref, 10_000_000, 100, noise = noise),
                RawObservation(moved, 10_000_000, 100, noise = noise)
            ),
            8
        )
        assertEquals(2, aligned.observations.size)
        assertTrue(aligned.observations.first().raw.width <= width)
        assertTrue(aligned.observations.first().raw.height <= height)
    }

    @Test
    fun deterministicPipelineWritesPhysicalEvidence() {
        val width = 32
        val height = 32
        val raw = RawPlane(width, height, pattern(width, height), BayerPattern.RGGB, 0f, 1f)
        val noise = NoiseModel(0.001f, 0.0001f)
        val ledger = EvidenceLedger()
        val pipeline = DeterministicBurstPipeline(ledger)
        val result = pipeline.reconstruct(
            frames = listOf(
                RawBurstFrame(1, RawObservation(raw, 8_000_000, 100, noise = noise)),
                RawBurstFrame(2, RawObservation(raw, 8_000_000, 100, noise = noise))
            ),
            calibration = SierraColorCalibration(
                WhiteBalanceGains(1f, 1f, 1f),
                Matrix3x3.IDENTITY
            ),
            timestampNs = 1234
        )
        assertEquals(setOf(1L, 2L), result.evidence.sourceFrameIds)
        assertTrue(result.evidence.evidenceSupport > 0.0)
        assertEquals(1, ledger.snapshot().size)
    }
}
