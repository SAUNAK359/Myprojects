package com.saunak.director

import com.saunak.director.engine.reconstruction.BayerPattern
import com.saunak.director.engine.reconstruction.DeterministicIsp
import com.saunak.director.engine.reconstruction.Matrix3x3
import com.saunak.director.engine.reconstruction.NoiseModel
import com.saunak.director.engine.reconstruction.RawImageFormation
import com.saunak.director.engine.reconstruction.RawObservation
import com.saunak.director.engine.reconstruction.RawPlane
import com.saunak.director.engine.reconstruction.SierraColorCalibration
import com.saunak.director.engine.reconstruction.WhiteBalanceGains
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RawImageFormationTest {
    @Test
    fun normalizeMapsBlackToZeroAndWhiteToOne() {
        val raw = RawPlane(2, 2, floatArrayOf(64f, 128f, 512f, 1023f), BayerPattern.RGGB, 64f, 1023f)
        val n = RawImageFormation.normalize(raw)
        assertEquals(0f, n[0], 1e-6f)
        assertEquals(1f, n[3], 1e-6f)
        assertTrue(n.all { it in 0f..1f })
    }

    @Test
    fun robustFusionRejectsGrossOutlier() {
        val good = RawPlane(2, 2, FloatArray(4) { 0.20f }, BayerPattern.RGGB, 0f, 1f)
        val outlier = RawPlane(2, 2, FloatArray(4) { 0.95f }, BayerPattern.RGGB, 0f, 1f)
        val noise = NoiseModel(shotNoiseSlope = 0.002f, readNoiseVariance = 0.00005f)
        val fused = RawImageFormation.fuseAligned(
            listOf(
                RawObservation(good, 10_000_000L, 100, noise = noise),
                RawObservation(good, 10_000_000L, 100, noise = noise),
                RawObservation(outlier, 10_000_000L, 100, alignmentConfidence = 0.2f, motionConfidence = 0.2f, noise = noise)
            )
        )
        fused.values.forEach { assertTrue(it < 0.35f) }
    }

    @Test
    fun demosaicProducesFiniteRgb() {
        val raw = RawPlane(
            4,
            4,
            floatArrayOf(
                0.8f, 0.5f, 0.8f, 0.5f,
                0.5f, 0.2f, 0.5f, 0.2f,
                0.8f, 0.5f, 0.8f, 0.5f,
                0.5f, 0.2f, 0.5f, 0.2f
            ),
            BayerPattern.RGGB,
            0f,
            1f
        )
        val rgb = RawImageFormation.demosaic(raw)
        assertEquals(4 * 4 * 3, rgb.rgb.size)
        assertTrue(rgb.rgb.all { it.isFinite() && it >= 0f })
    }

    @Test
    fun ispOutputIsBounded() {
        val raw = RawPlane(2, 2, floatArrayOf(0.9f, 0.5f, 0.5f, 0.2f), BayerPattern.RGGB, 0f, 1f)
        val linear = RawImageFormation.demosaic(raw)
        val rendered = DeterministicIsp.render(
            linear,
            SierraColorCalibration(
                whiteBalance = WhiteBalanceGains(1.4f, 1f, 1.8f),
                cameraToWorking = Matrix3x3.IDENTITY,
                exposureGain = 1.2f
            )
        )
        assertTrue(rendered.rgb.all { it.isFinite() && it in 0f..1f })
    }
}
