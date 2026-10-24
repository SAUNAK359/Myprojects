package com.saunak.director

import com.saunak.director.camera.CameraSensorTelemetry
import com.saunak.director.camera.ManualRawFrameSpec
import com.saunak.director.camera.matchesRequest
import com.saunak.director.engine.reconstruction.RawBurstReconstructor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ComputationalRawContractsTest {
    @Test
    fun `exact capture metadata accepts small HAL rounding but rejects wrong exposure`() {
        val requested = ManualRawFrameSpec(
            exposureTimeNs = 10_000_000L,
            sensitivityIso = 400,
            role = "REFERENCE",
            evOffset = 0.0
        )
        val close = telemetry(exposure = 10_350_000L, iso = 420)
        val wrongExposure = telemetry(exposure = 12_000_000L, iso = 400)
        val wrongIso = telemetry(exposure = 10_000_000L, iso = 520)

        assertTrue(close.matchesRequest(requested))
        assertFalse(wrongExposure.matchesRequest(requested))
        assertFalse(wrongIso.matchesRequest(requested))
    }

    @Test
    fun `adaptive Bayer output preserves full resolution when within memory budget`() {
        val reconstructor = RawBurstReconstructor(maxOutputPixels = 30_000_000L)
        assertEquals(1, reconstructor.chooseBinFactor(4032, 3024))
    }

    @Test
    fun `adaptive Bayer output bins large mobile sensors to bounded resolution`() {
        val reconstructor = RawBurstReconstructor(maxOutputPixels = 30_000_000L)
        assertEquals(2, reconstructor.chooseBinFactor(8160, 6120)) // ~50 MP -> ~12.5 MP
        assertEquals(2, reconstructor.chooseBinFactor(12000, 9000)) // 108 MP -> 27 MP
        assertEquals(4, reconstructor.chooseBinFactor(16320, 12240)) // ~200 MP -> ~12.5 MP
    }

    private fun telemetry(exposure: Long, iso: Int) = CameraSensorTelemetry(
        cameraId = "0",
        sensorTimestampNs = 1_000_000L,
        exposureTimeNs = exposure,
        sensitivityIso = iso,
        frameDurationNs = maxOf(exposure, 16_666_667L),
        rollingShutterSkewNs = 8_000_000L,
        focalLengthMm = 6.8f,
        aperture = 1.8f,
        focusDistanceDiopters = 0.0f,
        afState = null,
        aeState = null,
        awbState = null
    )
}
