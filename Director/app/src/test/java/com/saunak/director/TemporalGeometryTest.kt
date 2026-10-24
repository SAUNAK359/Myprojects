package com.saunak.director

import com.saunak.director.engine.geometry.CameraMotionClass
import com.saunak.director.engine.geometry.FrameReadoutModel
import com.saunak.director.engine.geometry.GyroSample
import com.saunak.director.engine.geometry.ImuSyncModel
import com.saunak.director.engine.geometry.TemporalGeometry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TemporalGeometryTest {
    @Test
    fun rowTimeIsMonotonicAcrossRollingShutter() {
        val model = FrameReadoutModel(
            exposureStartNs = 1_000_000_000L,
            exposureDurationNs = 10_000_000L,
            rollingShutterSkewNs = 20_000_000L,
            imageHeight = 100
        )
        assertTrue(model.rowTimeNs(0) < model.rowTimeNs(99))
    }

    @Test
    fun constantGyroIntegratesExpectedRotation() {
        val base = 1_000_000_000L
        val samples = (0..10).map { i ->
            GyroSample(base + i * 10_000_000L, 0f, 0f, 1f)
        }
        val sync = ImuSyncModel(0L, 100_000L, 1f)
        val (rotation, confidence) = TemporalGeometry.integrateGyro(samples, base, base + 100_000_000L, sync)
        assertEquals(0.1, rotation.z, 0.01)
        assertTrue(confidence > 0.9f)
        assertEquals(CameraMotionClass.HIGH, TemporalGeometry.motionClass(rotation))
    }
}
