package com.saunak.director

import com.saunak.director.engine.capture.MotionClass
import com.saunak.director.engine.motion.MotionEstimate
import com.saunak.director.engine.motion.MotionExposurePolicy
import com.saunak.director.engine.motion.MotionSnapshot
import com.saunak.director.engine.motion.MotionVector
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MotionModelsTest {
    @Test fun uncertaintyRaisesConservativeMotionRisk() {
        val estimate = MotionEstimate(
            risk = 0.30,
            uncertainty = 0.45,
            confidence = 0.55,
            observedAtNs = 10L,
            source = "test"
        )
        assertEquals(MotionClass.HIGH, estimate.toMotionClass())
        assertTrue(estimate.conservativeRisk >= 0.75)
    }

    @Test fun unobservedIsNotMediumFallback() {
        val estimate = MotionEstimate.unobserved(100L)
        assertEquals(MotionClass.HIGH, estimate.toMotionClass())
        assertEquals(1.0, estimate.uncertainty, 0.0)
        assertEquals(0.0, estimate.confidence, 0.0)
    }

    @Test fun highPixelVelocityShortensExposureCap() {
        val slow = MotionSnapshot(
            camera = MotionEstimate(0.08, 0.02, 0.95, MotionVector(4.0, 0.0), 0.0, 1L, "test"),
            subject = MotionEstimate(0.08, 0.02, 0.95, MotionVector(6.0, 0.0), 0.0, 1L, "test"),
            frameTimestampNs = 1L,
            fusionConfidence = 0.95,
            timestampComparableToImu = true
        )
        val fast = MotionSnapshot(
            camera = slow.camera,
            subject = MotionEstimate(0.78, 0.06, 0.92, MotionVector(800.0, 0.0), 8.0, 2L, "test"),
            frameTimestampNs = 2L,
            fusionConfidence = 0.92,
            timestampComparableToImu = true
        )
        assertTrue(MotionExposurePolicy.exposureCapNs(fast) < MotionExposurePolicy.exposureCapNs(slow))
    }
}
