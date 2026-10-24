package com.saunak.director

import com.saunak.director.engine.subject.NormalizedRect
import com.saunak.director.engine.subject.SubjectObservation
import com.saunak.director.engine.subject.SubjectTracker
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SubjectTrackerTest {
    @Test fun preservesIdentityAndPredictsForwardMotion() {
        val tracker = SubjectTracker(predictionHorizonMs = 120L)
        val first = tracker.update(
            SubjectObservation(
                observedAtNs = 1_000_000_000L,
                cameraId = "0",
                bounds = NormalizedRect(0.30, 0.20, 0.50, 0.70),
                confidence = 0.92,
                faceConfidence = 0.92,
                source = "test"
            )
        )
        val second = tracker.update(
            SubjectObservation(
                observedAtNs = 1_100_000_000L,
                cameraId = "0",
                bounds = NormalizedRect(0.34, 0.20, 0.54, 0.70),
                confidence = 0.94,
                faceConfidence = 0.94,
                source = "test"
            )
        )
        assertEquals(first.id, second.id)
        assertTrue(second.kinematics.velocityXPerSec > 0.0)
        assertTrue(second.predictedCenter.x > second.center.x)
        assertTrue(second.confidence > 0.8)
    }

    @Test fun cameraChangeStartsNewIdentity() {
        val tracker = SubjectTracker()
        val a = tracker.update(SubjectObservation(1_000_000_000L, "0", NormalizedRect(0.2, 0.2, 0.5, 0.8), 0.9, source = "test"))
        val b = tracker.update(SubjectObservation(1_050_000_000L, "1", NormalizedRect(0.2, 0.2, 0.5, 0.8), 0.9, source = "test"))
        assertNotEquals(a.id, b.id)
    }

    @Test fun missingObservationAgesTrackInsteadOfInventingMotion() {
        val tracker = SubjectTracker(maxMisses = 2)
        val initial = tracker.update(SubjectObservation(1_000_000_000L, "0", NormalizedRect(0.2, 0.2, 0.5, 0.8), 0.9, source = "test"))
        val aged = tracker.miss(1_100_000_000L)!!
        assertEquals(initial.id, aged.id)
        assertTrue(aged.confidence < initial.confidence)
        assertTrue(aged.uncertainty > initial.uncertainty)
    }
}
