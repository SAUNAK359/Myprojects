package com.saunak.director

import com.saunak.director.engine.subject.NormalizedPoint
import com.saunak.director.engine.subject.NormalizedRect
import com.saunak.director.engine.subject.PredictiveSubjectPolicy
import com.saunak.director.engine.subject.SubjectKinematics
import com.saunak.director.engine.subject.SubjectTrack
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PredictiveSubjectPolicyTest {
    private fun track(
        centerX: Double,
        velocityX: Double,
        accelerationX: Double = 0.0,
        confidence: Double = 0.92,
        uncertainty: Double = 0.08
    ): SubjectTrack {
        val bounds = NormalizedRect(centerX - 0.10, 0.20, centerX + 0.10, 0.78)
        val horizon = 120L
        val dt = horizon / 1000.0
        val predicted = NormalizedPoint(
            centerX + velocityX * dt + 0.5 * accelerationX * dt * dt,
            bounds.center.y
        ).clamped()
        return SubjectTrack(
            id = 1,
            cameraId = "0",
            bounds = bounds,
            center = bounds.center,
            kinematics = SubjectKinematics(
                velocityXPerSec = velocityX,
                velocityYPerSec = 0.0,
                accelerationXPerSec2 = accelerationX,
                accelerationYPerSec2 = 0.0,
                predictedCenter = predicted,
                predictionHorizonMs = horizon
            ),
            confidence = confidence,
            uncertainty = uncertainty,
            firstSeenNs = 1_000_000_000L,
            lastSeenNs = 1_200_000_000L,
            updateCount = 3,
            missedUpdates = 0,
            mask = null,
            pose = null,
            depth = null,
            source = "test"
        )
    }

    @Test fun movingSubjectGetsLeadRoomAndShorterExposureIntent() {
        val directive = PredictiveSubjectPolicy.evaluate(track(centerX = 0.50, velocityX = 0.70))
        assertTrue(directive.cropCenter.x > directive.autofocusPoint.x)
        assertTrue(directive.kinematicRisk > 0.25)
        assertTrue(directive.exposureCapMultiplier < 0.90)
        assertTrue(directive.shouldRetargetThreeA)
    }

    @Test fun stableSubjectAvoidsUnnecessaryThreeARetarget() {
        val directive = PredictiveSubjectPolicy.evaluate(track(centerX = 0.50, velocityX = 0.0))
        assertFalse(directive.shouldRetargetThreeA)
        assertTrue(directive.exposureCapMultiplier > 0.90)
    }

    @Test fun predictedFrameExitRaisesEdgeRisk() {
        val safe = PredictiveSubjectPolicy.evaluate(track(centerX = 0.50, velocityX = 0.10))
        val edge = PredictiveSubjectPolicy.evaluate(track(centerX = 0.86, velocityX = 0.70))
        assertTrue(edge.edgeRisk > safe.edgeRisk)
    }
}
