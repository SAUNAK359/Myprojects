package com.saunak.director.engine.motion

import android.content.Context
import android.os.SystemClock
import com.saunak.director.engine.belief.Belief
import com.saunak.director.engine.belief.SceneBeliefGraph
import com.saunak.director.engine.subject.SubjectTrack
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.max

/**
 * Always-on temporal motion estimator. It never substitutes a fixed motion category for missing
 * evidence: missing or conflicting evidence increases uncertainty, which is carried into planning.
 */
class MotionFusionEngine(
    context: Context,
    private val beliefs: SceneBeliefGraph
) : AutoCloseable {
    private val sensors = MotionSensorHub(context).also { it.start() }
    private val visual = VisualMotionTracker()
    private val latestRef = AtomicReference<MotionSnapshot?>(null)

    fun onFrame(
        frame: LumaMotionFrame,
        timestampComparableToImu: Boolean,
        exposureTimeNs: Long?
    ): MotionSnapshot {
        val trackedSubject = beliefs.get<SubjectTrack>(SceneBeliefGraph.PRIMARY_SUBJECT)?.value
        val visualObservation = visual.push(frame, trackedSubject?.bounds)
        val imu = sensors.observation(
            frameSensorTimestampNs = frame.sensorTimestampNs,
            frameCallbackElapsedRealtimeNs = frame.callbackElapsedRealtimeNs,
            timestampComparable = timestampComparableToImu
        )
        val previous = latestRef.get()
        val snapshot = fuse(frame, visualObservation, imu, exposureTimeNs, previous)
        latestRef.set(snapshot)
        publish(snapshot)
        return snapshot
    }

    /** `nowElapsedNs` is deliberately in Android elapsed-realtime, matching visual/IMU fusion time. */
    fun latest(nowElapsedNs: Long = SystemClock.elapsedRealtimeNanos()): MotionSnapshot {
        val value = latestRef.get() ?: return MotionSnapshot.unobserved(nowElapsedNs)
        val ageNs = (nowElapsedNs - value.frameTimestampNs).coerceAtLeast(0L)
        if (ageNs <= STALE_AFTER_NS) return value
        val agePenalty = ((ageNs - STALE_AFTER_NS).toDouble() / STALE_AFTER_NS).coerceIn(0.0, 1.0)
        return value.copy(
            camera = value.camera.copy(
                uncertainty = (value.camera.uncertainty + agePenalty * 0.65).coerceIn(0.0, 1.0),
                confidence = (value.camera.confidence * (1.0 - agePenalty)).coerceIn(0.0, 1.0)
            ),
            subject = value.subject.copy(
                uncertainty = (value.subject.uncertainty + agePenalty * 0.75).coerceIn(0.0, 1.0),
                confidence = (value.subject.confidence * (1.0 - agePenalty)).coerceIn(0.0, 1.0)
            ),
            fusionConfidence = (value.fusionConfidence * (1.0 - agePenalty)).coerceIn(0.0, 1.0)
        )
    }

    private fun fuse(
        frame: LumaMotionFrame,
        vision: VisualMotionObservation?,
        imu: ImuMotionObservation,
        exposureTimeNs: Long?,
        previous: MotionSnapshot?
    ): MotionSnapshot {
        val imuRisk = imuRisk(imu)
        val visualRisk = vision?.globalRisk
        val visualConfidence = vision?.confidence ?: 0.0
        val imuConfidence = imu.confidence
        val cameraDenominator = visualConfidence + imuConfidence
        val fusedCameraRisk = if (cameraDenominator > 1e-6) {
            ((visualRisk ?: 0.0) * visualConfidence + imuRisk * imuConfidence) / cameraDenominator
        } else 0.0
        val disagreement = if (visualRisk != null && visualConfidence > 0.15 && imuConfidence > 0.15) {
            abs(visualRisk - imuRisk)
        } else 0.0
        val cameraEvidence = (1.0 - (1.0 - visualConfidence) * (1.0 - imuConfidence)).coerceIn(0.0, 1.0)
        val cameraUncertainty = (1.0 - cameraEvidence + disagreement * 0.45).coerceIn(0.0, 1.0)

        val subjectRisk = vision?.subjectRisk ?: 0.0
        val subjectEvidence = vision?.subjectConfidence ?: 0.0
        val subjectUncertainty = (1.0 - subjectEvidence).coerceIn(0.0, 1.0)
        val exposureSeconds = (exposureTimeNs ?: DEFAULT_EXPOSURE_NS).coerceAtLeast(1L) / 1_000_000_000.0

        val cameraRaw = MotionEstimate(
            risk = fusedCameraRisk.coerceIn(0.0, 1.0),
            uncertainty = cameraUncertainty,
            confidence = cameraEvidence,
            vector = vision?.globalVector ?: MotionVector(),
            predictedBlurPx = (vision?.globalVector?.speedPxPerSecond ?: 0.0) * exposureSeconds,
            observedAtNs = frame.sensorTimestampNs,
            source = if (vision != null) "imu+visual_global" else "imu_only"
        )
        val subjectRaw = MotionEstimate(
            risk = subjectRisk.coerceIn(0.0, 1.0),
            uncertainty = subjectUncertainty,
            confidence = subjectEvidence,
            vector = vision?.residualSubjectVector ?: MotionVector(),
            predictedBlurPx = (vision?.residualSubjectVector?.speedPxPerSecond ?: 0.0) * exposureSeconds,
            observedAtNs = frame.sensorTimestampNs,
            source = when {
                vision == null -> "subject_unobserved"
                subjectEvidence >= 0.55 -> "tracked_subject_residual"
                else -> "global_residual_fallback"
            }
        )

        val camera = smooth(previous?.camera, cameraRaw)
        val subject = smooth(previous?.subject, subjectRaw)
        val fusionConfidence = (0.54 * camera.confidence + 0.46 * subject.confidence).coerceIn(0.0, 1.0)
        return MotionSnapshot(
            camera = camera,
            subject = subject,
            frameTimestampNs = frame.sensorTimestampNs,
            fusionConfidence = fusionConfidence,
            timestampComparableToImu = imu.timestampAligned
        )
    }

    /** Fast escalation, slow relaxation prevents oscillation without categorical hysteresis. */
    private fun smooth(previous: MotionEstimate?, current: MotionEstimate): MotionEstimate {
        if (previous == null) return current
        val alpha = if (current.conservativeRisk >= previous.conservativeRisk) RISE_ALPHA else FALL_ALPHA
        fun blend(a: Double, b: Double) = a * (1.0 - alpha) + b * alpha
        return current.copy(
            risk = blend(previous.risk, current.risk).coerceIn(0.0, 1.0),
            uncertainty = blend(previous.uncertainty, current.uncertainty).coerceIn(0.0, 1.0),
            confidence = blend(previous.confidence, current.confidence).coerceIn(0.0, 1.0),
            vector = MotionVector(
                blend(previous.vector.xPxPerSecond, current.vector.xPxPerSecond),
                blend(previous.vector.yPxPerSecond, current.vector.yPxPerSecond)
            ),
            predictedBlurPx = max(0.0, blend(previous.predictedBlurPx, current.predictedBlurPx))
        )
    }

    private fun imuRisk(imu: ImuMotionObservation): Double {
        val angular = (imu.angularVelocityRadPerSec / 2.2).coerceAtLeast(0.0)
        val acceleration = (imu.linearAccelerationStdMps2 / 4.0).coerceAtLeast(0.0)
        val combined = 0.78 * angular + 0.22 * acceleration
        return (combined / (1.0 + combined)).coerceIn(0.0, 1.0)
    }

    private fun publish(snapshot: MotionSnapshot) {
        // SceneBeliefGraph is aged using System.nanoTime(); sensor/IMU fusion keeps its own clock.
        val graphNowNs = System.nanoTime()
        beliefs.put(
            SceneBeliefGraph.CAMERA_MOTION,
            Belief(
                value = snapshot.camera,
                confidence = snapshot.camera.confidence,
                uncertainty = snapshot.camera.uncertainty,
                timestampNs = graphNowNs,
                source = snapshot.camera.source,
                halfLifeNs = CAMERA_HALF_LIFE_NS
            )
        )
        beliefs.put(
            SceneBeliefGraph.SUBJECT_MOTION,
            Belief(
                value = snapshot.subject,
                confidence = snapshot.subject.confidence,
                uncertainty = snapshot.subject.uncertainty,
                timestampNs = graphNowNs,
                source = snapshot.subject.source,
                halfLifeNs = SUBJECT_HALF_LIFE_NS
            )
        )
    }

    override fun close() {
        sensors.close()
        visual.reset()
        latestRef.set(null)
    }

    companion object {
        private const val DEFAULT_EXPOSURE_NS = 16_666_667L
        private const val CAMERA_HALF_LIFE_NS = 110_000_000L
        private const val SUBJECT_HALF_LIFE_NS = 150_000_000L
        private const val STALE_AFTER_NS = 220_000_000L
        private const val RISE_ALPHA = 0.72
        private const val FALL_ALPHA = 0.28
    }
}
