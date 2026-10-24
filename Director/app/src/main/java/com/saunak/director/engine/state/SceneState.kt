package com.saunak.director.engine.state

import com.saunak.director.camera.CameraSensorTelemetry
import com.saunak.director.director.SierraSceneSignals
import com.saunak.director.engine.motion.MotionSnapshot
import com.saunak.director.engine.subject.PredictiveSubjectDirective
import com.saunak.director.engine.subject.PredictiveSubjectPolicy
import com.saunak.director.engine.subject.SubjectTrack
import com.saunak.director.processing.TechnicalQuality

/**
 * Immutable decision boundary between perception and action. Every downstream capture decision sees
 * one coherent snapshot instead of independently querying asynchronous experts at different times.
 */
data class SceneState(
    val createdAtNs: Long,
    val sceneSignals: SierraSceneSignals,
    val technicalQuality: TechnicalQuality,
    val motion: MotionSnapshot,
    val telemetry: CameraSensorTelemetry,
    val zoomRatio: Float,
    val subject: SubjectTrack? = null,
    val predictiveSubject: PredictiveSubjectDirective? = null,
    val confidence: Double,
    val uncertainty: Double
) {
    init {
        require(createdAtNs > 0)
        require(zoomRatio >= 1f)
        require(confidence in 0.0..1.0)
        require(uncertainty in 0.0..1.0)
        require(predictiveSubject == null || subject != null)
    }
}

object SceneStateAssembler {
    fun assemble(
        scene: SierraSceneSignals,
        quality: TechnicalQuality,
        motion: MotionSnapshot,
        telemetry: CameraSensorTelemetry,
        zoomRatio: Float,
        subject: SubjectTrack? = null,
        nowNs: Long = System.nanoTime()
    ): SceneState {
        val motionConfidence = motion.fusionConfidence
        val telemetryConfidence = if (motion.timestampComparableToImu) 1.0 else 0.88
        val predictive = subject?.let(PredictiveSubjectPolicy::evaluate)
        val subjectConfidence = predictive?.confidence ?: if (scene.primaryHumanSubject) 0.35 else 0.70
        val subjectUncertainty = if (scene.primaryHumanSubject) {
            subject?.uncertainty ?: 0.85
        } else 0.0
        val confidence = (
            0.39 * scene.confidence +
                0.20 * motionConfidence +
                0.24 * telemetryConfidence +
                0.17 * subjectConfidence
            ).coerceIn(0.0, 1.0)
        val uncertainty = (
            0.42 * maxOf(motion.camera.uncertainty, motion.subject.uncertainty) +
                0.32 * (1.0 - scene.confidence) +
                0.26 * subjectUncertainty
            ).coerceIn(0.0, 1.0)
        return SceneState(
            createdAtNs = nowNs,
            sceneSignals = scene,
            technicalQuality = quality,
            motion = motion,
            telemetry = telemetry,
            zoomRatio = zoomRatio,
            subject = subject,
            predictiveSubject = predictive,
            confidence = confidence,
            uncertainty = uncertainty
        )
    }
}
