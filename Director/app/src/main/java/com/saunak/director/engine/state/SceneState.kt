package com.saunak.director.engine.state

import com.saunak.director.camera.CameraSensorTelemetry
import com.saunak.director.director.SierraSceneSignals
import com.saunak.director.engine.motion.MotionSnapshot
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
    val confidence: Double,
    val uncertainty: Double
) {
    init {
        require(createdAtNs > 0)
        require(zoomRatio >= 1f)
        require(confidence in 0.0..1.0)
        require(uncertainty in 0.0..1.0)
    }
}

object SceneStateAssembler {
    fun assemble(
        scene: SierraSceneSignals,
        quality: TechnicalQuality,
        motion: MotionSnapshot,
        telemetry: CameraSensorTelemetry,
        zoomRatio: Float,
        nowNs: Long = System.nanoTime()
    ): SceneState {
        val motionConfidence = motion.fusionConfidence
        val confidence = (
            0.48 * scene.confidence +
                0.22 * motionConfidence +
                0.30 * telemetryConfidence(telemetry, nowNs)
            ).coerceIn(0.0, 1.0)
        val uncertainty = (
            0.56 * maxOf(motion.camera.uncertainty, motion.subject.uncertainty) +
                0.44 * (1.0 - scene.confidence)
            ).coerceIn(0.0, 1.0)
        return SceneState(
            createdAtNs = nowNs,
            sceneSignals = scene,
            technicalQuality = quality,
            motion = motion,
            telemetry = telemetry,
            zoomRatio = zoomRatio,
            confidence = confidence,
            uncertainty = uncertainty
        )
    }

    private fun telemetryConfidence(telemetry: CameraSensorTelemetry, nowNs: Long): Double {
        val ageNs = (nowNs - telemetry.sensorTimestampNs).coerceAtLeast(0L)
        return when {
            ageNs <= 80_000_000L -> 1.0
            ageNs <= 180_000_000L -> 0.82
            ageNs <= 350_000_000L -> 0.55
            else -> 0.25
        }
    }
}
