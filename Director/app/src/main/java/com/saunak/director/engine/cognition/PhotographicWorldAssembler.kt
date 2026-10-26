package com.saunak.director.engine.cognition

import com.saunak.director.engine.state.SceneState
import kotlin.math.log2
import kotlin.math.max

class PhotographicWorldAssembler(
    private val policy: CognitionPolicy,
    private val reliabilityPlane: ReliabilityPlane
) {
    fun assemble(stateId: Long, state: SceneState, nowNs: Long = System.nanoTime()): PhotographicWorldState {
        val quality = state.technicalQuality
        val thresholds = policy.illuminationThresholds
        val illumination = when {
            quality.meanLuma >= thresholds.daylight -> IlluminationRegime.DAYLIGHT
            quality.meanLuma >= thresholds.bright -> IlluminationRegime.BRIGHT
            quality.meanLuma >= thresholds.normal -> IlluminationRegime.NORMAL
            quality.meanLuma >= thresholds.dim -> IlluminationRegime.DIM
            quality.meanLuma >= thresholds.low -> IlluminationRegime.LOW
            quality.meanLuma >= thresholds.veryLow -> IlluminationRegime.VERY_LOW
            else -> IlluminationRegime.NEAR_DARK
        }
        val p05 = quality.shadowP05.coerceIn(0.0, 255.0)
        val p95 = quality.highlightP95.coerceIn(0.0, 255.0)
        val dynamicRange = log2((p95 + 1.0) / (p05 + 1.0)).coerceIn(0.0, 14.0)
        val highlightTail = ((p95 - policy.riskThresholds.highlightStart) /
            (255.0 - policy.riskThresholds.highlightStart).coerceAtLeast(1.0)).coerceIn(0.0, 1.0)
        val shadowTail = ((policy.riskThresholds.shadowFloor - p05) /
            policy.riskThresholds.shadowFloor.coerceAtLeast(1.0)).coerceIn(0.0, 1.0)
        val highlightRisk = max(quality.brightClipFraction.coerceIn(0.0, 1.0), highlightTail)
        val shadowRisk = max(quality.darkClipFraction.coerceIn(0.0, 1.0), shadowTail)
        val telemetry = state.telemetry
        val rollingRisk = telemetry.rollingShutterSkewNs?.let { skew ->
            val denominator = (telemetry.frameDurationNs.takeIf { it > 0 } ?: telemetry.exposureTimeNs).coerceAtLeast(1L)
            (skew.toDouble() / denominator.toDouble() / policy.riskThresholds.rollingShutterSevereFraction)
                .coerceIn(0.0, 1.0)
        }
        val reliability = reliabilityPlane.worldReliability(
            createdAtNs = state.createdAtNs,
            confidence = state.confidence,
            uncertainty = state.uncertainty,
            nowNs = maxOf(nowNs, state.createdAtNs)
        )
        return PhotographicWorldState(
            stateId = stateId,
            createdAtNs = state.createdAtNs,
            sceneBranch = state.sceneSignals.branch,
            sceneConfidence = state.sceneSignals.confidence,
            illumination = illumination,
            meanLuma = quality.meanLuma,
            medianLuma = quality.medianLuma,
            dynamicRangeStops = dynamicRange,
            highlightClipRisk = highlightRisk,
            shadowNoiseRisk = shadowRisk,
            primaryHumanSubject = state.sceneSignals.primaryHumanSubject,
            faceCount = state.sceneSignals.faceCount,
            faceCoverage = state.sceneSignals.faceCoverage,
            motionRisk = state.motion.conservativeRisk.coerceIn(0.0, 1.0),
            cameraMotionUncertainty = state.motion.camera.uncertainty.coerceIn(0.0, 1.0),
            subjectMotionUncertainty = state.motion.subject.uncertainty.coerceIn(0.0, 1.0),
            rollingShutterRisk = rollingRisk,
            exposureTimeNs = telemetry.exposureTimeNs,
            sensitivityIso = telemetry.sensitivityIso,
            focusDistanceDiopters = telemetry.focusDistanceDiopters,
            zoomRatio = state.zoomRatio,
            cameraId = telemetry.cameraId,
            inputReliability = reliability
        )
    }
}
