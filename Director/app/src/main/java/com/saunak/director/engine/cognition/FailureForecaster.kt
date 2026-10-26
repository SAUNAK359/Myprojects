package com.saunak.director.engine.cognition

import kotlin.math.max

class FailureForecaster(private val policy: CognitionPolicy) {
    fun forecast(world: PhotographicWorldState, reliability: ReliabilityEnvelope): FailureForecast {
        val evidenceConfidence = reliability.effectiveConfidence
        val humanMotion = if (world.primaryHumanSubject) world.motionRisk else world.motionRisk * 0.72
        val motionBlur = RiskEstimate(
            probability = world.motionRisk,
            confidence = evidenceConfidence,
            reason = "temporal motion evidence"
        )
        val ghost = RiskEstimate(
            probability = if (world.primaryHumanSubject) max(humanMotion, world.subjectMotionUncertainty * 0.5) else humanMotion,
            confidence = evidenceConfidence * if (world.primaryHumanSubject) world.sceneConfidence else 0.72,
            reason = if (world.primaryHumanSubject) "tracked subject motion and uncertainty" else "scene motion evidence"
        )
        val sensorNoise = RiskEstimate(
            probability = world.shadowNoiseRisk,
            confidence = evidenceConfidence,
            reason = "shadow occupancy and exposure evidence"
        )
        val highlights = RiskEstimate(
            probability = world.highlightClipRisk,
            confidence = evidenceConfidence,
            reason = "highlight tail and clipping evidence"
        )
        val rolling = world.rollingShutterRisk?.let {
            RiskEstimate(
                probability = (it * (0.35 + 0.65 * world.motionRisk)).coerceIn(0.0, 1.0),
                confidence = evidenceConfidence,
                reason = "sensor rolling-shutter skew relative to frame duration"
            )
        } ?: RiskEstimate.unknown("camera did not expose rolling-shutter skew")
        val focus = if (world.focusDistanceDiopters != null) {
            RiskEstimate(
                probability = (world.motionRisk * 0.35).coerceIn(0.0, 1.0),
                confidence = evidenceConfidence * 0.55,
                reason = "focus telemetry present; no dedicated focus-quality model yet"
            )
        } else RiskEstimate.unknown("focus-distance telemetry unavailable")

        return FailureForecast(
            motionBlur = motionBlur,
            subjectGhosting = ghost,
            sensorNoise = sensorNoise,
            highlightClipping = highlights,
            rollingShutter = rolling,
            focusFailure = focus,
            flickerBanding = RiskEstimate.unknown("flicker detector not provisioned"),
            reflection = RiskEstimate.unknown("reflection classifier not provisioned"),
            flare = RiskEstimate.unknown("flare classifier not provisioned")
        )
    }
}
