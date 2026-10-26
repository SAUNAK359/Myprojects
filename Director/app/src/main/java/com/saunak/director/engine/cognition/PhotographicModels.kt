package com.saunak.director.engine.cognition

import com.saunak.director.processing.Look

enum class ReliabilityFlag {
    STALE,
    LOW_CONFIDENCE,
    HIGH_UNCERTAINTY,
    OUT_OF_DISTRIBUTION,
    SEMANTIC_CONFLICT,
    UNSUPPORTED,
    UNKNOWN_INPUT
}

data class ObservationProvenance(
    val frameIds: List<Long> = emptyList(),
    val cameraId: String? = null,
    val source: String,
    val notes: List<String> = emptyList()
) {
    init { require(source.isNotBlank()) }
}

data class ModelObservation<T>(
    val modelId: String,
    val modelVersion: String,
    val stateId: Long,
    val timestampNs: Long,
    val task: String,
    val value: T?,
    val confidence: Double,
    val aleatoricUncertainty: Double,
    val epistemicUncertainty: Double,
    val oodScore: Double,
    val expiresAtNs: Long,
    val provenance: ObservationProvenance,
    val flags: Set<ReliabilityFlag> = emptySet()
) {
    init {
        require(modelId.isNotBlank() && modelVersion.isNotBlank() && task.isNotBlank())
        require(stateId >= 0 && timestampNs > 0 && expiresAtNs >= timestampNs)
        require(confidence in 0.0..1.0)
        require(aleatoricUncertainty in 0.0..1.0)
        require(epistemicUncertainty in 0.0..1.0)
        require(oodScore in 0.0..1.0)
    }

    fun isFresh(nowNs: Long): Boolean = nowNs in timestampNs..expiresAtNs
}

data class ReliabilityEnvelope(
    val confidence: Double,
    val uncertainty: Double,
    val freshness: Double,
    val oodScore: Double = 0.0,
    val flags: Set<ReliabilityFlag> = emptySet()
) {
    init {
        require(confidence in 0.0..1.0)
        require(uncertainty in 0.0..1.0)
        require(freshness in 0.0..1.0)
        require(oodScore in 0.0..1.0)
    }

    val effectiveConfidence: Double
        get() = (confidence * (1.0 - uncertainty) * freshness * (1.0 - oodScore)).coerceIn(0.0, 1.0)
}

enum class IlluminationRegime { DAYLIGHT, BRIGHT, NORMAL, DIM, LOW, VERY_LOW, NEAR_DARK }

data class PhotographicWorldState(
    val stateId: Long,
    val createdAtNs: Long,
    val sceneBranch: String,
    val sceneConfidence: Double,
    val illumination: IlluminationRegime,
    val meanLuma: Double,
    val medianLuma: Double,
    val dynamicRangeStops: Double,
    val highlightClipRisk: Double,
    val shadowNoiseRisk: Double,
    val primaryHumanSubject: Boolean,
    val faceCount: Int,
    val faceCoverage: Double,
    val motionRisk: Double,
    val cameraMotionUncertainty: Double,
    val subjectMotionUncertainty: Double,
    val rollingShutterRisk: Double?,
    val exposureTimeNs: Long,
    val sensitivityIso: Int,
    val focusDistanceDiopters: Float?,
    val zoomRatio: Float,
    val cameraId: String?,
    val inputReliability: ReliabilityEnvelope
) {
    init {
        require(stateId > 0 && createdAtNs > 0)
        require(sceneBranch.isNotBlank())
        require(sceneConfidence in 0.0..1.0)
        require(dynamicRangeStops >= 0.0)
        require(highlightClipRisk in 0.0..1.0)
        require(shadowNoiseRisk in 0.0..1.0)
        require(faceCount >= 0 && faceCoverage in 0.0..1.0)
        require(motionRisk in 0.0..1.0)
        require(cameraMotionUncertainty in 0.0..1.0)
        require(subjectMotionUncertainty in 0.0..1.0)
        require(rollingShutterRisk == null || rollingShutterRisk in 0.0..1.0)
        require(exposureTimeNs > 0 && sensitivityIso > 0 && zoomRatio >= 1f)
    }
}

data class RiskEstimate(
    val probability: Double?,
    val confidence: Double,
    val reason: String
) {
    init {
        require(probability == null || probability in 0.0..1.0)
        require(confidence in 0.0..1.0)
        require(reason.isNotBlank())
    }

    val known: Boolean get() = probability != null && confidence > 0.0

    companion object {
        fun unknown(reason: String): RiskEstimate = RiskEstimate(null, 0.0, reason)
    }
}

data class FailureForecast(
    val motionBlur: RiskEstimate,
    val subjectGhosting: RiskEstimate,
    val sensorNoise: RiskEstimate,
    val highlightClipping: RiskEstimate,
    val rollingShutter: RiskEstimate,
    val focusFailure: RiskEstimate,
    val flickerBanding: RiskEstimate,
    val reflection: RiskEstimate,
    val flare: RiskEstimate
)

enum class CaptureFamily { SAFE_SINGLE, FAST_BURST, LOW_LIGHT_BURST, HDR_BURST, FLASH_FUSION_CANDIDATE, SCREEN_FLASH_CANDIDATE }
enum class IlluminationAction { AUTO, AVOID_FLASH, CONSIDER_FILL, CONSIDER_FLASH_FUSION, CONSIDER_SCREEN_FLASH }

data class PhotographicPriorities(
    val motionFreeze: Double,
    val highlightProtection: Double,
    val shadowRecovery: Double,
    val skinNaturalness: Double,
    val atmospherePreservation: Double,
    val detailPreservation: Double
) {
    init {
        listOf(motionFreeze, highlightProtection, shadowRecovery, skinNaturalness, atmospherePreservation, detailPreservation)
            .forEach { require(it in 0.0..1.0) }
    }
}

data class ThemeSelection(
    val themeId: String,
    val displayName: String,
    val fallbackLook: Look,
    val strength: Float,
    val score: Double,
    val confidence: Double,
    val reasonCodes: List<String>,
    val abstained: Boolean = false
) {
    init {
        require(themeId.isNotBlank() && displayName.isNotBlank())
        require(strength in 0f..1f)
        require(score in 0.0..1.0 && confidence in 0.0..1.0)
    }
}

data class LiveRenderIntent(
    val theme: ThemeSelection,
    val highlightProtection: Double,
    val shadowRecovery: Double,
    val subjectRelightEv: Double,
    val backgroundSuppressionEv: Double,
    val atmospherePreservation: Double,
    val skinNaturalness: Double,
    val aggressiveness: Double
) {
    init {
        require(highlightProtection in 0.0..1.0)
        require(shadowRecovery in 0.0..1.0)
        require(subjectRelightEv in -0.4..0.8)
        require(backgroundSuppressionEv in -0.6..0.1)
        require(atmospherePreservation in 0.0..1.0)
        require(skinNaturalness in 0.0..1.0)
        require(aggressiveness in 0.0..1.0)
    }
}

data class CaptureIntent(
    val family: CaptureFamily,
    val illuminationAction: IlluminationAction,
    val motionFreezePriority: Double,
    val highlightProtectionPriority: Double,
    val shadowRecoveryPriority: Double
) {
    init {
        require(motionFreezePriority in 0.0..1.0)
        require(highlightProtectionPriority in 0.0..1.0)
        require(shadowRecoveryPriority in 0.0..1.0)
    }
}

data class PhotographicIntent(
    val stateId: Long,
    val objective: String,
    val priorities: PhotographicPriorities,
    val capture: CaptureIntent,
    val render: LiveRenderIntent,
    val expiresAtNs: Long
) {
    init { require(stateId > 0 && objective.isNotBlank() && expiresAtNs > 0) }
}

data class CognitionDecision(
    val world: PhotographicWorldState,
    val reliability: ReliabilityEnvelope,
    val failures: FailureForecast,
    val intent: PhotographicIntent
)
