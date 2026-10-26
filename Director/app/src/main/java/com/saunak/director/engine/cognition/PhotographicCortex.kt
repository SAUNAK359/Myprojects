package com.saunak.director.engine.cognition

class PhotographicCortex(
    private val manifest: CognitionManifest,
    private val reliabilityPlane: ReliabilityPlane,
    private val themeCortex: ThemeCortex
) {
    fun compose(
        world: PhotographicWorldState,
        failures: FailureForecast,
        reliability: ReliabilityEnvelope,
        theme: ThemeSelection,
        nowNs: Long
    ): PhotographicIntent {
        val profile = themeCortex.profile(theme.themeId)
        val motion = failures.motionBlur.probability ?: world.motionRisk
        val highlight = failures.highlightClipping.probability ?: world.highlightClipRisk
        val shadow = failures.sensorNoise.probability ?: world.shadowNoiseRisk
        val priorities = PhotographicPriorities(
            motionFreeze = maxOf(motion, profile?.motionFreeze ?: 0.0).coerceIn(0.0, 1.0),
            highlightProtection = maxOf(highlight, profile?.highlightProtection ?: 0.0).coerceIn(0.0, 1.0),
            shadowRecovery = shadow.coerceIn(0.0, 1.0),
            skinNaturalness = if (world.primaryHumanSubject) maxOf(0.72, profile?.skinNaturalness ?: 0.0) else 0.25,
            atmospherePreservation = (profile?.atmospherePriority ?: 0.55).coerceIn(0.0, 1.0),
            detailPreservation = (1.0 - 0.32 * shadow).coerceIn(0.45, 1.0)
        )
        val lowLight = world.illumination in setOf(IlluminationRegime.LOW, IlluminationRegime.VERY_LOW, IlluminationRegime.NEAR_DARK)
        val highMotion = motion >= manifest.policy.riskThresholds.highMotion
        val captureFamily = when {
            lowLight && world.primaryHumanSubject && highMotion && world.sceneBranch != "selfie" -> CaptureFamily.FLASH_FUSION_CANDIDATE
            lowLight && world.sceneBranch == "selfie" -> CaptureFamily.SCREEN_FLASH_CANDIDATE
            highMotion -> CaptureFamily.FAST_BURST
            world.dynamicRangeStops >= 10.0 || highlight >= 0.45 -> CaptureFamily.HDR_BURST
            lowLight -> CaptureFamily.LOW_LIGHT_BURST
            else -> CaptureFamily.SAFE_SINGLE
        }
        val illuminationAction = when (captureFamily) {
            CaptureFamily.FLASH_FUSION_CANDIDATE -> IlluminationAction.CONSIDER_FLASH_FUSION
            CaptureFamily.SCREEN_FLASH_CANDIDATE -> IlluminationAction.CONSIDER_SCREEN_FLASH
            CaptureFamily.HDR_BURST, CaptureFamily.LOW_LIGHT_BURST -> IlluminationAction.AVOID_FLASH
            else -> IlluminationAction.AUTO
        }
        val requestedAggressiveness = (0.42 + theme.strength * 0.58).coerceIn(0.0, 1.0)
        val aggressiveness = reliabilityPlane.aggressiveness(reliability, requestedAggressiveness)
        val subjectRelight = ((profile?.subjectRelightEv ?: 0.0) * aggressiveness).coerceIn(-0.4, 0.8)
        val backgroundSuppression = ((profile?.backgroundSuppressionEv ?: 0.0) * aggressiveness).coerceIn(-0.6, 0.1)
        val objective = buildString {
            append("Preserve physical scene truth")
            if (world.primaryHumanSubject) append(" and natural subject identity")
            append("; prioritize ")
            append(if (priorities.motionFreeze >= priorities.highlightProtection) "motion fidelity" else "highlight fidelity")
            if (!theme.abstained) append("; render with ").append(theme.displayName)
        }
        return PhotographicIntent(
            stateId = world.stateId,
            objective = objective,
            priorities = priorities,
            capture = CaptureIntent(
                family = captureFamily,
                illuminationAction = illuminationAction,
                motionFreezePriority = priorities.motionFreeze,
                highlightProtectionPriority = priorities.highlightProtection,
                shadowRecoveryPriority = priorities.shadowRecovery
            ),
            render = LiveRenderIntent(
                theme = theme,
                highlightProtection = priorities.highlightProtection,
                shadowRecovery = priorities.shadowRecovery,
                subjectRelightEv = subjectRelight,
                backgroundSuppressionEv = backgroundSuppression,
                atmospherePreservation = priorities.atmospherePreservation,
                skinNaturalness = priorities.skinNaturalness,
                aggressiveness = aggressiveness
            ),
            expiresAtNs = nowNs + manifest.policy.intentTtlMs * 1_000_000L
        )
    }
}
