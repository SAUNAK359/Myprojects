package com.saunak.director.engine.cognition

import com.saunak.director.processing.Look
import kotlin.math.abs

class ThemeCortex(private val manifest: CognitionManifest) {
    fun select(
        world: PhotographicWorldState,
        failures: FailureForecast,
        reliability: ReliabilityEnvelope
    ): ThemeSelection {
        if (reliability.effectiveConfidence < manifest.policy.minStyleReliability) {
            return naturalFallback("reliability_below_style_threshold")
        }
        val weights = manifest.policy.selectorWeights
        val illuminationName = world.illumination.name.lowercase()
        val lowLightLevel = when (world.illumination) {
            IlluminationRegime.NEAR_DARK -> 1.0
            IlluminationRegime.VERY_LOW -> 0.9
            IlluminationRegime.LOW -> 0.75
            IlluminationRegime.DIM -> 0.45
            else -> 0.1
        }
        val riskBalance = (1.0 - 0.55 * (failures.motionBlur.probability ?: 0.0) -
            0.25 * (failures.highlightClipping.probability ?: 0.0)).coerceIn(0.0, 1.0)

        val candidates = manifest.themes.asSequence()
            .filterNot { world.sceneBranch in it.forbiddenBranches }
            .map { theme ->
                val sceneScore = when {
                    world.sceneBranch in theme.preferredBranches -> 1.0
                    "general" in theme.preferredBranches -> 0.55
                    theme.preferredBranches.isEmpty() -> 0.45
                    else -> 0.20
                }
                val illuminationScore = when {
                    illuminationName in theme.preferredIllumination -> 1.0
                    theme.preferredIllumination.isEmpty() -> 0.55
                    else -> 0.25
                }
                val humanScore = if (world.primaryHumanSubject) theme.humanAffinity else (1.0 - 0.55 * theme.humanAffinity)
                val lowLightScore = (1.0 - abs(lowLightLevel - theme.lowLightAffinity)).coerceIn(0.0, 1.0)
                val atmosphereScore = theme.atmospherePriority
                val raw = (
                    weights.scene * sceneScore +
                        weights.illumination * illuminationScore +
                        weights.human * humanScore +
                        weights.lowLight * lowLightScore +
                        weights.risk * riskBalance +
                        weights.atmosphere * atmosphereScore
                    ) / weights.total
                ScoredTheme(theme, raw.coerceIn(0.0, 1.0))
            }
            .sortedByDescending { it.score }
            .toList()

        val best = candidates.firstOrNull() ?: return naturalFallback("theme_catalog_empty")
        if (best.score < manifest.policy.minThemeScore) return naturalFallback("no_theme_cleared_min_score")
        val second = candidates.getOrNull(1)?.score ?: 0.0
        val margin = (best.score - second).coerceIn(0.0, 1.0)
        val confidence = (0.55 * reliability.effectiveConfidence + 0.30 * best.score + 0.15 * margin).coerceIn(0.0, 1.0)
        val strength = (best.theme.baseStrength * confidence.toFloat()).coerceIn(0f, 1f)
        return ThemeSelection(
            themeId = best.theme.id,
            displayName = best.theme.displayName,
            fallbackLook = best.theme.look(),
            strength = strength,
            score = best.score,
            confidence = confidence,
            reasonCodes = listOf(
                "scene:${world.sceneBranch}",
                "illumination:$illuminationName",
                "catalog_rank:1"
            )
        )
    }

    fun profile(themeId: String): ThemeProfileAsset? = manifest.themes.firstOrNull { it.id == themeId }

    private fun naturalFallback(reason: String): ThemeSelection = ThemeSelection(
        themeId = "natural_safety",
        displayName = "Natural",
        fallbackLook = Look.NATURAL,
        strength = 0f,
        score = 1.0,
        confidence = 1.0,
        reasonCodes = listOf(reason),
        abstained = true
    )

    private data class ScoredTheme(val theme: ThemeProfileAsset, val score: Double)
}
