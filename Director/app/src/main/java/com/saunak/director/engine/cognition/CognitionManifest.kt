package com.saunak.director.engine.cognition

import android.content.Context
import com.saunak.director.processing.Look
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class CognitionPolicy(
    @SerialName("schema_version") val schemaVersion: Int,
    @SerialName("stale_after_ms") val staleAfterMs: Long,
    @SerialName("intent_ttl_ms") val intentTtlMs: Long,
    @SerialName("min_style_reliability") val minStyleReliability: Double,
    @SerialName("min_theme_score") val minThemeScore: Double,
    @SerialName("illumination_thresholds") val illuminationThresholds: IlluminationThresholds,
    @SerialName("risk_thresholds") val riskThresholds: RiskThresholds,
    @SerialName("selector_weights") val selectorWeights: SelectorWeights
) {
    init {
        require(schemaVersion == CURRENT_SCHEMA)
        require(staleAfterMs > 0 && intentTtlMs > 0)
        require(minStyleReliability in 0.0..1.0 && minThemeScore in 0.0..1.0)
    }

    companion object { const val CURRENT_SCHEMA = 1 }
}

@Serializable
data class IlluminationThresholds(
    val daylight: Double,
    val bright: Double,
    val normal: Double,
    val dim: Double,
    val low: Double,
    @SerialName("very_low") val veryLow: Double
) {
    init { require(daylight > bright && bright > normal && normal > dim && dim > low && low > veryLow) }
}

@Serializable
data class RiskThresholds(
    @SerialName("highlight_start") val highlightStart: Double,
    @SerialName("shadow_floor") val shadowFloor: Double,
    @SerialName("high_motion") val highMotion: Double,
    @SerialName("medium_motion") val mediumMotion: Double,
    @SerialName("low_light_flash_luma") val lowLightFlashLuma: Double,
    @SerialName("rolling_shutter_severe_fraction") val rollingShutterSevereFraction: Double
)

@Serializable
data class SelectorWeights(
    val scene: Double,
    val illumination: Double,
    val human: Double,
    @SerialName("low_light") val lowLight: Double,
    val risk: Double,
    val atmosphere: Double
) {
    val total: Double get() = scene + illumination + human + lowLight + risk + atmosphere
    init { require(total > 0.0) }
}

@Serializable
data class ThemeProfileAsset(
    val id: String,
    @SerialName("display_name") val displayName: String,
    @SerialName("fallback_look") val fallbackLook: String,
    @SerialName("base_strength") val baseStrength: Float,
    @SerialName("preferred_branches") val preferredBranches: Set<String> = emptySet(),
    @SerialName("forbidden_branches") val forbiddenBranches: Set<String> = emptySet(),
    @SerialName("preferred_illumination") val preferredIllumination: Set<String> = emptySet(),
    @SerialName("human_affinity") val humanAffinity: Double,
    @SerialName("low_light_affinity") val lowLightAffinity: Double,
    @SerialName("highlight_protection") val highlightProtection: Double,
    @SerialName("atmosphere_priority") val atmospherePriority: Double,
    @SerialName("skin_naturalness") val skinNaturalness: Double,
    @SerialName("motion_freeze") val motionFreeze: Double,
    @SerialName("subject_relight_ev") val subjectRelightEv: Double,
    @SerialName("background_suppression_ev") val backgroundSuppressionEv: Double,
    @SerialName("negative_constraints") val negativeConstraints: Set<String> = emptySet()
) {
    init {
        require(id.isNotBlank() && displayName.isNotBlank())
        require(baseStrength in 0f..1f)
        listOf(humanAffinity, lowLightAffinity, highlightProtection, atmospherePriority, skinNaturalness, motionFreeze)
            .forEach { require(it in 0.0..1.0) }
        require(subjectRelightEv in -0.4..0.8)
        require(backgroundSuppressionEv in -0.6..0.1)
    }

    fun look(): Look = runCatching { Look.valueOf(fallbackLook) }.getOrDefault(Look.NATURAL)
}

@Serializable
data class CognitionManifest(
    val policy: CognitionPolicy,
    val themes: List<ThemeProfileAsset>
) {
    init { require(themes.map { it.id }.distinct().size == themes.size) }
}

object CognitionManifestLoader {
    const val ASSET_PATH = "sierra/cognition_v1.json"
    private val json = Json { ignoreUnknownKeys = false; explicitNulls = false }

    fun load(context: Context): CognitionManifest {
        val raw = context.assets.open(ASSET_PATH).bufferedReader().use { it.readText() }
        return json.decodeFromString(CognitionManifest.serializer(), raw)
    }
}
