package com.saunak.director.director

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class SierraRenderSpec(
    @SerialName("schema_version") val schemaVersion: Int = 1,
    @SerialName("intent_summary") val intentSummary: String = "",
    @SerialName("tonal_response") val tonal: SierraTonalResponse = SierraTonalResponse(),
    @SerialName("color_science") val color: SierraColorScience = SierraColorScience(),
    @SerialName("local_light") val localLight: SierraLocalLight = SierraLocalLight(),
    @SerialName("subject_background_hierarchy") val hierarchy: SierraSubjectHierarchy = SierraSubjectHierarchy(),
    @SerialName("material_rendering") val materials: SierraMaterialRendering = SierraMaterialRendering(),
    @SerialName("highlight_behavior") val highlights: SierraHighlightBehavior = SierraHighlightBehavior(),
    @SerialName("shadow_behavior") val shadows: SierraShadowBehavior = SierraShadowBehavior(),
    @SerialName("texture_model") val texture: SierraTextureModel = SierraTextureModel(),
    @SerialName("optical_character") val optical: SierraOpticalCharacter = SierraOpticalCharacter(),
    @SerialName("film_response") val film: SierraFilmResponse = SierraFilmResponse(),
    @SerialName("atmosphere") val atmosphere: SierraAtmosphere = SierraAtmosphere(),
    @SerialName("spatial_masks") val spatial: SierraSpatialMasks = SierraSpatialMasks(),
    @SerialName("finishing") val finishing: SierraFinishing = SierraFinishing(),
    @SerialName("execution") val execution: SierraRenderExecution = SierraRenderExecution(),
    @SerialName("reason_codes") val reasonCodes: List<String> = emptyList()
)

@Serializable
data class SierraTonalResponse(
    @SerialName("exposure_ev") val exposureEv: Double = 0.0,
    @SerialName("black_point") val blackPoint: Double = 0.0,
    @SerialName("toe_strength") val toeStrength: Double = 0.25,
    @SerialName("midtone_contrast") val midtoneContrast: Double = 0.0,
    @SerialName("shoulder_strength") val shoulderStrength: Double = 0.45,
    @SerialName("dynamic_range_compression") val dynamicRangeCompression: Double = 0.25
)

@Serializable
data class SierraColorScience(
    @SerialName("temperature") val temperature: Double = 0.0,
    @SerialName("tint") val tint: Double = 0.0,
    @SerialName("saturation") val saturation: Double = 1.0,
    @SerialName("vibrance") val vibrance: Double = 0.0,
    @SerialName("shadow_hue_shift_deg") val shadowHueShiftDeg: Double = 0.0,
    @SerialName("shadow_color_strength") val shadowColorStrength: Double = 0.0,
    @SerialName("highlight_hue_shift_deg") val highlightHueShiftDeg: Double = 0.0,
    @SerialName("highlight_color_strength") val highlightColorStrength: Double = 0.0,
    @SerialName("skin_protection") val skinProtection: Double = 0.85
)

@Serializable
data class SierraLocalLight(
    @SerialName("subject_ev") val subjectEv: Double = 0.0,
    @SerialName("background_ev") val backgroundEv: Double = 0.0,
    @SerialName("face_ev") val faceEv: Double = 0.0,
    @SerialName("rim_emphasis") val rimEmphasis: Double = 0.0,
    @SerialName("dodge_burn_strength") val dodgeBurnStrength: Double = 0.0
)

@Serializable
data class SierraSubjectHierarchy(
    @SerialName("subject_separation") val subjectSeparation: Double = 0.0,
    @SerialName("background_presence") val backgroundPresence: Double = 1.0,
    @SerialName("luminous_source_priority") val luminousSourcePriority: Double = 0.75,
    @SerialName("text_preservation") val textPreservation: Double = 1.0
)

@Serializable
data class SierraMaterialRendering(
    @SerialName("skin_texture") val skinTexture: Double = 0.9,
    @SerialName("hair_detail") val hairDetail: Double = 0.8,
    @SerialName("fabric_detail") val fabricDetail: Double = 0.8,
    @SerialName("foliage_detail") val foliageDetail: Double = 0.8,
    @SerialName("metal_specular") val metalSpecular: Double = 0.9,
    @SerialName("glass_specular") val glassSpecular: Double = 0.9
)

@Serializable
data class SierraHighlightBehavior(
    @SerialName("rolloff") val rolloff: Double = 0.55,
    @SerialName("compression") val compression: Double = 0.35,
    @SerialName("bloom") val bloom: Double = 0.0,
    @SerialName("halation") val halation: Double = 0.0,
    @SerialName("practical_light_preservation") val practicalLightPreservation: Double = 0.85
)

@Serializable
data class SierraShadowBehavior(
    @SerialName("depth") val depth: Double = 0.5,
    @SerialName("recovery") val recovery: Double = 0.25,
    @SerialName("black_anchor") val blackAnchor: Double = 0.75,
    @SerialName("color_strength") val colorStrength: Double = 0.0
)

@Serializable
data class SierraTextureModel(
    @SerialName("denoise") val denoise: Double = 0.25,
    @SerialName("chroma_denoise") val chromaDenoise: Double = 0.25,
    @SerialName("detail") val detail: Double = 0.4,
    @SerialName("microcontrast") val microcontrast: Double = 0.25,
    @SerialName("sharpening") val sharpening: Double = 0.25,
    @SerialName("deblur") val deblur: Double = 0.0
)

@Serializable
data class SierraOpticalCharacter(
    @SerialName("diffusion") val diffusion: Double = 0.0,
    @SerialName("vignette") val vignette: Double = 0.0,
    @SerialName("edge_softness") val edgeSoftness: Double = 0.0,
    @SerialName("lens_falloff") val lensFalloff: Double = 0.0
)

@Serializable
data class SierraFilmResponse(
    @SerialName("curve_strength") val curveStrength: Double = 0.0,
    @SerialName("grain_amount") val grainAmount: Double = 0.0,
    @SerialName("grain_size") val grainSize: Double = 0.25,
    @SerialName("color_response") val colorResponse: Double = 0.0,
    @SerialName("character") val character: String = "clean"
)

@Serializable
data class SierraAtmosphere(
    @SerialName("dehaze") val dehaze: Double = 0.0,
    @SerialName("glow") val glow: Double = 0.0,
    @SerialName("haze") val haze: Double = 0.0,
    @SerialName("local_contrast") val localContrast: Double = 0.0
)

@Serializable
data class SierraSpatialMasks(
    @SerialName("use_subject_mask") val useSubjectMask: Boolean = false,
    @SerialName("use_depth") val useDepth: Boolean = false,
    @SerialName("use_face_mask") val useFaceMask: Boolean = false,
    @SerialName("use_skin_mask") val useSkinMask: Boolean = false,
    @SerialName("use_sky_mask") val useSkyMask: Boolean = false,
    @SerialName("feather") val feather: Double = 0.35
)

@Serializable
data class SierraFinishing(
    @SerialName("output_contrast") val outputContrast: Double = 0.0,
    @SerialName("output_saturation") val outputSaturation: Double = 1.0,
    @SerialName("output_sharpen") val outputSharpen: Double = 0.0,
    @SerialName("preserve_blacks") val preserveBlacks: Boolean = true
)

@Serializable
data class SierraRenderExecution(
    @SerialName("working_long_edge") val workingLongEdge: Int = 2560,
    @SerialName("precision") val precision: String = "linear_srgb_argb8888",
    @SerialName("subject_mask_method") val subjectMaskMethod: String = "none",
    @SerialName("depth_method") val depthMethod: String = "none",
    @SerialName("operations") val operations: List<String> = emptyList(),
    @SerialName("device_rationale") val deviceRationale: String = ""
)

object SierraRenderSpecFactory {
    /** Local beta fallback only. Never substitutes for a failed Gemini render specification. */
    fun fromDirection(direction: PhotographicDirection, workingLongEdge: Int = 2560): SierraRenderSpec {
        val confidence = direction.themeConfidence.coerceIn(0.0, 1.0)
        val intensity = direction.themeStrength.coerceIn(0.0, 1.0) * (0.55 + 0.45 * confidence)
        return SierraRenderSpec(
            intentSummary = listOf(direction.mood, direction.story).filter(String::isNotBlank).joinToString(" • ").take(180),
            tonal = SierraTonalResponse(
                midtoneContrast = ((direction.midtoneContrast - 0.5) * 1.4 * intensity).coerceIn(-0.75, 0.85),
                toeStrength = (0.15 + direction.shadowDepth * 0.75).coerceIn(0.0, 1.0),
                shoulderStrength = direction.highlightProtection.coerceIn(0.0, 1.0),
                dynamicRangeCompression = (direction.highlightProtection * 0.65).coerceIn(0.0, 1.0)
            ),
            color = SierraColorScience(
                temperature = (direction.warmthBias * 3.0).coerceIn(-0.75, 0.75),
                saturation = (0.76 + direction.colorIntensity * 0.52).coerceIn(0.55, 1.45),
                vibrance = ((direction.colorIntensity - 0.5) * 0.55).coerceIn(-0.45, 0.45),
                skinProtection = direction.skinNaturalness.coerceIn(0.0, 1.0)
            ),
            localLight = SierraLocalLight(
                subjectEv = direction.subjectRelightEv.coerceIn(-1.0, 1.0),
                backgroundEv = direction.backgroundSuppressionEv.coerceIn(-1.0, 0.5)
            ),
            hierarchy = SierraSubjectHierarchy(
                subjectSeparation = (kotlin.math.abs(direction.subjectRelightEv - direction.backgroundSuppressionEv) * 0.65).coerceIn(0.0, 1.0),
                luminousSourcePriority = direction.atmospherePriority.coerceIn(0.0, 1.0)
            ),
            materials = SierraMaterialRendering(skinTexture = direction.skinNaturalness.coerceIn(0.0, 1.0)),
            highlights = SierraHighlightBehavior(
                rolloff = direction.highlightProtection.coerceIn(0.0, 1.0),
                compression = (direction.highlightProtection * 0.70).coerceIn(0.0, 1.0),
                practicalLightPreservation = direction.atmospherePriority.coerceIn(0.0, 1.0)
            ),
            shadows = SierraShadowBehavior(
                depth = direction.shadowDepth.coerceIn(0.0, 1.0),
                recovery = (1.0 - direction.shadowDepth).coerceIn(0.0, 1.0),
                blackAnchor = (0.45 + direction.shadowDepth * 0.5).coerceIn(0.0, 1.0)
            ),
            texture = SierraTextureModel(
                detail = direction.localContrast.coerceIn(0.0, 1.0),
                microcontrast = direction.localContrast.coerceIn(0.0, 1.0)
            ),
            optical = SierraOpticalCharacter(vignette = direction.vignette.coerceIn(0.0, 0.6)),
            film = SierraFilmResponse(grainAmount = direction.grain.coerceIn(0.0, 0.35), character = direction.selectedThemeId.ifBlank { "clean" }.take(48)),
            atmosphere = SierraAtmosphere(localContrast = ((direction.localContrast - 0.5) * 1.2).coerceIn(-0.6, 0.8)),
            spatial = SierraSpatialMasks(useSubjectMask = true, useDepth = true),
            finishing = SierraFinishing(),
            execution = SierraRenderExecution(
                workingLongEdge = workingLongEdge.coerceIn(1024, 4096),
                subjectMaskMethod = "tracked_subject_mask_if_available",
                depthMethod = "relative_subject_depth_if_available",
                operations = listOf("tone_curve", "color_matrix", "subject_local_light", "texture", "optical_finish")
            ),
            reasonCodes = direction.reasonCodes.take(8)
        )
    }
}
