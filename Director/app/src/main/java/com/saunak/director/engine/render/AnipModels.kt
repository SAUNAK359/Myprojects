package com.saunak.director.engine.render

import com.saunak.director.engine.subject.NormalizedRect
import com.saunak.director.processing.Look
import kotlin.math.max

enum class AnipSituation {
    SELFIE,
    SELFIE_LOW_LIGHT,
    PORTRAIT,
    PORTRAIT_LOW_LIGHT,
    NIGHT,
    NIGHT_NEON,
    SUNSET,
    STAGE,
    INTERIOR,
    LANDSCAPE,
    ARCHITECTURE,
    FOOD,
    PRODUCT,
    MACRO,
    DOCUMENT,
    GENERAL
}

data class AnipRequest(
    val sceneLabel: String,
    val objective: String,
    val look: Look,
    val lookStrength: Float,
    val subjectPresent: Boolean,
    val motionRisk: Double,
    val reconstructionIntegrity: Double = 1.0,
    val story: String = "",
    val mood: String = "",
    val primarySubject: String = "",
    val narrativeRole: String = "",
    val atmospherePriority: Double = 0.5,
    val skinNaturalness: Double = 0.8,
    val highlightProtectionIntent: Double = 0.6,
    val shadowDepthIntent: Double = 0.5,
    val midtoneContrastIntent: Double = 0.5,
    val localContrastIntent: Double = 0.5,
    val colorIntensityIntent: Double = 0.5,
    val warmthBiasIntent: Double = 0.0,
    val subjectRelightIntentEv: Double = 0.0,
    val backgroundSuppressionIntentEv: Double = 0.0,
    val directionConfidence: Double = 0.0
) {
    init {
        require(lookStrength in 0f..1f)
        require(motionRisk in 0.0..1.0)
        require(reconstructionIntegrity in 0.0..1.0)
        require(atmospherePriority in 0.0..1.0)
        require(skinNaturalness in 0.0..1.0)
        require(highlightProtectionIntent in 0.0..1.0)
        require(shadowDepthIntent in 0.0..1.0)
        require(midtoneContrastIntent in 0.0..1.0)
        require(localContrastIntent in 0.0..1.0)
        require(colorIntensityIntent in 0.0..1.0)
        require(warmthBiasIntent in -0.25..0.25)
        require(subjectRelightIntentEv in -0.4..0.8)
        require(backgroundSuppressionIntentEv in -0.6..0.1)
        require(directionConfidence in 0.0..1.0)
    }

    fun semanticText(): String = listOf(sceneLabel, objective, story, mood, primarySubject, narrativeRole)
        .joinToString(" ")
        .lowercase()
}

data class AnipSubjectContext(
    val bounds: NormalizedRect,
    val maskWidth: Int = 0,
    val maskHeight: Int = 0,
    val mask: FloatArray? = null,
    val maskConfidence: Double = 0.0,
    val depthSeparation: Double = 0.0,
    val trackConfidence: Double = 1.0
) {
    init {
        require(maskConfidence in 0.0..1.0)
        require(depthSeparation in -1.0..1.0)
        require(trackConfidence in 0.0..1.0)
        if (mask != null) {
            require(maskWidth > 0 && maskHeight > 0)
            require(mask.size == maskWidth * maskHeight)
            require(mask.all { it.isFinite() && it in 0f..1f })
        }
    }
}

data class ImageStatistics(
    val meanLuma: Double,
    val p05Luma: Double,
    val p50Luma: Double,
    val p95Luma: Double,
    val shadowFraction: Double,
    val highlightFraction: Double,
    val meanSaturation: Double,
    val edgeEnergy: Double,
    val noiseProxy: Double,
    val subjectMeanLuma: Double? = null,
    val backgroundMeanLuma: Double? = null
) {
    init {
        require(listOf(meanLuma, p05Luma, p50Luma, p95Luma, shadowFraction, highlightFraction,
            meanSaturation, edgeEnergy, noiseProxy).all { it.isFinite() })
    }
}

enum class AnipStyle { NATURAL_PRO, IMPACT, CINEMATIC }

data class RestorationRouting(
    val denoise: Double,
    val microDeblur: Double,
    val chromaRepair: Double,
    val detailRecovery: Double,
    val opticalCompensation: Double
) {
    init {
        require(listOf(denoise, microDeblur, chromaRepair, detailRecovery, opticalCompensation)
            .all { it in 0.0..1.0 })
    }
}

data class RenderRecipe(
    val style: AnipStyle,
    val globalExposureEv: Double,
    val shadowRecovery: Double,
    val highlightCompression: Double,
    val localContrast: Double,
    val saturation: Double,
    val warmth: Double,
    val coolShadowBias: Double,
    val subjectRelightEv: Double,
    val backgroundSuppressionEv: Double,
    val dehaze: Double,
    val vignette: Double,
    val grain: Double,
    val routing: RestorationRouting,
    val impact: Double,
    val situation: AnipSituation = AnipSituation.GENERAL,
    val blackAnchor: Double = 0.55,
    val toeStrength: Double = 0.35,
    val highlightGlowPreservation: Double = 0.45,
    val textureProtection: Double = 0.50,
    val skinProtection: Double = 0.50
) {
    init {
        require(globalExposureEv in -1.0..1.0)
        require(shadowRecovery in 0.0..1.0)
        require(highlightCompression in 0.0..1.0)
        require(localContrast in 0.0..1.0)
        require(saturation in 0.0..1.45)
        require(warmth in -0.25..0.25)
        require(coolShadowBias in 0.0..0.30)
        require(subjectRelightEv in -0.40..0.85)
        require(backgroundSuppressionEv in -0.60..0.10)
        require(dehaze in 0.0..0.65)
        require(vignette in 0.0..0.45)
        require(grain in 0.0..0.20)
        require(impact in 0.65..1.45)
        require(blackAnchor in 0.0..1.0)
        require(toeStrength in 0.0..1.0)
        require(highlightGlowPreservation in 0.0..1.0)
        require(textureProtection in 0.0..1.0)
        require(skinProtection in 0.0..1.0)
    }

    fun scaled(factor: Double, styleOverride: AnipStyle = style): RenderRecipe {
        val f = factor.coerceIn(0.70, 1.45)
        val scaledSaturation = if (saturation < 0.20) saturation else 1.0 + (saturation - 1.0) * f
        return copy(
            style = styleOverride,
            globalExposureEv = (globalExposureEv * f).coerceIn(-1.0, 1.0),
            shadowRecovery = (shadowRecovery * f).coerceIn(0.0, 1.0),
            highlightCompression = (highlightCompression * f).coerceIn(0.0, 1.0),
            localContrast = (localContrast * f).coerceIn(0.0, 1.0),
            saturation = scaledSaturation.coerceIn(0.0, 1.45),
            warmth = (warmth * f).coerceIn(-0.25, 0.25),
            coolShadowBias = (coolShadowBias * f).coerceIn(0.0, 0.30),
            subjectRelightEv = (subjectRelightEv * f).coerceIn(-0.40, 0.85),
            backgroundSuppressionEv = (backgroundSuppressionEv * f).coerceIn(-0.60, 0.10),
            dehaze = (dehaze * f).coerceIn(0.0, 0.65),
            vignette = (vignette * f).coerceIn(0.0, 0.45),
            grain = (grain * f).coerceIn(0.0, 0.20),
            routing = routing.copy(
                denoise = (routing.denoise * f).coerceIn(0.0, 1.0),
                microDeblur = (routing.microDeblur * f).coerceIn(0.0, 1.0),
                chromaRepair = (routing.chromaRepair * f).coerceIn(0.0, 1.0),
                detailRecovery = (routing.detailRecovery * f).coerceIn(0.0, 1.0),
                opticalCompensation = (routing.opticalCompensation * f).coerceIn(0.0, 1.0)
            ),
            impact = (impact * f).coerceIn(0.65, 1.45)
        )
    }
}

data class RenderField(
    val width: Int,
    val height: Int,
    val exposureEv: FloatArray,
    val contrast: FloatArray,
    val saturation: FloatArray,
    val warmth: FloatArray,
    val denoise: FloatArray,
    val detail: FloatArray,
    val dehaze: FloatArray,
    val subjectWeight: FloatArray
) {
    init {
        require(width > 1 && height > 1)
        val size = width * height
        require(listOf(exposureEv, contrast, saturation, warmth, denoise, detail, dehaze, subjectWeight)
            .all { values -> values.size == size && values.all { it.isFinite() } })
    }
}

data class RenderCandidateScore(
    val total: Double,
    val exposure: Double,
    val contrast: Double,
    val color: Double,
    val detail: Double,
    val subject: Double,
    val artifactPenalty: Double
) {
    init { require(total.isFinite()) }
}

data class AnipRenderResult(
    val recipe: RenderRecipe,
    val score: RenderCandidateScore,
    val sourceStats: ImageStatistics,
    val outputStats: ImageStatistics,
    val candidateCount: Int
) {
    init { require(candidateCount >= 1) }
}

internal fun Double.safeRatio(denominator: Double): Double = this / max(1e-6, denominator)
