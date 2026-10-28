package com.saunak.director.director

import com.saunak.director.processing.TechnicalQuality
import kotlin.math.abs

/**
 * Continuous, measurement-driven render planning for Sierra beta. This replaces named-look fallback
 * as the local creative mechanism. It never pretends to be Gemini and never substitutes for a
 * missing Gemini Render Spec in AI mode.
 */
object SierraBetaRenderPlanner {
    fun plan(
        quality: TechnicalQuality,
        signals: SierraSceneSignals,
        situation: SierraSituationSignature,
        workingLongEdge: Int = 2560
    ): SierraRenderSpec {
        val lowLight = situation.lowLight
        val human = signals.primaryHumanSubject
        val highlightRisk = (quality.brightClipFraction * 5.0 + (quality.highlightP95 - 220.0).coerceAtLeast(0.0) / 80.0).coerceIn(0.0, 1.0)
        val shadowRisk = (quality.darkClipFraction * 4.0 + (65.0 - quality.shadowP05).coerceAtLeast(0.0) / 90.0).coerceIn(0.0, 1.0)
        val flatness = ((28.0 - quality.lumaStdDev) / 28.0).coerceIn(0.0, 1.0)
        val detailEvidence = (quality.edgeEnergy / 10.0).coerceIn(0.0, 1.0)
        val medianTargetError = ((118.0 - quality.medianLuma) / 150.0).coerceIn(-0.65, 0.65)
        val exposure = when {
            highlightRisk > 0.55 -> (-0.32 * highlightRisk).coerceAtLeast(-0.42)
            lowLight && quality.highlightP95 < 185.0 -> (0.28 * shadowRisk + 0.10).coerceAtMost(0.38)
            else -> medianTargetError * 0.20
        }
        val separation = signals.subjectSeparation.coerceIn(0.0, 1.0)
        val subjectLift = if (human && quality.medianLuma < 112.0) (0.10 + 0.24 * shadowRisk).coerceAtMost(0.30) else 0.0
        val backgroundEv = if (human) (-0.08 - 0.18 * separation).coerceAtLeast(-0.28) else 0.0
        val blackAnchor = when {
            lowLight -> (0.82 + 0.12 * (1.0 - shadowRisk)).coerceIn(0.76, 0.94)
            else -> (0.66 + 0.18 * highlightRisk).coerceIn(0.58, 0.88)
        }
        val saturation = when {
            lowLight -> 0.94
            human -> 0.99
            situation.brightSky -> 1.03
            else -> 1.0
        }
        val localContrast = (0.06 + 0.20 * flatness + 0.10 * (1.0 - highlightRisk)).coerceIn(0.02, 0.28)
        val detail = (0.24 + 0.34 * detailEvidence - if (lowLight) 0.10 else 0.0).coerceIn(0.12, 0.62)

        return SierraRenderSpec(
            intentSummary = buildString {
                append("Sierra beta measured ").append(situation.recommendedStrategy.name.lowercase().replace('_', ' '))
                append(" render")
                if (lowLight) append(" • low-light black and practical preservation")
                if (highlightRisk > 0.35) append(" • controlled highlights")
                if (human) append(" • natural subject hierarchy")
            }.take(180),
            tonal = SierraTonalResponse(
                exposureEv = exposure,
                blackPoint = 0.0,
                toeStrength = (0.20 + 0.38 * blackAnchor).coerceIn(0.18, 0.60),
                midtoneContrast = (0.08 + 0.26 * flatness - 0.10 * highlightRisk).coerceIn(-0.05, 0.30),
                shoulderStrength = (0.40 + 0.48 * highlightRisk).coerceIn(0.35, 0.90),
                dynamicRangeCompression = (0.12 + 0.50 * highlightRisk + 0.10 * shadowRisk).coerceIn(0.10, 0.72)
            ),
            color = SierraColorScience(
                temperature = 0.0,
                tint = 0.0,
                saturation = saturation,
                vibrance = if (human) 0.02 else (0.04 + 0.08 * flatness).coerceAtMost(0.12),
                shadowHueShiftDeg = 0.0,
                shadowColorStrength = 0.0,
                highlightHueShiftDeg = 0.0,
                highlightColorStrength = 0.0,
                skinProtection = if (human) 0.92 else 0.74
            ),
            localLight = SierraLocalLight(
                subjectEv = subjectLift,
                backgroundEv = backgroundEv,
                faceEv = 0.0,
                rimEmphasis = 0.0,
                dodgeBurnStrength = if (human) (0.08 + separation * 0.12).coerceAtMost(0.20) else 0.04
            ),
            hierarchy = SierraSubjectHierarchy(
                subjectSeparation = if (human) (0.20 + separation * 0.45).coerceIn(0.18, 0.72) else 0.10,
                backgroundPresence = if (human) 0.80 else 1.0,
                luminousSourcePriority = if (lowLight) 0.94 else 0.82,
                textPreservation = 1.0
            ),
            materials = SierraMaterialRendering(
                skinTexture = if (human) 0.94 else 0.82,
                hairDetail = if (human) 0.86 else 0.80,
                fabricDetail = 0.84,
                foliageDetail = if (situation.recommendedStrategy == DirectorStrategy.LANDSCAPE) 0.90 else 0.78,
                metalSpecular = 0.92,
                glassSpecular = 0.94
            ),
            highlights = SierraHighlightBehavior(
                rolloff = (0.48 + 0.42 * highlightRisk).coerceIn(0.42, 0.92),
                compression = (0.22 + 0.48 * highlightRisk).coerceIn(0.18, 0.75),
                bloom = 0.0,
                halation = 0.0,
                practicalLightPreservation = if (lowLight) 0.96 else 0.88
            ),
            shadows = SierraShadowBehavior(
                depth = blackAnchor,
                recovery = (0.16 + 0.28 * shadowRisk).coerceIn(0.12, 0.44),
                blackAnchor = blackAnchor,
                colorStrength = 0.0
            ),
            texture = SierraTextureModel(
                denoise = 0.0,
                chromaDenoise = 0.0,
                detail = detail,
                microcontrast = (detail * 0.68).coerceIn(0.10, 0.42),
                sharpening = (0.14 + detailEvidence * 0.18).coerceIn(0.12, 0.34),
                deblur = 0.0
            ),
            optical = SierraOpticalCharacter(
                diffusion = 0.0,
                vignette = if (human) 0.06 else 0.0,
                edgeSoftness = 0.0,
                lensFalloff = if (human) 0.03 else 0.0
            ),
            film = SierraFilmResponse(
                curveStrength = (0.08 + 0.16 * flatness).coerceIn(0.06, 0.24),
                grainAmount = if (lowLight && detailEvidence < 0.25) 0.02 else 0.0,
                grainSize = 0.20,
                colorResponse = 0.0,
                character = "measured_beta"
            ),
            atmosphere = SierraAtmosphere(
                dehaze = if (situation.recommendedStrategy in setOf(DirectorStrategy.LANDSCAPE, DirectorStrategy.ARCHITECTURE)) (0.04 + flatness * 0.10).coerceAtMost(0.16) else 0.0,
                glow = 0.0,
                haze = 0.0,
                localContrast = localContrast
            ),
            spatial = SierraSpatialMasks(
                useSubjectMask = human,
                useDepth = human && signals.relativeDepthConfidence >= 0.28,
                useFaceMask = false,
                useSkinMask = false,
                useSkyMask = false,
                feather = 0.34
            ),
            finishing = SierraFinishing(
                outputContrast = (0.02 + flatness * 0.08).coerceAtMost(0.10),
                outputSaturation = 1.0,
                outputSharpen = (0.05 + detailEvidence * 0.08).coerceAtMost(0.13),
                preserveBlacks = true
            ),
            execution = SierraRenderExecution(
                workingLongEdge = workingLongEdge.coerceIn(1024, 4096),
                precision = "linear_srgb_argb8888",
                subjectMaskMethod = if (human) "tracked_subject_mask_if_available" else "none",
                depthMethod = if (human && signals.relativeDepthConfidence >= 0.28) "relative_subject_depth_if_available" else "none",
                operations = listOf("tone_curve", "color_matrix", "local_light", "texture", "optical_finish"),
                deviceRationale = "Continuous beta render from measured luminance, clipping, detail, scene structure and subject evidence."
            ),
            reasonCodes = buildList {
                add("beta_continuous_render")
                if (lowLight) add("low_light")
                if (highlightRisk > 0.35) add("highlight_risk")
                if (shadowRisk > 0.35) add("shadow_risk")
                if (human) add("human_subject")
                if (abs(exposure) > 0.05) add("measured_exposure_bias")
            }.take(8)
        )
    }
}
