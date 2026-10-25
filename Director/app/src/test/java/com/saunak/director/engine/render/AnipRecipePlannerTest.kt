package com.saunak.director.engine.render

import com.saunak.director.engine.subject.NormalizedRect
import com.saunak.director.processing.Look
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AnipRecipePlannerTest {
    private fun stats(
        p50: Double = 0.28,
        p05: Double = 0.03,
        p95: Double = 0.96,
        shadows: Double = 0.34,
        highlights: Double = 0.08,
        saturation: Double = 0.24,
        edge: Double = 0.045,
        noise: Double = 0.028,
        subject: Double? = 0.22,
        background: Double? = 0.42
    ) = ImageStatistics(
        meanLuma = 0.34,
        p05Luma = p05,
        p50Luma = p50,
        p95Luma = p95,
        shadowFraction = shadows,
        highlightFraction = highlights,
        meanSaturation = saturation,
        edgeEnergy = edge,
        noiseProxy = noise,
        subjectMeanLuma = subject,
        backgroundMeanLuma = background
    )

    @Test fun darkPortraitGetsVisibleRelightAndRecovery() {
        val request = AnipRequest(
            sceneLabel = "night portrait",
            objective = "subject separation and neon atmosphere",
            look = Look.NIGHT_NEON,
            lookStrength = 0.8f,
            subjectPresent = true,
            motionRisk = 0.25,
            reconstructionIntegrity = 0.92
        )
        val recipe = AnipRecipePlanner.base(request, stats())
        assertTrue(recipe.subjectRelightEv >= 0.35)
        assertTrue(recipe.shadowRecovery >= 0.55)
        assertTrue(recipe.highlightCompression >= 0.55)
        assertTrue(recipe.backgroundSuppressionEv <= -0.12)
        assertTrue(recipe.routing.denoise >= 0.55)
        assertTrue(recipe.routing.detailRecovery >= 0.35)
    }

    @Test fun highDynamicRangeSceneCompressesHighlightsAggressively() {
        val request = AnipRequest("landscape sky", "recover clouds", Look.LANDSCAPE_VIVID, 0.7f, false, 0.08, 0.97)
        val recipe = AnipRecipePlanner.base(request, stats(p50 = 0.48, shadows = 0.16, highlights = 0.14, subject = null, background = null))
        assertTrue(recipe.highlightCompression >= 0.70)
        assertTrue(recipe.dehaze >= 0.18)
        assertTrue(recipe.localContrast >= 0.35)
        assertTrue(recipe.saturation > 1.05)
    }

    @Test fun explicitMonochromeRemainsMonochrome() {
        val request = AnipRequest("portrait", "classic monochrome", Look.MONOCHROME, 1f, true, 0.12, 0.95)
        val recipe = AnipRecipePlanner.base(request, stats())
        assertEquals(0.0, recipe.saturation, 1e-9)
        assertTrue(AnipRecipePlanner.candidates(recipe).all { it.saturation < 0.20 })
    }

    @Test fun candidateSearchContainsStrongerVisibleOptions() {
        val request = AnipRequest("street", "impact", Look.SOCIAL_POP, 0.8f, false, 0.1, 0.90)
        val base = AnipRecipePlanner.base(request, stats(subject = null, background = null))
        val candidates = AnipRecipePlanner.candidates(base)
        assertEquals(4, candidates.size)
        assertTrue(candidates.last().localContrast > candidates.first().localContrast)
        assertTrue(candidates.last().shadowRecovery >= candidates.first().shadowRecovery)
    }

    @Test fun subjectRenderFieldIsSmoothAndDirectional() {
        val request = AnipRequest("portrait", "separate subject", Look.CLEAN_PORTRAIT, 0.6f, true, 0.15, 0.94)
        val recipe = AnipRecipePlanner.base(request, stats())
        val subject = NormalizedRect(0.32, 0.18, 0.68, 0.82)
        val field = AnipRenderFieldBuilder.build(recipe, subject, width = 48, height = 48)
        val center = field.exposureEv[24 * 48 + 24]
        val corner = field.exposureEv[0]
        assertTrue(center > corner + 0.15f)
        var maxAdjacentJump = 0f
        for (y in 0 until 48) for (x in 0 until 47) {
            val a = field.exposureEv[y * 48 + x]
            val b = field.exposureEv[y * 48 + x + 1]
            maxAdjacentJump = maxOf(maxAdjacentJump, kotlin.math.abs(a - b))
        }
        assertTrue(maxAdjacentJump < 0.12f)
    }

    @Test fun semanticMaskOverridesBoxForSubjectRelight() {
        val request = AnipRequest("portrait", "strong subject separation", Look.CLEAN_PORTRAIT, 0.8f, true, 0.10, 0.96)
        val recipe = AnipRecipePlanner.base(request, stats())
        val mask = FloatArray(16 * 16)
        for (y in 2 until 14) for (x in 9 until 15) mask[y * 16 + x] = 1f
        val context = AnipSubjectContext(
            bounds = NormalizedRect(0.18, 0.12, 0.88, 0.88),
            maskWidth = 16,
            maskHeight = 16,
            mask = mask,
            maskConfidence = 1.0,
            depthSeparation = 0.65,
            trackConfidence = 0.95
        )
        val field = AnipRenderFieldBuilder.build(recipe, context, width = 48, height = 48)
        val maskSide = field.subjectWeight[24 * 48 + 38]
        val emptySide = field.subjectWeight[24 * 48 + 10]
        assertTrue(maskSide > emptySide + 0.55f)
        assertTrue(field.exposureEv[24 * 48 + 38] > field.exposureEv[24 * 48 + 10] + 0.25f)
    }
}
