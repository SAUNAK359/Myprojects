package com.saunak.director

import com.saunak.director.director.DirectorStrategy
import com.saunak.director.director.SierraBetaRenderPlanner
import com.saunak.director.director.SierraSceneSignals
import com.saunak.director.director.SierraSituationSignature
import com.saunak.director.processing.TechnicalQuality
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SierraBetaRenderPlannerTest {
    private fun quality(
        mean: Double = 118.0,
        std: Double = 38.0,
        dark: Double = .02,
        bright: Double = .01,
        p05: Double = 42.0,
        median: Double = 116.0,
        p95: Double = 208.0,
        edge: Double = 7.0
    ) = TechnicalQuality(
        meanLuma = mean,
        lumaStdDev = std,
        darkClipFraction = dark,
        brightClipFraction = bright,
        edgeEnergy = edge,
        catastrophic = false,
        heuristicScore = .84,
        shadowP05 = p05,
        medianLuma = median,
        highlightP95 = p95,
        topMeanLuma = mean + 5.0,
        bottomMeanLuma = mean - 5.0,
        horizontalEdgeEnergy = edge,
        verticalEdgeEnergy = edge,
        strongEdgeFraction = .12
    )

    @Test fun betaPlannerNeverRequestsUnsupportedRenderOperations() {
        val q = quality()
        val signals = SierraSceneSignals("portrait", .9, faceCount = 1, faceCoverage = .15, primaryHumanSubject = true, subjectSeparation = .45)
        val situation = SierraSituationSignature.from(signals, q, 1080, 1440)
        val spec = SierraBetaRenderPlanner.plan(q, signals, situation)
        assertEquals(0.0, spec.texture.denoise, .0)
        assertEquals(0.0, spec.texture.chromaDenoise, .0)
        assertEquals(0.0, spec.texture.deblur, .0)
        assertEquals(0.0, spec.optical.diffusion, .0)
        assertEquals(0.0, spec.optical.edgeSoftness, .0)
        assertTrue(!spec.spatial.useFaceMask && !spec.spatial.useSkinMask && !spec.spatial.useSkyMask)
        assertTrue(spec.finishing.preserveBlacks)
        assertEquals("linear_srgb_argb8888", spec.execution.precision)
    }

    @Test fun lowLightProtectsBlacksAndPracticalLightsWithoutGenericShadowLift() {
        val q = quality(mean = 44.0, std = 31.0, dark = .13, bright = .015, p05 = 8.0, median = 38.0, p95 = 178.0, edge = 3.5)
        val signals = SierraSceneSignals("low_light", .92)
        val situation = SierraSituationSignature.from(signals, q, 1440, 1080)
        val spec = SierraBetaRenderPlanner.plan(q, signals, situation)
        assertEquals(DirectorStrategy.NIGHT, situation.recommendedStrategy)
        assertTrue(spec.shadows.blackAnchor >= .76)
        assertTrue(spec.highlights.practicalLightPreservation >= .90)
        assertTrue(spec.shadows.recovery <= .44)
        assertTrue(spec.reasonCodes.contains("low_light"))
        assertEquals("measured_beta", spec.film.character)
    }

    @Test fun portraitUsesSubjectHierarchyAndOptionalMeasuredDepthOnly() {
        val q = quality()
        val signals = SierraSceneSignals(
            branch = "portrait",
            confidence = .92,
            faceCount = 1,
            faceCoverage = .18,
            primaryHumanSubject = true,
            subjectSeparation = .62,
            relativeDepthConfidence = .54
        )
        val situation = SierraSituationSignature.from(signals, q, 1080, 1440)
        val spec = SierraBetaRenderPlanner.plan(q, signals, situation)
        assertTrue(spec.hierarchy.subjectSeparation > .20)
        assertTrue(spec.spatial.useSubjectMask)
        assertTrue(spec.spatial.useDepth)
        assertTrue(spec.localLight.backgroundEv <= 0.0)
    }
}
