package com.saunak.director

import com.saunak.director.director.SierraLocalPlanner
import com.saunak.director.director.SierraSceneSignals
import com.saunak.director.processing.TechnicalQuality
import com.saunak.director.storage.SierraCropPolicy
import com.saunak.director.tools.ToolIds
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SierraLocalPlannerTest {
    private fun quality(
        mean: Double = 120.0,
        std: Double = 40.0,
        dark: Double = 0.01,
        bright: Double = 0.01,
        score: Double = 0.82,
        p05: Double = 45.0,
        median: Double = 120.0,
        p95: Double = 205.0,
        top: Double = 125.0,
        bottom: Double = 115.0,
        strongEdges: Double = 0.12
    ) = TechnicalQuality(
        meanLuma = mean,
        lumaStdDev = std,
        darkClipFraction = dark,
        brightClipFraction = bright,
        edgeEnergy = 8.0,
        catastrophic = false,
        heuristicScore = score,
        shadowP05 = p05,
        medianLuma = median,
        highlightP95 = p95,
        topMeanLuma = top,
        bottomMeanLuma = bottom,
        horizontalEdgeEnergy = 8.0,
        verticalEdgeEnergy = 8.0,
        strongEdgeFraction = strongEdges
    )

    @Test fun portraitShapedFrameDoesNotAutomaticallyBecomePortraitScene() {
        val plan = SierraLocalPlanner.plan(quality(), 720, 1280)
        assertEquals("general", plan.skillBranch)
        assertEquals("4:5", plan.aspectRatio)
        assertTrue(plan.toolCalls.none { it.id in ToolIds.LOOK_IDS })
    }

    @Test fun explicitPrimaryHumanSignalUsesPortraitBranchWithoutLegacyLookTool() {
        val plan = SierraLocalPlanner.plan(
            quality(), 720, 1280,
            SierraSceneSignals("portrait", .95, faceCount = 1, faceCoverage = .12, primaryHumanSubject = true)
        )
        assertEquals("portrait", plan.skillBranch)
        assertEquals("4:5", plan.aspectRatio)
        assertTrue(plan.toolCalls.none { it.id in ToolIds.LOOK_IDS })
    }

    @Test fun brightUrbanSceneProtectsSkyWithCameraExposureOnly() {
        val plan = SierraLocalPlanner.plan(
            quality(mean = 166.0, std = 52.0, p05 = 72.0, median = 182.0, p95 = 234.0, top = 204.0, bottom = 136.0, strongEdges = .29),
            1536, 1152,
            SierraSceneSignals("architecture", .84, brightSky = true, urbanStructure = true)
        )
        assertEquals("architecture", plan.skillBranch)
        assertTrue(plan.toolCalls.any { it.id == ToolIds.EXPOSURE_EV && it.value == -0.17 })
        assertTrue(plan.toolCalls.none { it.id in ToolIds.LOOK_IDS })
    }

    @Test fun lowLightAddsSafePositiveExposureWithoutThemeBias() {
        val plan = SierraLocalPlanner.plan(
            quality(mean = 42.0, median = 38.0, p95 = 150.0),
            1280, 720,
            SierraSceneSignals("low_light", .9)
        )
        assertEquals("low_light", plan.skillBranch)
        assertTrue(plan.toolCalls.any { it.id == ToolIds.EXPOSURE_EV && it.value == 0.33 })
        assertTrue(plan.toolCalls.none { it.id in ToolIds.LOOK_IDS })
        assertTrue(plan.recapture)
    }

    @Test fun cropPolicyRejectsUnknownRatios() {
        assertTrue(SierraCropPolicy.isSupported("4:5"))
        assertTrue(SierraCropPolicy.isSupported("2.39:1"))
        assertFalse(SierraCropPolicy.isSupported("7:5"))
        assertEquals(4 to 5, SierraCropPolicy.ratioOrNull("4:5"))
    }
}
