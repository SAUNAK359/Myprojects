package com.saunak.director

import com.saunak.director.director.SierraLocalPlanner
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
        score: Double = 0.82
    ) = TechnicalQuality(mean, std, dark, bright, 8.0, false, score)

    @Test fun portraitFramesChooseFourFiveAndPortraitFinish() {
        val plan = SierraLocalPlanner.plan(quality(), 720, 1280)
        assertEquals("portrait", plan.skillBranch)
        assertEquals("4:5", plan.aspectRatio)
        assertEquals("sierra_beta", plan.planner)
        assertTrue(plan.toolCalls.any { it.id == ToolIds.LOOK_CLEAN_PORTRAIT })
    }

    @Test fun lowLightAddsSafePositiveExposureAndNightFinish() {
        val plan = SierraLocalPlanner.plan(quality(mean = 42.0), 1280, 720)
        assertEquals("low_light", plan.skillBranch)
        assertTrue(plan.toolCalls.any { it.id == ToolIds.EXPOSURE_EV && it.value == 0.33 })
        assertTrue(plan.toolCalls.any { it.id == ToolIds.LOOK_NIGHT_NEON })
        assertTrue(plan.recapture)
    }

    @Test fun brightFramePullsExposureDown() {
        val plan = SierraLocalPlanner.plan(quality(mean = 215.0, bright = 0.08), 1280, 960)
        assertTrue(plan.toolCalls.any { it.id == ToolIds.EXPOSURE_EV && it.value == -0.33 })
    }

    @Test fun cropPolicyRejectsUnknownRatios() {
        assertTrue(SierraCropPolicy.isSupported("4:5"))
        assertTrue(SierraCropPolicy.isSupported("2.39:1"))
        assertFalse(SierraCropPolicy.isSupported("7:5"))
        assertEquals(4 to 5, SierraCropPolicy.ratioOrNull("4:5"))
    }
}
