package com.saunak.director

import com.saunak.director.ai.VlmContractValidator
import com.saunak.director.director.EvaluationOutput
import com.saunak.director.director.FinalRenderAdjustment
import com.saunak.director.director.PhotographicDirection
import com.saunak.director.director.PlanOutput
import com.saunak.director.director.ThemeBlendComponent
import com.saunak.director.director.ToolCall
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class VlmContractValidatorTest {
    @Test fun clampsOutOfRangePlanScore() {
        val result = VlmContractValidator.plan(PlanOutput("scene", "objective", "general", 1.2, false))
        assertEquals(1.0, result.referenceScore, .0001)
    }

    @Test fun capsTooManyCallsIndependentlyOfLookAmbiguity() {
        val calls = List(9) { index -> ToolCall(100 + index) }
        val result = VlmContractValidator.plan(PlanOutput("scene", "objective", "general", .5, false, calls))
        assertEquals(8, result.toolCalls.size)
    }

    @Test fun legacyEvaluationConstructorRemainsCompatible() {
        val result = VlmContractValidator.evaluation(EvaluationOutput(.8, .8, .8, .8, .8, true, listOf("good"), emptyList()))
        assertEquals(.8, result.score, .0001)
        assertTrue(result.accept)
        assertEquals(.8, result.intentMatch, .0001)
    }

    @Test fun photographicDirectionIsAggressivelyBounded() {
        val direction = PhotographicDirection(
            story = "x".repeat(400),
            selectedThemeId = "urban_rain_noir",
            themeStrength = 4.0,
            themeConfidence = -2.0,
            themeBlend = listOf(
                ThemeBlendComponent("a", 2.0),
                ThemeBlendComponent("b", .4),
                ThemeBlendComponent("c", .3),
                ThemeBlendComponent("d", .2)
            ),
            warmthBias = 2.0,
            subjectRelightEv = 8.0,
            backgroundSuppressionEv = -8.0,
            grain = .9,
            vignette = .9
        )
        val result = VlmContractValidator.plan(
            PlanOutput("scene", "objective", "general", photographicDirection = direction)
        ).photographicDirection
        assertEquals(1.0, result.themeStrength, .0001)
        assertEquals(0.0, result.themeConfidence, .0001)
        assertEquals(3, result.themeBlend.size)
        assertEquals(.25, result.warmthBias, .0001)
        assertEquals(.8, result.subjectRelightEv, .0001)
        assertEquals(-.6, result.backgroundSuppressionEv, .0001)
        assertEquals(.20, result.grain, .0001)
        assertEquals(.35, result.vignette, .0001)
        assertTrue(result.story.length <= 160)
    }

    @Test fun finalReviewAdjustmentCannotEscapeSafetyEnvelope() {
        val result = VlmContractValidator.evaluation(
            EvaluationOutput(
                score = 2.0,
                accept = false,
                renderAdjustment = FinalRenderAdjustment(
                    themeStrengthDelta = 2.0,
                    midtoneContrastDelta = 2.0,
                    localContrastDelta = -2.0,
                    subjectRelightDeltaEv = 4.0,
                    backgroundSuppressionDeltaEv = -4.0,
                    warmthDelta = 1.0,
                    colorIntensityDelta = -1.0
                )
            )
        )
        assertEquals(1.0, result.score, .0001)
        assertEquals(.20, result.renderAdjustment.themeStrengthDelta, .0001)
        assertEquals(.25, result.renderAdjustment.midtoneContrastDelta, .0001)
        assertEquals(-.25, result.renderAdjustment.localContrastDelta, .0001)
        assertEquals(.25, result.renderAdjustment.subjectRelightDeltaEv, .0001)
        assertEquals(-.20, result.renderAdjustment.backgroundSuppressionDeltaEv, .0001)
        assertEquals(.10, result.renderAdjustment.warmthDelta, .0001)
        assertEquals(-.15, result.renderAdjustment.colorIntensityDelta, .0001)
    }
}
