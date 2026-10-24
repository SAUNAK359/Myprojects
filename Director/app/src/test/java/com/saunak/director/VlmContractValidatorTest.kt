package com.saunak.director

import com.saunak.director.ai.VlmContractValidator
import com.saunak.director.director.EvaluationOutput
import com.saunak.director.director.PlanOutput
import com.saunak.director.director.ToolCall
import org.junit.Assert.assertEquals
import org.junit.Test

class VlmContractValidatorTest {
    @Test fun clampsOutOfRangePlanScore() {
        val result = VlmContractValidator.plan(PlanOutput("scene", "objective", "general", 1.2, false))
        assertEquals(1.0, result.referenceScore, .0001)
    }

    @Test fun capsTooManyCalls() {
        val calls = List(9) { ToolCall(30) }
        val result = VlmContractValidator.plan(PlanOutput("scene", "objective", "general", .5, false, calls))
        assertEquals(8, result.toolCalls.size)
    }

    @Test fun validatesEvaluationBounds() {
        val result = VlmContractValidator.evaluation(EvaluationOutput(.8, .8, .8, .8, .8, true, listOf("good"), emptyList()))
        assertEquals(.8, result.score, .0001)
    }
}
