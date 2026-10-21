package com.saunak.director

import com.saunak.director.ai.VlmContractValidator
import com.saunak.director.director.EvaluationOutput
import com.saunak.director.director.PlanOutput
import com.saunak.director.director.ToolCall
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class VlmContractValidatorTest {
    @Test fun rejectsOutOfRangePlanScore() {
        assertThrows(IllegalArgumentException::class.java) {
            VlmContractValidator.plan(PlanOutput("scene", "objective", "general", 1.2, false))
        }
    }

    @Test fun rejectsTooManyCalls() {
        val calls = List(9) { ToolCall(30) }
        assertThrows(IllegalArgumentException::class.java) {
            VlmContractValidator.plan(PlanOutput("scene", "objective", "general", .5, false, calls))
        }
    }

    @Test fun validatesEvaluationBounds() {
        val result = VlmContractValidator.evaluation(EvaluationOutput(.8, .8, .8, .8, .8, true, listOf("good"), emptyList()))
        assertEquals(.8, result.score, .0001)
    }
}
