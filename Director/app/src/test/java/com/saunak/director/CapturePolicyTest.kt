package com.saunak.director

import com.saunak.director.director.CapturePolicy
import com.saunak.director.director.EvaluationOutput
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CapturePolicyTest {
    @Test fun neverExceedsConfiguredThreePhysicalCaptures() {
        assertEquals(1, CapturePolicy.physicalBudget(0))
        assertEquals(2, CapturePolicy.physicalBudget(2))
        assertEquals(3, CapturePolicy.physicalBudget(99))
    }

    @Test fun catastrophicFrameNeverPasses() {
        val eval = EvaluationOutput(.95, .95, .9, .9, .9, true)
        assertFalse(CapturePolicy.accepted(eval, true))
        assertTrue(CapturePolicy.accepted(eval, false))
    }
}
