package com.saunak.director

import com.saunak.director.director.PlanOutput
import com.saunak.director.director.SierraPlanReconciler
import com.saunak.director.director.SierraSceneSignals
import com.saunak.director.director.ToolCall
import com.saunak.director.tools.ToolIds
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SierraPlanReconcilerTest {
    @Test fun environmentalSceneRejectsPortraitOnlyCloudActions() {
        val cloud = PlanOutput(
            scene = "portrait",
            objective = "isolate subject",
            skillBranch = "portrait",
            toolCalls = listOf(
                ToolCall(ToolIds.EXTENSION, stringValue = "bokeh"),
                ToolCall(ToolIds.EXPOSURE_EV, value = 0.33),
                ToolCall(ToolIds.LOOK_CLEAN_PORTRAIT, value = .5)
            )
        )
        val local = PlanOutput(
            scene = "architecture",
            objective = "protect sky and structure",
            skillBranch = "architecture",
            toolCalls = listOf(
                ToolCall(ToolIds.EXPOSURE_EV, value = -0.17),
                ToolCall(ToolIds.LOOK_LANDSCAPE_VIVID, value = .30)
            )
        )
        val result = SierraPlanReconciler.reconcile(
            cloud,
            local,
            SierraSceneSignals("architecture", .84, primaryHumanSubject = false, brightSky = true, urbanStructure = true)
        )
        assertEquals("architecture", result.skillBranch)
        assertFalse(result.toolCalls.any { it.id == ToolIds.LOOK_CLEAN_PORTRAIT || it.stringValue == "bokeh" })
        assertFalse(result.toolCalls.any { it.id == ToolIds.EXPOSURE_EV && (it.value ?: 0.0) > 0.0 })
        assertTrue(result.toolCalls.any { it.id == ToolIds.EXPOSURE_EV && it.value == -0.17 })
        assertTrue(result.toolCalls.any { it.id == ToolIds.LOOK_LANDSCAPE_VIVID })
    }
}
