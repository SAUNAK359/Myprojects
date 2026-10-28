package com.saunak.director

import com.saunak.director.director.PlanOutput
import com.saunak.director.director.SierraPlanReconciler
import com.saunak.director.director.SierraSceneSignals
import com.saunak.director.director.ToolCall
import com.saunak.director.tools.ToolIds
import org.junit.Assert.assertEquals
import org.junit.Test

class SierraPlanReconcilerTest {
    @Test fun cloudPhotographicAuthorityPassesThroughUnchanged() {
        val cloud = PlanOutput(
            scene = "portrait",
            objective = "isolate subject",
            skillBranch = "portrait",
            toolCalls = listOf(
                ToolCall(ToolIds.EXPOSURE_EV, value = 0.33)
            ),
            planner = "sierra_ai"
        )
        val local = PlanOutput(
            scene = "architecture",
            objective = "protect sky and structure",
            skillBranch = "architecture",
            toolCalls = listOf(ToolCall(ToolIds.EXPOSURE_EV, value = -0.17)),
            planner = "sierra_beta"
        )
        val result = SierraPlanReconciler.reconcile(
            cloud,
            local,
            SierraSceneSignals("architecture", .84, primaryHumanSubject = false, brightSky = true, urbanStructure = true)
        )
        assertEquals(cloud, result)
    }
}
