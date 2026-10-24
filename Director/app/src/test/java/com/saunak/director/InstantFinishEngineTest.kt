package com.saunak.director

import com.saunak.director.director.PlanOutput
import com.saunak.director.director.ToolCall
import com.saunak.director.processing.InstantFinishEngine
import com.saunak.director.processing.TechnicalQuality
import com.saunak.director.tools.ToolIds
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class InstantFinishEngineTest {
    private val cleanQuality = TechnicalQuality(
        meanLuma = 120.0,
        lumaStdDev = 42.0,
        darkClipFraction = .01,
        brightClipFraction = .01,
        edgeEnergy = 8.0,
        catastrophic = false,
        heuristicScore = .88
    )

    @Test fun portraitGetsCleanPortraitFinishWhenModelOmitsLook() {
        val plan = PlanOutput("single person", "clean flattering portrait", "portrait", .85, false, emptyList())
        val result = InstantFinishEngine.enrich(plan, cleanQuality)
        assertEquals(ToolIds.LOOK_CLEAN_PORTRAIT, result.toolCalls.last().id)
    }

    @Test fun explicitCinematicLookIsPreserved() {
        val plan = PlanOutput(
            "street portrait",
            "cinematic frame",
            "portrait",
            .82,
            false,
            listOf(ToolCall(ToolIds.LOOK_CINEMATIC_COOL, .6))
        )
        val result = InstantFinishEngine.enrich(plan, cleanQuality)
        assertEquals(1, result.toolCalls.size)
        assertEquals(ToolIds.LOOK_CINEMATIC_COOL, result.toolCalls.first().id)
    }

    @Test fun clippedFrameReducesFinishStrength() {
        val clipped = cleanQuality.copy(darkClipFraction = .20)
        val plan = PlanOutput("city", "social ready", "general", .7, false, emptyList())
        val result = InstantFinishEngine.enrich(plan, clipped)
        val strength = result.toolCalls.last().value ?: 1.0
        assertTrue(strength < .50)
    }
}
