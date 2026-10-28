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
        heuristicScore = .88,
        shadowP05 = 45.0,
        medianLuma = 120.0,
        highlightP95 = 205.0,
        topMeanLuma = 125.0,
        bottomMeanLuma = 115.0,
        horizontalEdgeEnergy = 8.0,
        verticalEdgeEnergy = 8.0,
        strongEdgeFraction = .12
    )

    @Test fun betaEnrichmentDoesNotInjectLegacyNamedLooks() {
        val plan = PlanOutput("single person", "clean flattering portrait", "portrait", .85, false, emptyList())
        val result = InstantFinishEngine.enrich(plan, cleanQuality)
        assertTrue(result.toolCalls.none { it.id in ToolIds.LOOK_IDS })
    }

    @Test fun preexistingToolCallsAreNotReinterpretedByFinisher() {
        val plan = PlanOutput("street portrait", "cinematic frame", "portrait", .82, false, listOf(ToolCall(ToolIds.LOOK_CINEMATIC_COOL, .6)))
        val result = InstantFinishEngine.enrich(plan, cleanQuality)
        assertEquals(plan.toolCalls, result.toolCalls)
    }

    @Test fun generalSceneDoesNotGetArtificialSocialPopByDefault() {
        val plan = PlanOutput("everyday scene", "natural rendering", "general", .8, false, emptyList())
        val result = InstantFinishEngine.enrich(plan, cleanQuality)
        assertTrue(result.toolCalls.none { it.id == ToolIds.LOOK_SOCIAL_POP })
    }

    @Test fun brightSkyStillDoesNotInjectLegacyLandscapeLook() {
        val brightSky = cleanQuality.copy(highlightP95 = 235.0, topMeanLuma = 205.0, bottomMeanLuma = 130.0)
        val plan = PlanOutput("city skyline", "protect sky", "landscape", .8, false, emptyList())
        val result = InstantFinishEngine.enrich(plan, brightSky)
        assertTrue(result.toolCalls.none { it.id in ToolIds.LOOK_IDS })
    }

    @Test fun aiPlanIsPassThrough() {
        val plan = PlanOutput(
            scene = "portrait",
            objective = "authoritative ai plan",
            skillBranch = "portrait",
            toolCalls = listOf(ToolCall(ToolIds.EXPOSURE_EV, -.33)),
            planner = "sierra_ai"
        )
        assertEquals(plan, InstantFinishEngine.enrich(plan, cleanQuality))
    }
}
