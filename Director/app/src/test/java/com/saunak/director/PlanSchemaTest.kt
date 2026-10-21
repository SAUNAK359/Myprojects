package com.saunak.director

import com.saunak.director.director.PlanOutput
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PlanSchemaTest {
    @Test fun parsesShutterPlan() {
        val plan = Json.decodeFromString(PlanOutput.serializer(), """{"scene":"portrait","objective":"natural face exposure","branch":"portrait","reference_score":0.81,"reference_accept":false,"calls":[{"id":3,"v":-0.3,"x":null,"y":null,"s":null}],"recapture":true}""")
        assertEquals("portrait", plan.scene)
        assertEquals(3, plan.toolCalls.single().id)
        assertTrue(plan.recapture)
        assertEquals(0.81, plan.referenceScore, 0.0001)
    }
}
