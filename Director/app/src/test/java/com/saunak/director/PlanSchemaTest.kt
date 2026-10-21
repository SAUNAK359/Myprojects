package com.saunak.director

import com.saunak.director.director.PlanOutput
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Test

class PlanSchemaTest {
    @Test fun parsesCompactPlan() {
        val plan = Json.decodeFromString(PlanOutput.serializer(), """{"scene":"portrait","objective":"natural","branch":"portrait","calls":[{"id":3,"v":-1}],"capture":true}""")
        assertEquals("portrait", plan.scene)
        assertEquals(3, plan.toolCalls.single().id)
    }
}
