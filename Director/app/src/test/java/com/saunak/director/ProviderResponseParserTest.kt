package com.saunak.director

import com.saunak.director.ai.ProviderResponseParser
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test

class ProviderResponseParserTest {
    private val plan = """{"scene":"portrait","objective":"clean cinematic portrait","branch":"portrait","reference_score":0.8,"reference_accept":false,"calls":[],"recapture":true}"""

    @Test fun extractsFencedJson() {
        val parsed = ProviderResponseParser.directorJson("```json\n$plan\n```", ProviderResponseParser.Kind.PLAN)
        assertEquals("portrait", JSONObject(parsed).getString("scene"))
    }

    @Test fun skipsReasoningObjectAndFindsDirectorPlan() {
        val text = "Reasoning metadata {\"step\":1}. Final answer: $plan"
        val parsed = ProviderResponseParser.directorJson(text, ProviderResponseParser.Kind.PLAN)
        assertEquals("clean cinematic portrait", JSONObject(parsed).getString("objective"))
    }

    @Test fun unwrapsNestedPlanObject() {
        val wrapped = """{"result":{"plan":$plan}}"""
        val parsed = ProviderResponseParser.directorJson(wrapped, ProviderResponseParser.Kind.PLAN)
        assertEquals("portrait", JSONObject(parsed).getString("branch"))
    }

    @Test fun extractsTextFromContentArray() {
        val raw = JSONObject()
            .put("choices", org.json.JSONArray().put(
                JSONObject().put("message", JSONObject().put(
                    "content", org.json.JSONArray().put(JSONObject().put("type", "text").put("text", plan))
                ))
            ))
            .toString()
        assertEquals(plan, ProviderResponseParser.messageText(raw, "test"))
    }
}
