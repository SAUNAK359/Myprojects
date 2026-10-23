package com.saunak.director

import com.saunak.director.ai.ProviderResponseParser
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Test

class ProviderResponseParserTest {
    private val json = Json { ignoreUnknownKeys = true }
    private val plan = """{"scene":"portrait","objective":"clean cinematic portrait","branch":"portrait","reference_score":0.8,"reference_accept":false,"calls":[],"recapture":true}"""

    @Test fun extractsFencedJson() {
        val parsed = ProviderResponseParser.directorJson("```json\n$plan\n```", ProviderResponseParser.Kind.PLAN)
        assertEquals("portrait", json.parseToJsonElement(parsed).jsonObject["scene"]!!.jsonPrimitive.content)
    }

    @Test fun skipsReasoningObjectAndFindsDirectorPlan() {
        val text = "Reasoning metadata {\"step\":1}. Final answer: $plan"
        val parsed = ProviderResponseParser.directorJson(text, ProviderResponseParser.Kind.PLAN)
        assertEquals("clean cinematic portrait", json.parseToJsonElement(parsed).jsonObject["objective"]!!.jsonPrimitive.content)
    }

    @Test fun unwrapsNestedPlanObject() {
        val wrapped = """{"result":{"plan":$plan}}"""
        val parsed = ProviderResponseParser.directorJson(wrapped, ProviderResponseParser.Kind.PLAN)
        assertEquals("portrait", json.parseToJsonElement(parsed).jsonObject["branch"]!!.jsonPrimitive.content)
    }

    @Test fun extractsTextFromContentArray() {
        val escaped = plan.replace("\\", "\\\\").replace("\"", "\\\"")
        val raw = """{"choices":[{"message":{"content":[{"type":"text","text":"$escaped"}]}}]}"""
        assertEquals(plan, ProviderResponseParser.messageText(raw, "test"))
    }
}
