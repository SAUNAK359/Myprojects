package com.saunak.director.ai

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * Normalizes OpenAI-compatible chat responses across NVIDIA NIM, OpenRouter, Groq,
 * Gemini's OpenAI facade, and custom providers.
 *
 * Uses kotlinx.serialization rather than Android org.json so the exact same parser can be
 * exercised by local JVM tests and used on-device.
 */
object ProviderResponseParser {
    enum class Kind { PLAN, EVALUATION }

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    fun messageText(raw: String, providerLabel: String): String {
        val root = runCatching { json.parseToJsonElement(raw).jsonObject }
            .getOrElse { error("$providerLabel returned unreadable API JSON") }
        val choices = root["choices"] as? JsonArray ?: error("$providerLabel returned no choices")
        require(choices.isNotEmpty()) { "$providerLabel returned no choices" }
        val choice = choices.firstOrNull() as? JsonObject ?: error("$providerLabel returned an invalid choice")
        val message = choice["message"] as? JsonObject

        if (message != null) {
            val parsed = message["parsed"]
            when (parsed) {
                is JsonObject -> return parsed.toString()
                is JsonArray -> if (parsed.isNotEmpty()) return parsed.toString()
                else -> Unit
            }

            val content = message["content"]
            val text = contentToText(content)
            if (text.isNotBlank()) return text
        }

        val choiceText = (choice["text"] as? JsonPrimitive)?.contentOrNull?.trim().orEmpty()
        if (choiceText.isNotBlank()) return choiceText
        error("$providerLabel returned empty output")
    }

    fun directorJson(text: String, kind: Kind): String {
        val trimmed = text.trim()
        parseCandidate(trimmed, kind)?.let { return it.toString() }

        val unfenced = trimmed
            .removePrefix("```json")
            .removePrefix("```JSON")
            .removePrefix("```")
            .removeSuffix("```")
            .trim()
        parseCandidate(unfenced, kind)?.let { return it.toString() }

        for (candidate in balancedObjects(trimmed)) {
            parseCandidate(candidate, kind)?.let { return it.toString() }
        }

        throw IllegalArgumentException("AI provider returned non-JSON output")
    }

    private fun contentToText(content: JsonElement?): String = when (content) {
        null, JsonNull -> ""
        is JsonPrimitive -> content.contentOrNull?.trim().orEmpty()
        is JsonObject -> content.toString()
        is JsonArray -> buildString {
            for (part in content) {
                when (part) {
                    is JsonPrimitive -> part.contentOrNull?.takeIf { it.isNotBlank() }?.let { appendLine(it) }
                    is JsonObject -> {
                        val type = (part["type"] as? JsonPrimitive)?.contentOrNull.orEmpty()
                        val text = sequenceOf("text", "content")
                            .mapNotNull { key -> (part[key] as? JsonPrimitive)?.contentOrNull }
                            .firstOrNull { it.isNotBlank() }
                            ?: if (type == "output_text") (part["value"] as? JsonPrimitive)?.contentOrNull.orEmpty() else ""
                        if (text.isNotBlank()) appendLine(text)
                    }
                    else -> Unit
                }
            }
        }.trim()
    }

    private fun parseCandidate(candidate: String, kind: Kind): JsonObject? {
        val obj = runCatching { json.parseToJsonElement(candidate).jsonObject }.getOrNull() ?: return null
        return findMatchingObject(obj, kind)
    }

    private fun findMatchingObject(obj: JsonObject, kind: Kind, depth: Int = 0): JsonObject? {
        if (matches(obj, kind)) return obj
        if (depth >= 3) return null
        val wrapperKeys = arrayOf("result", "output", "response", "data", "analysis", "plan", "evaluation")
        for (key in wrapperKeys) {
            val child = obj[key] as? JsonObject ?: continue
            findMatchingObject(child, kind, depth + 1)?.let { return it }
        }
        return null
    }

    private fun matches(obj: JsonObject, kind: Kind): Boolean = when (kind) {
        Kind.PLAN -> {
            val scene = (obj["scene"] as? JsonPrimitive)?.contentOrNull.orEmpty()
            val objective = (obj["objective"] as? JsonPrimitive)?.contentOrNull.orEmpty()
            scene.isNotBlank() && objective.isNotBlank()
        }
        Kind.EVALUATION -> obj["score"] != null && obj["score"] !is JsonNull
    }

    internal fun balancedObjects(text: String): List<String> {
        val results = ArrayList<String>(3)
        var start = -1
        var depth = 0
        var inString = false
        var escaped = false

        for (i in text.indices) {
            val ch = text[i]
            if (inString) {
                if (escaped) {
                    escaped = false
                } else when (ch) {
                    '\\' -> escaped = true
                    '"' -> inString = false
                }
                continue
            }

            when (ch) {
                '"' -> if (depth > 0) inString = true
                '{' -> {
                    if (depth == 0) start = i
                    depth++
                }
                '}' -> if (depth > 0) {
                    depth--
                    if (depth == 0 && start >= 0) {
                        results += text.substring(start, i + 1)
                        start = -1
                    }
                }
            }
        }
        return results
    }
}
