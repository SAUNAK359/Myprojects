package com.saunak.director.ai

import org.json.JSONArray
import org.json.JSONObject

/**
 * Normalizes OpenAI-compatible chat responses across NVIDIA NIM, OpenRouter, Groq,
 * Gemini's OpenAI facade, and custom providers.
 *
 * Providers are allowed to return message.content as a string, object, or content-part array.
 * Models may also prepend reasoning/prose/fences. This parser extracts only a balanced JSON
 * object that matches Director's expected semantic contract; camera safety is still enforced
 * later by VlmContractValidator + ToolValidator.
 */
object ProviderResponseParser {
    enum class Kind { PLAN, EVALUATION }

    fun messageText(raw: String, providerLabel: String): String {
        val root = runCatching { JSONObject(raw) }
            .getOrElse { error("$providerLabel returned unreadable API JSON") }
        val choices = root.optJSONArray("choices") ?: error("$providerLabel returned no choices")
        require(choices.length() > 0) { "$providerLabel returned no choices" }
        val choice = choices.optJSONObject(0) ?: error("$providerLabel returned an invalid choice")
        val message = choice.optJSONObject("message")

        // Some structured-output implementations expose a parsed object directly.
        if (message != null) {
            val parsed = message.opt("parsed")
            when (parsed) {
                is JSONObject -> return parsed.toString()
                is JSONArray -> if (parsed.length() > 0) return parsed.toString()
            }

            val content = message.opt("content")
            val text = contentToText(content)
            if (text.isNotBlank()) return text
        }

        // Legacy/OpenAI-compatible completion fallback.
        val choiceText = choice.optString("text").trim()
        if (choiceText.isNotBlank()) return choiceText
        error("$providerLabel returned empty output")
    }

    fun directorJson(text: String, kind: Kind): String {
        val trimmed = text.trim()

        // Fast path: response is already a JSON object.
        parseCandidate(trimmed, kind)?.let { return it.toString() }

        // Common markdown wrappers.
        val unfenced = trimmed
            .removePrefix("```json")
            .removePrefix("```JSON")
            .removePrefix("```")
            .removeSuffix("```")
            .trim()
        parseCandidate(unfenced, kind)?.let { return it.toString() }

        // Models can prepend reasoning or prose. Do not use first-{ / last-}; scan balanced
        // objects while respecting quoted strings and escapes, then select the object matching
        // the Director contract.
        for (candidate in balancedObjects(trimmed)) {
            parseCandidate(candidate, kind)?.let { return it.toString() }
        }

        throw IllegalArgumentException("AI provider returned non-JSON output")
    }

    private fun contentToText(content: Any?): String = when (content) {
        null, JSONObject.NULL -> ""
        is String -> content.trim()
        is JSONObject -> content.toString()
        is JSONArray -> buildString {
            for (i in 0 until content.length()) {
                when (val part = content.opt(i)) {
                    is String -> appendLine(part)
                    is JSONObject -> {
                        val type = part.optString("type")
                        val text = when {
                            part.has("text") -> part.optString("text")
                            part.has("content") -> part.optString("content")
                            type == "output_text" && part.has("value") -> part.optString("value")
                            else -> ""
                        }
                        if (text.isNotBlank()) appendLine(text)
                    }
                }
            }
        }.trim()
        else -> content.toString().trim()
    }

    private fun parseCandidate(candidate: String, kind: Kind): JSONObject? {
        val obj = runCatching { JSONObject(candidate) }.getOrNull() ?: return null
        findMatchingObject(obj, kind)?.let { return it }
        return null
    }

    private fun findMatchingObject(obj: JSONObject, kind: Kind, depth: Int = 0): JSONObject? {
        if (matches(obj, kind)) return obj
        if (depth >= 2) return null
        val wrapperKeys = arrayOf("result", "output", "response", "data", "analysis", "plan", "evaluation")
        for (key in wrapperKeys) {
            val child = obj.optJSONObject(key) ?: continue
            findMatchingObject(child, kind, depth + 1)?.let { return it }
        }
        return null
    }

    private fun matches(obj: JSONObject, kind: Kind): Boolean = when (kind) {
        Kind.PLAN -> obj.optString("scene").isNotBlank() && obj.optString("objective").isNotBlank()
        Kind.EVALUATION -> obj.has("score") && !obj.isNull("score")
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
