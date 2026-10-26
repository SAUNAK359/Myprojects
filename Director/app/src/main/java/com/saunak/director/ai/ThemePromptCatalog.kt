package com.saunak.director.ai

import android.content.Context
import org.json.JSONObject

/** Reads the same asset-driven theme catalog used by Sierra cognition; no duplicated theme list. */
class ThemePromptCatalog(context: Context) {
    private val ids: Set<String>
    private val compact: String

    init {
        val root = JSONObject(context.applicationContext.assets.open("sierra/cognition_v1.json").bufferedReader().use { it.readText() })
        val themes = root.getJSONArray("themes")
        val collected = LinkedHashSet<String>()
        compact = buildString {
            for (i in 0 until themes.length()) {
                val theme = themes.getJSONObject(i)
                val id = theme.getString("id")
                collected += id
                if (isNotEmpty()) append(" | ")
                append(id)
                    .append("=")
                    .append(theme.getString("display_name"))
                    .append(";look=")
                    .append(theme.getString("fallback_look"))
                    .append(";human=")
                    .append(String.format(java.util.Locale.US, "%.2f", theme.optDouble("human_affinity", 0.5)))
                    .append(";lowlight=")
                    .append(String.format(java.util.Locale.US, "%.2f", theme.optDouble("low_light_affinity", 0.5)))
                    .append(";atmosphere=")
                    .append(String.format(java.util.Locale.US, "%.2f", theme.optDouble("atmosphere_priority", 0.5)))
                    .append(";avoid=")
                val negative = theme.optJSONArray("negative_constraints")
                if (negative != null) for (j in 0 until negative.length()) {
                    if (j > 0) append(',')
                    append(negative.optString(j))
                }
            }
        }
        ids = collected.toSet()
    }

    fun compactDescription(): String = compact
    fun contains(themeId: String): Boolean = themeId in ids
    fun requireInstalled(themeId: String) {
        require(themeId.isBlank() || contains(themeId)) { "Gemini selected unknown Sierra theme: $themeId" }
    }
}
