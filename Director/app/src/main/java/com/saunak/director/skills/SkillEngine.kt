package com.saunak.director.skills

import android.content.Context
import org.json.JSONObject

class SkillEngine(context: Context) {
    private val rootJson = JSONObject(context.assets.open("skills.json").bufferedReader().use { it.readText() })
    init { require(rootJson.optInt("version") >= 2) { "Director skill IR v2 required" } }

    fun planningContext(): String {
        val routing = rootJson.getJSONArray("routing")
        val routes = buildString {
            for (i in 0 until routing.length()) {
                val item = routing.getJSONObject(i)
                if (isNotEmpty()) append(" | ")
                append(item.getString("id")).append(": ").append(item.getString("summary"))
            }
        }
        return "Root: ${rootJson.getString("root")} Routing: $routes Tool policy: ${rootJson.optString("tools")}".trim()
    }

    fun resolveBranch(requested: String?): String {
        val key = requested?.trim()?.lowercase().orEmpty()
        return if (rootJson.getJSONObject("branches").has(key)) key else "general"
    }

    fun branch(name: String): String = rootJson.getJSONObject("branches").getString(resolveBranch(name))
    fun evaluationContext(branch: String): String = "Scene branch: ${branch(branch)} Evaluation: ${rootJson.optString("evaluation")}".trim()
    fun availableBranches(): Set<String> = rootJson.getJSONObject("branches").keys().asSequence().toSet()
}
