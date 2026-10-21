package com.saunak.director.skills

import android.content.Context
import org.json.JSONObject

class SkillEngine(context: Context) {
    private val raw = context.assets.open("skills.json").bufferedReader().use { it.readText() }
    private val root = JSONObject(raw)
    fun rootContext(): String = root.getString("root")
    fun branch(name: String): String { val branches = root.getJSONObject("branches"); return if (branches.has(name)) branches.getString(name) else branches.getString("general") }
    fun planningContext(): String {
        val branches = root.getJSONObject("branches")
        val compact = buildString { branches.keys().asSequence().toList().sorted().forEach { key -> append("[").append(key).append("] ").append(branches.getString(key)).append(' ') } }
        return "${rootContext()} Scene branches: $compact Tool policy: ${root.optString("tools")}".trim()
    }
    fun evaluationContext(): String = root.optString("evaluation")
    fun combined(branch: String): String = "${rootContext()} ${branch(branch)}"
}
