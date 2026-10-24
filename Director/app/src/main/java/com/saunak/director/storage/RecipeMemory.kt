package com.saunak.director.storage

import android.content.Context
import com.saunak.director.director.PlanOutput
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.util.Locale

@Serializable
data class SavedRecipe(
    val scene: String,
    val objective: String,
    val skillBranch: String,
    val score: Double,
    val plan: PlanOutput,
    val createdAtMs: Long = System.currentTimeMillis()
)

class RecipeMemory(context: Context) {
    private val file = File(context.filesDir, "director/recipes.json")
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

    @Synchronized
    fun remember(plan: PlanOutput, score: Double) {
        val current = load().toMutableList()
        current.removeAll {
            it.skillBranch == plan.skillBranch &&
                it.scene.equals(plan.scene, true) &&
                it.objective.equals(plan.objective, true)
        }
        current.add(0, SavedRecipe(plan.scene, plan.objective, plan.skillBranch, score.coerceIn(0.0, 1.0), plan))
        save(current.sortedByDescending { it.score }.take(MAX_RECIPES))
    }

    fun promptHints(limit: Int = 4): String {
        val items = load().sortedByDescending { it.score }.take(limit)
        if (items.isEmpty()) return "No approved cached recipes yet."
        return buildString {
            appendLine("Previously successful Director recipes. Reuse only when relevant and adapt to the current scene:")
            items.forEachIndexed { index, recipe ->
                appendLine(
                    "${index + 1}. branch=${recipe.skillBranch}; scene=${recipe.scene}; score=${String.format(Locale.US, "%.3f", recipe.score)}; objective=${recipe.objective}; calls=${recipe.plan.toolCalls}"
                )
            }
        }
    }

    fun count(): Int = load().size

    @Synchronized
    private fun load(): List<SavedRecipe> = runCatching {
        if (!file.isFile) emptyList() else json.decodeFromString<List<SavedRecipe>>(file.readText())
    }.getOrDefault(emptyList())

    @Synchronized
    private fun save(values: List<SavedRecipe>) {
        file.parentFile?.mkdirs()
        runCatching { file.writeText(json.encodeToString(values)) }
    }

    companion object { private const val MAX_RECIPES = 60 }
}
