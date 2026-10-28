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

    /**
     * Small recency context for Gemini. This is explicitly NOT a novelty scorer: repeated treatment
     * is allowed when it is still the strongest photographic choice. It only prevents repetition by habit.
     */
    fun recentCreativeContext(limit: Int = 6): String {
        val items = load().sortedByDescending { it.createdAtMs }.take(limit.coerceIn(1, 10))
        if (items.isEmpty()) return "RECENT PHOTO MEMORY: none."
        return buildString {
            appendLine("RECENT PHOTO MEMORY (context only; never force novelty):")
            items.forEachIndexed { index, recipe ->
                val p = recipe.plan
                val d = p.photographicDirection
                val box = p.directorSpec?.primaryEntity?.targetBox
                append(index + 1).append(". scene=").append(recipe.scene.take(42))
                append("; objective=").append(recipe.objective.take(72))
                append("; ratio=").append(p.aspectRatio)
                append("; crop=").append(String.format(Locale.US, "%.2f,%.2f", p.cropCenterX, p.cropCenterY))
                if (box != null) append("; entity_box=").append(String.format(Locale.US, "%.2f,%.2f,%.2f,%.2f", box.left, box.top, box.right, box.bottom))
                if (d.selectedThemeId.isNotBlank()) append("; theme=").append(d.selectedThemeId.take(40))
                if (d.mood.isNotBlank()) append("; mood=").append(d.mood.take(52))
                p.renderSpec?.intentSummary?.takeIf(String::isNotBlank)?.let { append("; render=").append(it.take(72)) }
                appendLine()
            }
            append("Do not repeat these merely from habit. If the current scene genuinely benefits from the same treatment, use it again without penalty.")
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
