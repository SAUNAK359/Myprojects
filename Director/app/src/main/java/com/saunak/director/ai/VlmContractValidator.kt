package com.saunak.director.ai

import com.saunak.director.director.EvaluationOutput
import com.saunak.director.director.PlanOutput

object VlmContractValidator {
    fun plan(value: PlanOutput): PlanOutput {
        val scene = clean(value.scene)
        val objective = clean(value.objective)
        val branch = clean(value.skillBranch)
        require(scene.isNotBlank()) { "Invalid scene" }
        require(objective.isNotBlank()) { "Invalid objective" }
        require(branch.isNotBlank()) { "Invalid branch" }
        require(value.referenceScore.isFinite()) { "Invalid reference score" }

        // Provider models sometimes exceed our presentation limits by a few characters or
        // return more suggestions than Director can use. Normalize those harmless differences
        // here; ToolValidator remains the hard safety boundary for every camera operation.
        return value.copy(
            scene = scene.take(64),
            objective = objective.take(180),
            skillBranch = branch.take(48),
            referenceScore = value.referenceScore.coerceIn(0.0, 1.0),
            toolCalls = value.toolCalls.take(8)
        )
    }

    fun evaluation(value: EvaluationOutput): EvaluationOutput {
        val scores = listOf(value.score, value.technical, value.composition, value.subject, value.aesthetic)
        require(scores.all { it.isFinite() }) { "Invalid evaluation score" }
        return value.copy(
            score = value.score.coerceIn(0.0, 1.0),
            technical = value.technical.coerceIn(0.0, 1.0),
            composition = value.composition.coerceIn(0.0, 1.0),
            subject = value.subject.coerceIn(0.0, 1.0),
            aesthetic = value.aesthetic.coerceIn(0.0, 1.0),
            issues = value.issues.take(8).map { clean(it).take(96) },
            deltaCalls = value.deltaCalls.take(8)
        )
    }

    private fun clean(value: String): String = value.filter { it >= ' ' && it != '\u007f' }.trim()
}
