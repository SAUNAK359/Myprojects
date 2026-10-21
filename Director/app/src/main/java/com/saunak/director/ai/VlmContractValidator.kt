package com.saunak.director.ai

import com.saunak.director.director.EvaluationOutput
import com.saunak.director.director.PlanOutput

object VlmContractValidator {
    fun plan(value: PlanOutput): PlanOutput {
        require(value.scene.isNotBlank() && value.scene.length <= 64) { "Invalid scene" }
        require(value.objective.isNotBlank() && value.objective.length <= 180) { "Invalid objective" }
        require(value.skillBranch.isNotBlank() && value.skillBranch.length <= 48) { "Invalid branch" }
        require(value.referenceScore.isFinite() && value.referenceScore in 0.0..1.0) { "Invalid reference score" }
        require(value.toolCalls.size <= 8) { "Too many tool calls" }
        return value.copy(scene = clean(value.scene), objective = clean(value.objective), skillBranch = clean(value.skillBranch))
    }

    fun evaluation(value: EvaluationOutput): EvaluationOutput {
        val scores = listOf(value.score, value.technical, value.composition, value.subject, value.aesthetic)
        require(scores.all { it.isFinite() && it in 0.0..1.0 }) { "Invalid evaluation score" }
        require(value.issues.size <= 8 && value.issues.all { it.length <= 96 }) { "Invalid issue list" }
        require(value.deltaCalls.size <= 8) { "Too many delta calls" }
        return value.copy(issues = value.issues.map(::clean))
    }

    private fun clean(value: String): String = value.filter { it >= ' ' && it != '\u007f' }.trim()
}
