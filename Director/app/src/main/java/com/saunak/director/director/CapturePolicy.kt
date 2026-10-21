package com.saunak.director.director

import com.saunak.director.director.EvaluationOutput

object CapturePolicy {
    const val CONFIGURED_MAX_PHYSICAL_CAPTURES = 3
    const val MAX_EVALUATIONS = 3
    const val MIN_ACCEPT_SCORE = 0.78
    const val MIN_TECHNICAL_SCORE = 0.65

    fun physicalBudget(thermalBudget: Int): Int = thermalBudget.coerceIn(1, CONFIGURED_MAX_PHYSICAL_CAPTURES)
    fun accepted(eval: EvaluationOutput, catastrophic: Boolean): Boolean = !catastrophic && eval.accept && eval.score >= MIN_ACCEPT_SCORE && eval.technical >= MIN_TECHNICAL_SCORE
    fun combinedFallbackScore(vlmScore: Double, technicalHeuristic: Double): Double = (0.85 * vlmScore.coerceIn(0.0, 1.0) + 0.15 * technicalHeuristic.coerceIn(0.0, 1.0)).coerceIn(0.0, 1.0)
}
