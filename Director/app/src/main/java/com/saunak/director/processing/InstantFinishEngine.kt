package com.saunak.director.processing

import com.saunak.director.director.PlanOutput
import com.saunak.director.engine.SierraEngine
import com.saunak.director.engine.subject.PredictiveSubjectPolicy

/**
 * Fast pre-capture local enrichment only. Final rendering for both AI and beta is now owned by
 * SierraRenderSpec + SierraRenderExecutor, so this layer must not inject legacy named look tools.
 */
object InstantFinishEngine {
    fun enrich(plan: PlanOutput, quality: TechnicalQuality): PlanOutput {
        if (plan.planner.startsWith("sierra_ai")) return plan
        return applyPredictiveFraming(plan)
    }

    private fun applyPredictiveFraming(plan: PlanOutput): PlanOutput {
        val branch = plan.skillBranch.lowercase()
        if (branch != "portrait" && branch != "selfie") return plan
        val track = SierraEngine.getOrNull()?.subjects?.latest() ?: return plan
        val directive = PredictiveSubjectPolicy.evaluate(track)
        if (directive.confidence < FRAMING_MIN_CONFIDENCE) return plan
        val influence = (0.26 + 0.34 * directive.confidence + 0.24 * directive.edgeRisk).coerceIn(0.0, MAX_FRAMING_INFLUENCE)
        val target = directive.cropCenter
        val x = (plan.cropCenterX * (1.0 - influence) + target.x * influence).coerceIn(0.08, 0.92)
        val y = (plan.cropCenterY * (1.0 - influence) + target.y * influence).coerceIn(0.08, 0.92)
        return plan.copy(cropCenterX = x, cropCenterY = y)
    }

    private const val FRAMING_MIN_CONFIDENCE = 0.52
    private const val MAX_FRAMING_INFLUENCE = 0.72
}
