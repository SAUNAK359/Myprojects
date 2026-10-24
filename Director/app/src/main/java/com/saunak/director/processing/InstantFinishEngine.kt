package com.saunak.director.processing

import com.saunak.director.director.PlanOutput
import com.saunak.director.director.ToolCall
import com.saunak.director.engine.SierraEngine
import com.saunak.director.engine.subject.PredictiveSubjectPolicy
import com.saunak.director.tools.ToolIds

/** Local finishing intelligence for the post-ready straight-from-camera path. */
object InstantFinishEngine {
    fun enrich(plan: PlanOutput, quality: TechnicalQuality): PlanOutput {
        val framedPlan = applyPredictiveFraming(plan)
        val calls = framedPlan.toolCalls.toMutableList()
        val existingLookIndex = calls.indexOfLast { it.id in ToolIds.LOOK_IDS }

        if (existingLookIndex >= 0) {
            val current = calls[existingLookIndex]
            val base = current.value ?: defaultStrength(current.id)
            calls[existingLookIndex] = current.copy(value = adaptStrength(base, quality))
            return framedPlan.copy(toolCalls = calls)
        }

        val branch = framedPlan.skillBranch.lowercase()
        val intent = "${framedPlan.scene} ${framedPlan.objective}".lowercase()
        val lookId = when {
            "natural" in intent || "documentary" in intent || "true to life" in intent -> ToolIds.LOOK_NATURAL
            "neon" in intent || "nightlife" in intent || (branch == "low_light" && ("city" in intent || "street" in intent)) -> ToolIds.LOOK_NIGHT_NEON
            "golden hour" in intent || "sunset" in intent || "sunrise" in intent -> ToolIds.LOOK_GOLDEN_HOUR
            "moody" in intent || "dramatic" in intent || "dark cinematic" in intent -> ToolIds.LOOK_MOODY
            "film" in intent || "teal" in intent || "orange" in intent -> ToolIds.LOOK_FILM_TEAL_ORANGE
            "cinematic" in intent && ("cool" in intent || "blue" in intent) -> ToolIds.LOOK_CINEMATIC_COOL
            "cinematic" in intent -> ToolIds.LOOK_CINEMATIC_WARM
            branch == "portrait" || branch == "selfie" || branch == "group" -> ToolIds.LOOK_CLEAN_PORTRAIT
            branch == "landscape" -> ToolIds.LOOK_LANDSCAPE_VIVID
            branch == "architecture" -> ToolIds.LOOK_NATURAL
            branch == "food" -> ToolIds.LOOK_WARM
            branch == "low_light" -> ToolIds.LOOK_CINEMATIC_COOL
            else -> ToolIds.LOOK_NATURAL
        }

        if (lookId != ToolIds.LOOK_NATURAL) {
            calls += ToolCall(id = lookId, value = adaptStrength(defaultStrength(lookId), quality))
        }
        return framedPlan.copy(toolCalls = calls.take(8))
    }

    private fun applyPredictiveFraming(plan: PlanOutput): PlanOutput {
        val branch = plan.skillBranch.lowercase()
        // A single primary-subject lead-room crop is inappropriate for group scenes.
        if (branch != "portrait" && branch != "selfie") return plan
        val track = SierraEngine.getOrNull()?.subjects?.latest() ?: return plan
        val directive = PredictiveSubjectPolicy.evaluate(track)
        if (directive.confidence < FRAMING_MIN_CONFIDENCE) return plan

        val influence = (0.26 + 0.34 * directive.confidence + 0.24 * directive.edgeRisk)
            .coerceIn(0.0, MAX_FRAMING_INFLUENCE)
        val target = directive.cropCenter
        val x = (plan.cropCenterX * (1.0 - influence) + target.x * influence).coerceIn(0.08, 0.92)
        val y = (plan.cropCenterY * (1.0 - influence) + target.y * influence).coerceIn(0.08, 0.92)
        return plan.copy(cropCenterX = x, cropCenterY = y)
    }

    private fun defaultStrength(id: Int): Double = when (id) {
        ToolIds.LOOK_SOCIAL_POP -> .34
        ToolIds.LOOK_FILM_TEAL_ORANGE -> .48
        ToolIds.LOOK_GOLDEN_HOUR -> .46
        ToolIds.LOOK_MOODY -> .42
        ToolIds.LOOK_CLEAN_PORTRAIT -> .38
        ToolIds.LOOK_NIGHT_NEON -> .46
        ToolIds.LOOK_CINEMATIC_WARM, ToolIds.LOOK_CINEMATIC_COOL -> .50
        ToolIds.LOOK_SOFT_PORTRAIT -> .44
        ToolIds.LOOK_LANDSCAPE_VIVID -> .38
        ToolIds.LOOK_WARM, ToolIds.LOOK_COOL -> .42
        ToolIds.LOOK_MONOCHROME -> 1.0
        else -> 0.0
    }

    private fun adaptStrength(base: Double, quality: TechnicalQuality): Double {
        var strength = base
        if (quality.darkClipFraction > .12 || quality.brightClipFraction > .08) strength *= .78
        val brightSky = quality.highlightP95 >= 210.0 && quality.topMeanLuma - quality.bottomMeanLuma >= 28.0
        if (brightSky) strength *= .82
        if (quality.lumaStdDev < 24.0 && quality.edgeEnergy > 2.0) strength *= 1.05
        if (quality.edgeEnergy < 1.2) strength *= .88
        return strength.coerceIn(0.12, 0.70)
    }

    private const val FRAMING_MIN_CONFIDENCE = 0.52
    private const val MAX_FRAMING_INFLUENCE = 0.72
}
