package com.saunak.director.processing

import com.saunak.director.director.PlanOutput
import com.saunak.director.director.ToolCall
import com.saunak.director.tools.ToolIds

/**
 * Local finishing intelligence for the "post-ready straight from camera" path.
 * It never mutates physical camera controls. The VLM remains responsible for semantic scene
 * understanding; this layer guarantees a tasteful finish without another network round-trip.
 */
object InstantFinishEngine {
    fun enrich(plan: PlanOutput, quality: TechnicalQuality): PlanOutput {
        val calls = plan.toolCalls.toMutableList()
        val existingLookIndex = calls.indexOfLast { it.id in ToolIds.LOOK_IDS }

        if (existingLookIndex >= 0) {
            val current = calls[existingLookIndex]
            val base = current.value ?: defaultStrength(current.id)
            calls[existingLookIndex] = current.copy(value = adaptStrength(base, quality))
            return plan.copy(toolCalls = calls)
        }

        val branch = plan.skillBranch.lowercase()
        val intent = "${plan.scene} ${plan.objective}".lowercase()
        val lookId = when {
            "natural" in intent || "documentary" in intent || "true to life" in intent -> ToolIds.LOOK_NATURAL
            "neon" in intent || "nightlife" in intent || (branch == "low_light" && ("city" in intent || "street" in intent)) -> ToolIds.LOOK_NIGHT_NEON
            "golden hour" in intent || "sunset" in intent || "sunrise" in intent -> ToolIds.LOOK_GOLDEN_HOUR
            "moody" in intent || "dramatic" in intent || "dark cinematic" in intent -> ToolIds.LOOK_MOODY
            "film" in intent || "teal" in intent || "orange" in intent -> ToolIds.LOOK_FILM_TEAL_ORANGE
            "cinematic" in intent && ("cool" in intent || "blue" in intent) -> ToolIds.LOOK_CINEMATIC_COOL
            "cinematic" in intent -> ToolIds.LOOK_CINEMATIC_WARM
            branch == "portrait" || branch == "selfie" || branch == "group" -> ToolIds.LOOK_CLEAN_PORTRAIT
            branch == "landscape" || branch == "architecture" -> ToolIds.LOOK_LANDSCAPE_VIVID
            branch == "food" -> ToolIds.LOOK_WARM
            branch == "low_light" -> ToolIds.LOOK_CINEMATIC_COOL
            else -> ToolIds.LOOK_SOCIAL_POP
        }

        if (lookId != ToolIds.LOOK_NATURAL) {
            calls += ToolCall(id = lookId, value = adaptStrength(defaultStrength(lookId), quality))
        }
        return plan.copy(toolCalls = calls.take(8))
    }

    private fun defaultStrength(id: Int): Double = when (id) {
        ToolIds.LOOK_SOCIAL_POP -> .50
        ToolIds.LOOK_FILM_TEAL_ORANGE -> .56
        ToolIds.LOOK_GOLDEN_HOUR -> .54
        ToolIds.LOOK_MOODY -> .48
        ToolIds.LOOK_CLEAN_PORTRAIT -> .44
        ToolIds.LOOK_NIGHT_NEON -> .52
        ToolIds.LOOK_CINEMATIC_WARM, ToolIds.LOOK_CINEMATIC_COOL -> .62
        ToolIds.LOOK_SOFT_PORTRAIT -> .55
        ToolIds.LOOK_LANDSCAPE_VIVID -> .52
        ToolIds.LOOK_WARM, ToolIds.LOOK_COOL -> .55
        ToolIds.LOOK_MONOCHROME -> 1.0
        else -> 0.0
    }

    private fun adaptStrength(base: Double, quality: TechnicalQuality): Double {
        var strength = base
        // Avoid crushing already-clipped shadows/highlights. Stronger finish is reserved for
        // technically clean frames where it cannot hide detail.
        if (quality.darkClipFraction > .12 || quality.brightClipFraction > .08) strength *= .78
        if (quality.lumaStdDev < 24.0 && quality.edgeEnergy > 2.0) strength *= 1.08
        if (quality.edgeEnergy < 1.2) strength *= .88
        return strength.coerceIn(0.18, 0.78)
    }
}
