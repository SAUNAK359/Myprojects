package com.saunak.director.ai

import com.saunak.director.director.EvaluationOutput
import com.saunak.director.director.FinalRenderAdjustment
import com.saunak.director.director.PhotographicDirection
import com.saunak.director.director.PlanOutput
import com.saunak.director.director.ThemeBlendComponent
import com.saunak.director.tools.ToolIds

object VlmContractValidator {
    fun plan(value: PlanOutput): PlanOutput {
        val scene = clean(value.scene)
        val objective = clean(value.objective)
        val branch = clean(value.skillBranch)
        require(scene.isNotBlank()) { "Invalid scene" }
        require(objective.isNotBlank()) { "Invalid objective" }
        require(branch.isNotBlank()) { "Invalid branch" }
        require(value.referenceScore.isFinite()) { "Invalid reference score" }

        val calls = value.toolCalls.take(8)
        val cameraCalls = calls.filter { it.id in ToolIds.CAMERA_IDS }
        require(cameraCalls.map { it.id }.distinct().size == cameraCalls.size) {
            "Sierra plan is ambiguous: duplicate camera control"
        }
        require(calls.count { it.id in ToolIds.LOOK_IDS } <= 1) {
            "Sierra plan is ambiguous: multiple final looks"
        }
        val extension = calls.firstOrNull { it.id == ToolIds.EXTENSION }
        if (extension != null && extension.stringValue?.lowercase() != "none") {
            require(cameraCalls.none { it.id != ToolIds.EXTENSION }) {
                "Sierra plan is ambiguous: OEM extension and direct camera controls requested together"
            }
        }
        val flashOn = calls.any { it.id == ToolIds.FLASH && it.stringValue?.lowercase() != "off" }
        val torchOn = calls.any { it.id == ToolIds.TORCH && (it.value ?: 0.0) >= 0.5 }
        require(!(flashOn && torchOn)) { "Sierra plan is ambiguous: flash and torch requested together" }

        return value.copy(
            scene = scene.take(64),
            objective = objective.take(180),
            skillBranch = branch.take(48),
            referenceScore = value.referenceScore.coerceIn(0.0, 1.0),
            toolCalls = calls,
            cropCenterX = value.cropCenterX.coerceIn(0.0, 1.0),
            cropCenterY = value.cropCenterY.coerceIn(0.0, 1.0),
            photographicDirection = sanitizeDirection(value.photographicDirection)
        )
    }

    fun evaluation(value: EvaluationOutput): EvaluationOutput {
        val scores = listOf(value.score, value.technical, value.composition, value.subject, value.aesthetic, value.intentMatch, value.naturalness)
        require(scores.all { it.isFinite() }) { "Invalid evaluation score" }
        return value.copy(
            score = value.score.coerceIn(0.0, 1.0),
            technical = value.technical.coerceIn(0.0, 1.0),
            composition = value.composition.coerceIn(0.0, 1.0),
            subject = value.subject.coerceIn(0.0, 1.0),
            aesthetic = value.aesthetic.coerceIn(0.0, 1.0),
            intentMatch = value.intentMatch.coerceIn(0.0, 1.0),
            naturalness = value.naturalness.coerceIn(0.0, 1.0),
            issues = value.issues.take(8).map { clean(it).take(96) },
            deltaCalls = value.deltaCalls.take(4),
            renderAdjustment = sanitizeAdjustment(value.renderAdjustment)
        )
    }

    private fun sanitizeDirection(v: PhotographicDirection): PhotographicDirection {
        val blend = v.themeBlend.take(3).mapNotNull { component ->
            val id = clean(component.themeId).take(64)
            if (id.isBlank() || !component.weight.isFinite()) null
            else ThemeBlendComponent(id, component.weight.coerceIn(0.0, 1.0))
        }
        return v.copy(
            story = clean(v.story).take(160),
            mood = clean(v.mood).take(96),
            primarySubject = clean(v.primarySubject).take(96),
            narrativeRole = clean(v.narrativeRole).take(48),
            selectedThemeId = clean(v.selectedThemeId).take(64),
            themeStrength = finite(v.themeStrength).coerceIn(0.0, 1.0),
            themeConfidence = finite(v.themeConfidence).coerceIn(0.0, 1.0),
            themeBlend = blend,
            atmospherePriority = finite(v.atmospherePriority).coerceIn(0.0, 1.0),
            skinNaturalness = finite(v.skinNaturalness).coerceIn(0.0, 1.0),
            highlightProtection = finite(v.highlightProtection).coerceIn(0.0, 1.0),
            shadowDepth = finite(v.shadowDepth).coerceIn(0.0, 1.0),
            midtoneContrast = finite(v.midtoneContrast).coerceIn(0.0, 1.0),
            localContrast = finite(v.localContrast).coerceIn(0.0, 1.0),
            colorIntensity = finite(v.colorIntensity).coerceIn(0.0, 1.0),
            warmthBias = finite(v.warmthBias).coerceIn(-0.25, 0.25),
            subjectRelightEv = finite(v.subjectRelightEv).coerceIn(-0.4, 0.8),
            backgroundSuppressionEv = finite(v.backgroundSuppressionEv).coerceIn(-0.6, 0.1),
            grain = finite(v.grain).coerceIn(0.0, 0.20),
            vignette = finite(v.vignette).coerceIn(0.0, 0.35),
            reasonCodes = v.reasonCodes.take(8).map { clean(it).take(64) }
        )
    }

    private fun sanitizeAdjustment(v: FinalRenderAdjustment): FinalRenderAdjustment = v.copy(
        themeStrengthDelta = finite(v.themeStrengthDelta).coerceIn(-0.20, 0.20),
        midtoneContrastDelta = finite(v.midtoneContrastDelta).coerceIn(-0.25, 0.25),
        localContrastDelta = finite(v.localContrastDelta).coerceIn(-0.25, 0.25),
        shadowDepthDelta = finite(v.shadowDepthDelta).coerceIn(-0.20, 0.20),
        highlightProtectionDelta = finite(v.highlightProtectionDelta).coerceIn(-0.20, 0.20),
        subjectRelightDeltaEv = finite(v.subjectRelightDeltaEv).coerceIn(-0.25, 0.25),
        backgroundSuppressionDeltaEv = finite(v.backgroundSuppressionDeltaEv).coerceIn(-0.20, 0.20),
        warmthDelta = finite(v.warmthDelta).coerceIn(-0.10, 0.10),
        colorIntensityDelta = finite(v.colorIntensityDelta).coerceIn(-0.15, 0.15)
    )

    private fun finite(v: Double): Double = if (v.isFinite()) v else 0.0
    private fun clean(value: String): String = value.filter { it >= ' ' && it != '\u007f' }.trim()
}
