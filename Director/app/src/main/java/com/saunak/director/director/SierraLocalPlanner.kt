package com.saunak.director.director

import com.saunak.director.processing.TechnicalQuality
import com.saunak.director.tools.ToolIds

/**
 * Deterministic on-device planner used by Sierra β. It intentionally operates on measurable
 * frame statistics and the same semantic tool contract as Sierra AI, so cloud availability only
 * changes who plans the shot; execution/safety remain identical.
 */
object SierraLocalPlanner {
    fun plan(
        quality: TechnicalQuality,
        frameWidth: Int,
        frameHeight: Int
    ): PlanOutput {
        require(frameWidth > 0 && frameHeight > 0)

        val orientation = when {
            frameHeight > frameWidth * 1.10 -> "portrait"
            frameWidth > frameHeight * 1.25 -> "landscape"
            else -> "general"
        }
        val lowLight = quality.meanLuma < 70.0
        val bright = quality.meanLuma > 195.0 || quality.brightClipFraction > 0.05
        val flat = quality.lumaStdDev < 24.0
        val highDynamicRange = quality.lumaStdDev > 58.0 || (quality.darkClipFraction + quality.brightClipFraction) > 0.05

        val branch = when {
            lowLight -> "low_light"
            orientation == "portrait" -> "portrait"
            orientation == "landscape" -> "landscape"
            else -> "general"
        }

        val aspectRatio = when {
            orientation == "portrait" -> "4:5"
            orientation == "landscape" && highDynamicRange -> "16:9"
            orientation == "landscape" -> "3:2"
            else -> "4:3"
        }

        val lookId = when {
            lowLight -> ToolIds.LOOK_NIGHT_NEON
            orientation == "portrait" -> ToolIds.LOOK_CLEAN_PORTRAIT
            orientation == "landscape" && !flat -> ToolIds.LOOK_LANDSCAPE_VIVID
            flat -> ToolIds.LOOK_SOCIAL_POP
            else -> ToolIds.LOOK_NATURAL
        }

        val lookStrength = when {
            quality.brightClipFraction > 0.08 || quality.darkClipFraction > 0.12 -> 0.42
            lowLight -> 0.52
            else -> 0.62
        }

        val calls = buildList {
            when {
                quality.meanLuma < 55.0 -> add(ToolCall(id = ToolIds.EXPOSURE_EV, value = 0.33))
                bright -> add(ToolCall(id = ToolIds.EXPOSURE_EV, value = -0.33))
            }
            add(ToolCall(id = lookId, value = lookStrength))
        }

        val objective = buildString {
            append("Sierra β ")
            append(branch.replace('_', ' '))
            append(" capture")
            if (highDynamicRange) append(" with highlight protection")
            if (lowLight) append(" and restrained low-light finish")
        }

        return PlanOutput(
            scene = branch,
            objective = objective,
            skillBranch = branch,
            referenceScore = quality.heuristicScore.coerceIn(0.0, 1.0),
            referenceAccept = !quality.catastrophic,
            toolCalls = calls,
            recapture = calls.any { it.id in ToolIds.CAMERA_IDS },
            aspectRatio = aspectRatio,
            cropCenterX = 0.5,
            cropCenterY = 0.5,
            planner = "sierra_beta"
        )
    }
}
