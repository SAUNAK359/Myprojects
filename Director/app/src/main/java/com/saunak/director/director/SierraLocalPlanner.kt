package com.saunak.director.director

import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.camera.DirectorExtension
import com.saunak.director.processing.TechnicalQuality
import com.saunak.director.tools.ToolIds

/** Deterministic on-device planner used by Sierra beta. */
object SierraLocalPlanner {
    fun plan(
        quality: TechnicalQuality,
        frameWidth: Int,
        frameHeight: Int,
        signals: SierraSceneSignals? = null,
        capabilities: CapabilityRegistry? = null
    ): PlanOutput {
        require(frameWidth > 0 && frameHeight > 0)

        val fallbackBranch = when {
            quality.meanLuma < 70.0 && quality.medianLuma < 65.0 -> "low_light"
            frameWidth > frameHeight * 1.18 -> "landscape"
            else -> "general"
        }
        val branch = signals?.branch ?: fallbackBranch
        val lowLight = branch == "low_light"
        val brightSky = signals?.brightSky == true || (
            quality.highlightP95 >= 218.0 && quality.topMeanLuma - quality.bottomMeanLuma >= 28.0
        )
        val highDynamicRange = brightSky || quality.lumaStdDev > 52.0 ||
            quality.highlightP95 - quality.shadowP05 > 150.0
        val environmental = branch == "landscape" || branch == "architecture"
        val portraitSubject = branch == "portrait" || branch == "selfie" || branch == "group"
        val flat = quality.lumaStdDev < 24.0

        val aspectRatio = when {
            portraitSubject -> if (frameHeight >= frameWidth) "4:5" else "3:2"
            environmental && frameWidth > frameHeight * 1.12 -> if (brightSky) "16:9" else "3:2"
            environmental && frameHeight > frameWidth * 1.12 -> "4:5"
            frameHeight > frameWidth * 1.18 -> "4:5"
            else -> "4:3"
        }

        val lookId = when {
            lowLight -> ToolIds.LOOK_CINEMATIC_COOL
            portraitSubject -> ToolIds.LOOK_CLEAN_PORTRAIT
            branch == "architecture" || branch == "landscape" -> ToolIds.LOOK_LANDSCAPE_VIVID
            flat -> ToolIds.LOOK_SOCIAL_POP
            else -> ToolIds.LOOK_NATURAL
        }
        val lookStrength = when {
            branch == "architecture" -> 0.30
            branch == "landscape" -> 0.36
            portraitSubject -> 0.40
            lowLight -> 0.42
            flat -> 0.28
            else -> 0.0
        }

        val severeHighlights = quality.highlightP95 >= 248.0 || quality.brightClipFraction > 0.03
        val exposureEv = when {
            severeHighlights -> -0.33
            brightSky -> -0.17
            quality.meanLuma > 195.0 -> -0.17
            lowLight && quality.medianLuma < 50.0 && quality.highlightP95 < 175.0 -> 0.33
            quality.medianLuma < 82.0 && quality.highlightP95 < 190.0 -> 0.17
            else -> null
        }

        val hdrSupported = capabilities?.let {
            DirectorExtension.HDR in it.extensions && ToolIds.EXTENSION in it.allowedToolIds
        } == true
        val useHdr = environmental && highDynamicRange && hdrSupported

        val calls = buildList {
            if (useHdr) {
                add(ToolCall(id = ToolIds.EXTENSION, stringValue = DirectorExtension.HDR.wireName))
            } else if (exposureEv != null) {
                add(ToolCall(id = ToolIds.EXPOSURE_EV, value = exposureEv))
            }
            if (lookId != ToolIds.LOOK_NATURAL || lookStrength > 0.0) {
                add(ToolCall(id = lookId, value = lookStrength))
            }
        }

        val objective = buildString {
            append("Sierra beta ")
            append(branch.replace('_', ' '))
            append(" capture")
            if (brightSky) append(" with sky highlight protection and preserved foreground detail")
            else if (highDynamicRange) append(" with controlled dynamic range")
            if (branch == "architecture") append(" and neutral structural detail")
            if (lowLight) append(" with restrained low-light rendering")
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
