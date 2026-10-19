package ai.auraforge.director.director

import ai.auraforge.director.camera.CameraCapabilities
import ai.auraforge.director.core.CaptureProgram
import ai.auraforge.director.core.ColorStyle
import ai.auraforge.director.core.CompositionGuide
import ai.auraforge.director.core.ExtensionPreference
import ai.auraforge.director.core.FlashIntent
import ai.auraforge.director.core.FocusIntent
import ai.auraforge.director.core.HardwareEffect
import ai.auraforge.director.core.OutputIntent
import ai.auraforge.director.core.PhotoPlan
import ai.auraforge.director.core.SceneState
import ai.auraforge.director.core.WhiteBalanceIntent

/**
 * Bounded-latency safety planner.
 *
 * Director's VLM remains the preferred creative planner, but the camera must never sit in
 * "READING SCENE" forever because a model/backend failed to initialize or an inference stalled.
 * This fallback uses measured scene/camera evidence only and creates a conservative, executable
 * treatment. A later valid VLM plan may replace it if the VLM recovers within the session.
 */
class DeterministicPhotoFallback {
    fun plan(
        scene: SceneState,
        caps: CameraCapabilities,
        availableExtensions: Set<String>
    ): PhotoPlan {
        val hasHuman = scene.poses.isNotEmpty() || scene.face != null || scene.has("person")
        val singleHuman = hasHuman && scene.personCount <= 1
        val q = scene.quality

        val lowLight = q.meanLuma < 0.30f || q.shadowClipping > 0.34f
        val highDynamicRange = q.highlightClipping > 0.055f && q.shadowClipping > 0.10f
        val meaningfulMotion = q.cameraMotion > 0.32f

        val program = when {
            lowLight && !meaningfulMotion -> CaptureProgram.NIGHT
            highDynamicRange -> CaptureProgram.HDR
            hasHuman -> CaptureProgram.PORTRAIT
            else -> CaptureProgram.STANDARD
        }

        val extension = when {
            lowLight && !meaningfulMotion && "NIGHT" in availableExtensions -> ExtensionPreference.NIGHT
            highDynamicRange && "HDR" in availableExtensions -> ExtensionPreference.HDR
            singleHuman && "BOKEH" in availableExtensions -> ExtensionPreference.BOKEH
            "AUTO" in availableExtensions -> ExtensionPreference.AUTO
            else -> ExtensionPreference.NONE
        }

        val focus = when {
            hasHuman -> FocusIntent.FACE
            scene.entities.isNotEmpty() -> FocusIntent.SUBJECT
            else -> FocusIntent.AUTO
        }

        val exposureEv = when {
            q.meanLuma < 0.22f -> 0.55f
            q.meanLuma < 0.34f -> 0.30f
            q.highlightClipping > 0.12f -> -0.45f
            q.meanLuma > 0.70f -> -0.25f
            else -> 0f
        }

        val output = if (caps.supportsUltraHdr && highDynamicRange) OutputIntent.ULTRA_HDR else OutputIntent.JPEG
        val theme = when (program) {
            CaptureProgram.NIGHT -> "Adaptive Night"
            CaptureProgram.HDR -> "Balanced HDR"
            CaptureProgram.PORTRAIT -> "Adaptive Portrait"
            else -> "Adaptive Auto"
        }

        return PhotoPlan(
            theme = theme,
            confidence = 0.72f,
            program = program,
            compositionGuide = if (hasHuman) CompositionGuide.THIRDS else CompositionGuide.THIRDS,
            extensionPreference = extension,
            focusIntent = focus,
            flashIntent = FlashIntent.OFF,
            whiteBalanceIntent = WhiteBalanceIntent.AUTO,
            colorStyle = ColorStyle.NATURAL,
            hardwareEffect = HardwareEffect.OFF,
            outputIntent = output,
            zoomRatio = 1f,
            exposureEv = exposureEv,
            apertureFNumber = null,
            focusX = scene.poses.firstOrNull()?.bbox?.centerX(),
            focusY = scene.poses.firstOrNull()?.bbox?.let { it.top + it.height() * 0.24f },
            contrast = 0f,
            saturation = 0f,
            temperature = 0f,
            vignette = 0f,
            retouchStrength = 0f,
            backgroundBlurStrength = if (singleHuman && extension == ExtensionPreference.BOKEH) 0.55f else 0f,
            facePriority = hasHuman,
            burstCount = if (meaningfulMotion) 3 else 1,
            subjectGuidance = null,
            cameraGuidance = if (meaningfulMotion) "Hold steady for the final frame" else null,
            rationale = "Bounded-latency local fallback based on measured light, motion, subject and camera capabilities",
            createdAtMs = scene.timestampMs
        )
    }
}
