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
 * Deterministic safety-net photographer. It is intentionally conservative: it uses measured scene
 * physics and real camera capabilities to create a coherent, immediately executable treatment when
 * the generative Director is unavailable or exceeds its latency budget.
 */
class FallbackPhotoPlanner {
    fun plan(
        scene: SceneState,
        caps: CameraCapabilities,
        availableExtensions: Set<String>,
        timestampMs: Long = scene.timestampMs
    ): PhotoPlan {
        val q = scene.quality
        val people = scene.personCount
        val hasPerson = people > 0 || scene.poses.isNotEmpty() || scene.face != null
        val highDynamicRange = q.highlightClipping > 0.055f || (q.contrast > 0.62f && q.shadowClipping > 0.08f)
        val genuinelyDark = q.meanLuma < 0.20f && q.highlightClipping < 0.10f
        val moving = q.cameraMotion > 0.34f

        val program = when {
            genuinelyDark && !moving -> CaptureProgram.NIGHT
            highDynamicRange -> CaptureProgram.HDR
            hasPerson && people <= 2 -> CaptureProgram.PORTRAIT
            moving -> CaptureProgram.ACTION
            else -> CaptureProgram.STANDARD
        }

        val extension = when {
            program == CaptureProgram.NIGHT && "NIGHT" in availableExtensions -> ExtensionPreference.NIGHT
            program == CaptureProgram.HDR && "HDR" in availableExtensions -> ExtensionPreference.HDR
            program == CaptureProgram.PORTRAIT && people == 1 && "BOKEH" in availableExtensions -> ExtensionPreference.BOKEH
            "AUTO" in availableExtensions -> ExtensionPreference.AUTO
            else -> ExtensionPreference.NONE
        }

        val output = when {
            caps.supportsUltraHdr && highDynamicRange -> OutputIntent.ULTRA_HDR
            else -> OutputIntent.JPEG
        }

        val exposureEv = when {
            q.meanLuma < 0.17f -> 0.45f
            q.meanLuma < 0.29f -> 0.25f
            q.meanLuma > 0.83f || q.highlightClipping > 0.12f -> -0.45f
            q.meanLuma > 0.70f -> -0.20f
            else -> 0f
        }

        val focus = when {
            hasPerson -> FocusIntent.FACE
            scene.entities.isNotEmpty() -> FocusIntent.SUBJECT
            else -> FocusIntent.AUTO
        }

        val style = when {
            genuinelyDark -> ColorStyle.CINEMATIC_COOL
            hasPerson && q.meanLuma < 0.55f -> ColorStyle.CINEMATIC_WARM
            q.contrast > 0.72f -> ColorStyle.MUTED
            else -> ColorStyle.NATURAL
        }

        val grid = when {
            people >= 3 -> CompositionGuide.CENTER
            hasPerson -> CompositionGuide.THIRDS
            kotlin.math.abs(q.deviceRollDeg) < 1.5f -> CompositionGuide.THIRDS
            else -> CompositionGuide.CENTER
        }

        val zoom = when {
            hasPerson && people == 1 && caps.maxDigitalZoom >= 1.2f -> 1.15f
            else -> 1f
        }

        val theme = when (program) {
            CaptureProgram.PORTRAIT -> "Adaptive Portrait"
            CaptureProgram.NIGHT -> "Cinematic Night"
            CaptureProgram.HDR -> "Balanced HDR"
            CaptureProgram.ACTION -> "Motion Ready"
            else -> "Smart Cinematic"
        }

        return PhotoPlan(
            theme = theme,
            confidence = 0.72f,
            program = program,
            compositionGuide = grid,
            extensionPreference = extension,
            focusIntent = focus,
            flashIntent = if (genuinelyDark && hasPerson && caps.flashAvailable) FlashIntent.AUTO else FlashIntent.OFF,
            whiteBalanceIntent = WhiteBalanceIntent.AUTO,
            colorStyle = style,
            hardwareEffect = HardwareEffect.OFF,
            outputIntent = output,
            zoomRatio = zoom,
            exposureEv = exposureEv,
            apertureFNumber = null,
            focusX = null,
            focusY = null,
            contrast = when { q.contrast < 0.28f -> 0.10f; q.contrast > 0.78f -> -0.08f; else -> 0f },
            saturation = if (style == ColorStyle.MUTED) -0.10f else 0f,
            temperature = if (style == ColorStyle.CINEMATIC_WARM) 0.10f else if (style == ColorStyle.CINEMATIC_COOL) -0.08f else 0f,
            vignette = if (hasPerson) 0.08f else 0f,
            retouchStrength = if (hasPerson) 0.08f else 0f,
            backgroundBlurStrength = if (extension == ExtensionPreference.BOKEH) 0.42f else 0f,
            facePriority = hasPerson,
            burstCount = when { moving -> 3; hasPerson -> 2; else -> 1 },
            subjectGuidance = null,
            cameraGuidance = when {
                q.cameraMotion > 0.55f -> "Hold steady for a moment"
                kotlin.math.abs(q.deviceRollDeg) > 4f -> "Level the camera slightly"
                else -> null
            },
            rationale = "Latency-safe treatment derived from measured light, motion, subject and camera capabilities",
            createdAtMs = timestampMs
        )
    }
}
