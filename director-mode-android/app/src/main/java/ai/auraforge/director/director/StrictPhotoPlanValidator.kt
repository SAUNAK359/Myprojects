package ai.auraforge.director.director

import ai.auraforge.director.camera.CameraCapabilities
import ai.auraforge.director.core.ExtensionPreference
import ai.auraforge.director.core.OutputIntent
import ai.auraforge.director.core.PhotoPlan
import ai.auraforge.director.core.SceneState
import kotlin.math.abs

/**
 * Final output/capability contract for Director V3. This layer is intentionally stricter than the
 * creative model: it only passes features that the current capture implementation can genuinely
 * execute end-to-end.
 */
class StrictPhotoPlanValidator {
    fun validate(
        plan: PhotoPlan?,
        scene: SceneState,
        caps: CameraCapabilities,
        availableExtensions: Set<String>
    ): PhotoPlan? {
        if (plan == null || !plan.confidence.isFinite()) return null
        val faceVisible = scene.poses.isNotEmpty() || scene.face != null

        val safeExtension = when {
            plan.extensionPreference == ExtensionPreference.NONE -> ExtensionPreference.NONE
            plan.extensionPreference.name in availableExtensions -> plan.extensionPreference
            else -> ExtensionPreference.NONE
        }

        // RAW+JPEG capability can be discovered, but the current gallery controller is JPEG-only.
        // Never claim RAW until the dedicated DNG/dual-output writer lands.
        val safeOutput = when (plan.outputIntent) {
            OutputIntent.ULTRA_HDR -> if (caps.supportsUltraHdr) OutputIntent.ULTRA_HDR else OutputIntent.JPEG
            OutputIntent.RAW_JPEG -> OutputIntent.JPEG
            OutputIntent.JPEG -> OutputIntent.JPEG
        }

        val nearestAperture = plan.apertureFNumber?.takeIf { it.isFinite() }?.let { requested ->
            caps.apertures.minByOrNull { abs(it - requested) }
        }

        val hardwareMaxZoom = caps.maxDigitalZoom.coerceAtLeast(1f)
        return plan.copy(
            confidence = plan.confidence.coerceIn(0f, 1f),
            extensionPreference = safeExtension,
            outputIntent = safeOutput,
            zoomRatio = plan.zoomRatio.takeIf { it.isFinite() }?.coerceIn(0.5f, minOf(5f, hardwareMaxZoom)) ?: 1f,
            exposureEv = plan.exposureEv.takeIf { it.isFinite() }?.coerceIn(-2f, 2f) ?: 0f,
            apertureFNumber = nearestAperture,
            focusX = plan.focusX?.takeIf { it.isFinite() }?.coerceIn(0.08f, 0.92f),
            focusY = plan.focusY?.takeIf { it.isFinite() }?.coerceIn(0.08f, 0.92f),
            contrast = plan.contrast.takeIf { it.isFinite() }?.coerceIn(-1f, 1f) ?: 0f,
            saturation = plan.saturation.takeIf { it.isFinite() }?.coerceIn(-1f, 1f) ?: 0f,
            temperature = plan.temperature.takeIf { it.isFinite() }?.coerceIn(-1f, 1f) ?: 0f,
            vignette = plan.vignette.takeIf { it.isFinite() }?.coerceIn(0f, 1f) ?: 0f,
            retouchStrength = if (faceVisible) plan.retouchStrength.takeIf { it.isFinite() }?.coerceIn(0f, 0.45f) ?: 0f else 0f,
            backgroundBlurStrength = plan.backgroundBlurStrength.takeIf { it.isFinite() }?.coerceIn(0f, 1f) ?: 0f,
            facePriority = plan.facePriority && faceVisible,
            burstCount = plan.burstCount.coerceIn(1, 3),
            subjectGuidance = plan.subjectGuidance?.trim()?.take(96),
            cameraGuidance = plan.cameraGuidance?.trim()?.take(96),
            rationale = plan.rationale?.trim()?.take(180)
        )
    }
}
