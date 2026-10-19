package ai.auraforge.director.director

import ai.auraforge.director.core.FocusIntent
import ai.auraforge.director.core.PhotoPlan
import ai.auraforge.director.core.PlanReadiness
import ai.auraforge.director.core.SceneState

/**
 * READY means Director has a validated treatment and the current frame is usable.
 * It is deliberately not a perfection gate: minor sharpness/exposure imperfections can continue
 * refining after READY, while genuinely unusable conditions still hold the shutter white.
 */
class BoundedPhotoReadinessEvaluator {
    fun evaluate(scene: SceneState, plan: PhotoPlan?): PlanReadiness {
        if (plan == null) return PlanReadiness(false, 0f, "AI designing shot")

        val q = scene.quality
        val subjectRequired = plan.facePriority || plan.focusIntent == FocusIntent.FACE
        val subjectPresent = !subjectRequired || scene.poses.isNotEmpty() || scene.face != null
        val planConfidence = plan.confidence.coerceIn(0f, 1f)
        val clippingScore = (1f - (q.highlightClipping + q.shadowClipping) * 1.6f).coerceIn(0f, 1f)
        val subjectScore = if (subjectPresent) 1f else 0.25f

        val score = (
            0.34f * planConfidence +
                0.20f * q.exposureScore +
                0.14f * q.sharpness +
                0.14f * q.stabilityScore +
                0.10f * clippingScore +
                0.08f * subjectScore
            ).coerceIn(0f, 1f)

        val extremeMotion = q.cameraMotion > 0.74f
        val unusableExposure = q.meanLuma < 0.035f || q.meanLuma > 0.975f
        val subjectMissing = subjectRequired && !subjectPresent
        val planUncertain = planConfidence < 0.45f

        val ready = !extremeMotion && !unusableExposure && !subjectMissing && !planUncertain

        val reason = when {
            planUncertain -> "AI refining treatment"
            extremeMotion -> "Hold camera steady"
            unusableExposure -> "Finding usable exposure"
            subjectMissing -> "Acquiring subject"
            q.highlightClipping > 0.20f -> "Treatment locked · protecting highlights"
            q.shadowClipping > 0.38f -> "Treatment locked · lifting shadows"
            q.sharpness < 0.20f -> "Treatment locked · focus refining"
            else -> "Treatment locked"
        }
        return PlanReadiness(ready, score, reason)
    }
}
