package ai.auraforge.director.director

import ai.auraforge.director.core.DirectorPlan
import ai.auraforge.director.core.Geometry
import ai.auraforge.director.core.PoseState
import ai.auraforge.director.core.SceneState
import kotlin.math.abs
import kotlin.math.max

class DirectorBrain(
    private val templates: ShotTemplateRepository,
    private val config: DirectorConfig,
    private val synthesizer: PoseSynthesizer = PoseSynthesizer()
) {
    private var committedTemplateId: String? = null
    private var lastPlanAt = 0L

    fun releasePlan() {
        committedTemplateId = null
        lastPlanAt = 0L
    }

    fun plan(scene: SceneState, force: Boolean = false): DirectorPlan? {
        if (scene.poses.isEmpty()) return null
        val now = scene.timestampMs
        if (!force && now - lastPlanAt < config.director.replanIntervalMs) {
            committedTemplateId?.let { id ->
                templates.templates.firstOrNull { it.id == id }?.let { return materialize(it, scene) }
            }
        }

        val feasible = templates.templates
            .asSequence()
            .filter { isFeasible(it, scene) }
            .map { it to score(it, scene) }
            .sortedByDescending { it.second }
            .toList()

        if (feasible.isEmpty()) return null

        val committed = committedTemplateId?.let { id -> feasible.firstOrNull { it.first.id == id } }
        val chosen = committed ?: feasible.first()

        val currentQuality = currentAesthetic(scene)
        val expectedGain = (chosen.second - currentQuality).coerceIn(-1f, 1f)
        if (expectedGain < config.director.minimumExpectedGain && committed == null) return null

        if (committedTemplateId == null || force) committedTemplateId = chosen.first.id
        lastPlanAt = now
        return materialize(chosen.first, scene, chosen.second, expectedGain)
    }

    private fun isFeasible(t: ShotTemplate, scene: SceneState): Boolean {
        if (scene.personCount < t.minimumPeople) return false
        if (t.requires.any { label -> label != "person" && !scene.has(label) }) return false
        if (t.forbids.any(scene::has)) return false

        val primary = scene.poses.firstOrNull() ?: return false
        if (primary.bbox.height() < 0.26f && t.pose.scale > 0.78f) return false

        // Conservative safety policy: animal templates never place the human inside the animal bbox.
        t.relation?.let { r ->
            val obj = scene.primary(r.objectLabel) ?: return false
            if (obj.bbox.width() > 0.82f) return false
        }
        return true
    }

    private fun score(t: ShotTemplate, scene: SceneState): Float {
        var s = 0.50f * t.aestheticPrior
        s += 0.16f * (1f - t.difficulty)

        val pose = scene.poses.first()
        val bodyCoverage = pose.bbox.height().coerceIn(0f, 1f)
        s += 0.10f * when {
            bodyCoverage in 0.42f..0.90f -> 1f
            bodyCoverage < 0.25f -> 0.2f
            else -> 0.6f
        }

        if (t.preferredObjects.isNotEmpty()) {
            val found = t.preferredObjects.count(scene::has).toFloat() / t.preferredObjects.size
            s += 0.14f * found
        } else {
            s += 0.07f
        }

        val clutter = backgroundClutter(scene, pose)
        s += 0.06f * (1f - clutter)
        s += 0.04f * scene.quality.technicalScore

        return s.coerceIn(0f, 1f)
    }

    private fun currentAesthetic(scene: SceneState): Float {
        val pose = scene.poses.firstOrNull() ?: return 0f
        val joints = pose.landmarks
        if (joints.size < 29) return 0.35f

        val shoulderTilt = abs(joints[11].y - joints[12].y)
        val hipTilt = abs(joints[23].y - joints[24].y)
        val stance = abs(joints[27].x - joints[28].x)
        val leftArm = Geometry.angleDeg(joints[11], joints[13], joints[15])
        val rightArm = Geometry.angleDeg(joints[12], joints[14], joints[16])
        val asymmetry = (abs(leftArm - rightArm) / 120f).coerceIn(0f, 1f)

        val centeredPenalty = (1f - abs(pose.bbox.centerX() - 0.5f) * 2f).coerceIn(0f, 1f)
        val naturalStance = (stance / 0.22f).coerceIn(0f, 1f)
        val poseInterest = (0.32f * asymmetry + 0.18f * (shoulderTilt + hipTilt).coerceIn(0f, 1f) + 0.22f * naturalStance)
        val composition = 0.55f + 0.45f * (1f - centeredPenalty * 0.35f)

        return (0.38f * poseInterest + 0.34f * composition + 0.28f * scene.quality.technicalScore).coerceIn(0f, 1f)
    }

    private fun backgroundClutter(scene: SceneState, pose: PoseState): Float {
        if (scene.entities.isEmpty()) return 0f
        val penalties = scene.entities
            .filterNot { it.label.equals("person", true) }
            .map { e ->
                val overlap = Geometry.intersectionOverUnion(pose.bbox, e.bbox)
                val nearHead = if (e.bbox.contains(pose.bbox.centerX(), pose.bbox.top + pose.bbox.height() * 0.12f)) 0.4f else 0f
                max(overlap, nearHead) * e.score
            }
        return (penalties.sum() / max(1, penalties.size)).coerceIn(0f, 1f)
    }

    private fun materialize(
        template: ShotTemplate,
        scene: SceneState,
        confidence: Float = template.aestheticPrior,
        expectedGain: Float = 0.1f
    ): DirectorPlan {
        val anchor = template.relation?.let { scene.primary(it.objectLabel) }
        val (targetPoses, relation) = synthesizer.synthesize(template, scene.poses, anchor)
        return DirectorPlan(
            templateId = template.id,
            displayName = template.name,
            targetPoses = targetPoses,
            relation = relation,
            confidence = confidence,
            expectedGain = expectedGain,
            style = template.style
        )
    }
}
