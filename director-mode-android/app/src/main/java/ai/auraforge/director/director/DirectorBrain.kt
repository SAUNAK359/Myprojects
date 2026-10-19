package ai.auraforge.director.director

import ai.auraforge.director.body.CropClass
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

    fun candidateTemplates(scene: SceneState, limit: Int = 12): List<ShotTemplate> =
        templates.templates.asSequence()
            .filter { isFeasible(it, scene) }
            .map { it to score(it, scene) }
            .sortedByDescending { it.second }
            .take(limit)
            .map { it.first }
            .toList()

    fun plan(scene: SceneState, force: Boolean = false, blueprint: CreativeBlueprint? = null): DirectorPlan? {
        if (scene.poses.isEmpty()) return null
        val now = scene.timestampMs

        val blueprintTemplate = blueprint?.let { bp ->
            templates.templates.firstOrNull { it.id == bp.recipeId && isFeasible(it, scene) }
        }

        if (!force && blueprintTemplate == null && now - lastPlanAt < config.director.replanIntervalMs) {
            committedTemplateId?.let { id ->
                templates.templates.firstOrNull { it.id == id }?.let { template ->
                    if (isFeasible(template, scene)) return materialize(template, scene)
                }
            }
        }

        val feasible = templates.templates
            .asSequence()
            .filter { isFeasible(it, scene) }
            .map { it to score(it, scene) }
            .sortedByDescending { it.second }
            .toList()
        if (feasible.isEmpty()) return null

        val chosen = if (blueprintTemplate != null) {
            val baseScore = score(blueprintTemplate, scene)
            blueprintTemplate to max(baseScore, blueprint.confidence.coerceIn(0f, 1f))
        } else {
            val committed = committedTemplateId?.let { id -> feasible.firstOrNull { it.first.id == id } }
            val best = feasible.first()
            if (committed != null && best.second < committed.second + PLAN_SWITCH_MARGIN) committed else best
        }

        val currentQuality = currentAesthetic(scene)
        val expectedGain = (chosen.second - currentQuality).coerceIn(-1f, 1f)
        if (expectedGain < config.director.minimumExpectedGain && committedTemplateId == null && blueprintTemplate == null) return null

        committedTemplateId = chosen.first.id
        lastPlanAt = now
        return materialize(chosen.first, scene, chosen.second, expectedGain, blueprint)
    }

    private fun isFeasible(t: ShotTemplate, scene: SceneState): Boolean {
        if (scene.personCount < t.minimumPeople) return false
        if (t.requires.any { label -> label != "person" && !scene.has(label) }) return false
        if (t.forbids.any(scene::has)) return false

        val primary = scene.poses.firstOrNull() ?: return false
        val profile = scene.bodyProfiles[primary.id]
        val currentCrop = profile?.crop ?: inferCrop(primary)
        if (currentCrop !in t.supportedCrops) return false

        val canonicalJoints = t.pose.joints.keys
        val required = requiredJointsForCrop(currentCrop).intersect(canonicalJoints)
        if (required.isEmpty()) return false

        val visible = profile?.visibleJoints ?: primary.landmarks.indices
            .filterTo(mutableSetOf()) { primary.landmarks[it].confidence >= 0.35f }
        val visibleRatio = required.count { it in visible }.toFloat() / required.size.coerceAtLeast(1)
        if (visibleRatio < MIN_REQUIRED_VISIBILITY) return false

        t.relation?.let { relation ->
            val obj = scene.primary(relation.objectLabel) ?: return false
            if (obj.bbox.width() > 0.90f && obj.bbox.height() > 0.90f) return false
        }
        return true
    }

    private fun score(t: ShotTemplate, scene: SceneState): Float {
        var s = 0.44f * t.aestheticPrior
        s += 0.14f * (1f - t.difficulty)

        val pose = scene.poses.first()
        val profile = scene.bodyProfiles[pose.id]
        val bodyCoverage = pose.bbox.height().coerceIn(0f, 1f)
        s += 0.12f * if (profile != null && profile.crop in t.supportedCrops) 1f else 0.55f
        s += 0.07f * when {
            bodyCoverage in 0.30f..0.92f -> 1f
            bodyCoverage < 0.18f -> 0.25f
            else -> 0.62f
        }

        if (t.preferredObjects.isNotEmpty()) {
            val found = t.preferredObjects.count(scene::has).toFloat() / t.preferredObjects.size
            s += 0.10f * found
        } else s += 0.05f

        s += 0.05f * (1f - backgroundClutter(scene, pose))
        s += 0.03f * scene.quality.technicalScore

        val subjectLuma = scene.quality.lumaGrid.mean(pose.bbox)
        val globalLuma = scene.quality.lumaGrid.meanAll()
        if (subjectLuma != null && globalLuma != null) {
            val separation = abs(subjectLuma - globalLuma).coerceIn(0f, 0.35f) / 0.35f
            s += 0.05f * separation
        }
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
        val poseInterest = 0.32f * asymmetry + 0.18f * (shoulderTilt + hipTilt).coerceIn(0f, 1f) + 0.22f * naturalStance
        val composition = 0.55f + 0.45f * (1f - centeredPenalty * 0.35f)
        return (0.38f * poseInterest + 0.34f * composition + 0.28f * scene.quality.technicalScore).coerceIn(0f, 1f)
    }

    private fun backgroundClutter(scene: SceneState, pose: PoseState): Float {
        if (scene.entities.isEmpty()) return 0f
        val penalties = scene.entities.filterNot { it.label.equals("person", true) }.map { e ->
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
        expectedGain: Float = 0.1f,
        blueprint: CreativeBlueprint? = null
    ): DirectorPlan {
        val anchor = template.relation?.let { scene.primary(it.objectLabel) }
        val (targetPoses, relation) = synthesizer.synthesize(template, scene.poses, scene.bodyProfiles, anchor, blueprint)
        return DirectorPlan(
            templateId = template.id,
            displayName = template.name,
            targetPoses = targetPoses,
            relation = relation,
            confidence = confidence,
            expectedGain = expectedGain,
            style = template.style,
            source = if (blueprint != null) "VLM" else "DETERMINISTIC",
            subjectGuidance = blueprint?.subjectGuidance,
            cameraGuidance = blueprint?.cameraGuidance,
            lightingGuidance = blueprint?.lightingGuidance,
            backgroundGuidance = blueprint?.backgroundGuidance,
            rationale = blueprint?.rationale
        )
    }

    private fun inferCrop(pose: PoseState): CropClass {
        fun visible(i: Int) = (pose.landmarks.getOrNull(i)?.confidence ?: 0f) >= 0.42f
        return when {
            visible(27) || visible(28) -> CropClass.FULL_BODY
            visible(25) || visible(26) -> CropClass.THREE_QUARTER
            visible(23) && visible(24) -> CropClass.WAIST_UP
            visible(13) || visible(14) -> CropClass.CHEST_UP
            visible(11) && visible(12) -> CropClass.HEAD_SHOULDERS
            else -> CropClass.FACE
        }
    }

    private fun requiredJointsForCrop(crop: CropClass): Set<Int> = when (crop) {
        CropClass.FACE -> setOf(0)
        CropClass.HEAD_SHOULDERS -> setOf(0, 11, 12)
        CropClass.CHEST_UP -> setOf(0, 11, 12, 13, 14)
        CropClass.WAIST_UP -> setOf(0, 11, 12, 13, 14, 23, 24)
        CropClass.THREE_QUARTER -> setOf(0, 11, 12, 13, 14, 23, 24, 25, 26)
        CropClass.FULL_BODY, CropClass.ENVIRONMENTAL -> setOf(0, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28)
    }

    companion object {
        private const val MIN_REQUIRED_VISIBILITY = 0.70f
        private const val PLAN_SWITCH_MARGIN = 0.07f
    }
}
