package ai.auraforge.director.director

import ai.auraforge.director.body.BodyProfile
import ai.auraforge.director.body.CropClass
import ai.auraforge.director.core.EntityState
import ai.auraforge.director.core.Point2
import ai.auraforge.director.core.PoseState
import ai.auraforge.director.core.TargetPose
import ai.auraforge.director.core.TargetRelation
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Retargets a canonical recipe onto the observed person's proportions.
 * Canonical joint coordinates define intent/direction only; live BodyProfile segment lengths define
 * the target skeleton dimensions. No fixed full-body mannequin is ever imposed on a cropped subject.
 */
class PoseSynthesizer {

    fun synthesize(
        template: ShotTemplate,
        livePoses: List<PoseState>,
        bodyProfiles: Map<Int, BodyProfile>,
        anchorObject: EntityState?
    ): Pair<List<TargetPose>, TargetRelation?> {
        if (livePoses.isEmpty()) return emptyList<TargetPose>() to null

        val count = min(max(1, template.minimumPeople), livePoses.size)
        val targets = mutableListOf<TargetPose>()

        for (index in 0 until count) {
            val live = livePoses[index]
            val profile = bodyProfiles[live.id]
            val mirrored = shouldMirror(template, live, anchorObject, index)
            val personOffset = groupOffset(index, count)
            val joints = retarget(template, live, profile, mirrored, personOffset, anchorObject)
            if (joints.isNotEmpty()) targets += TargetPose(index, joints, template.name)
        }

        val relation = template.relation?.let {
            TargetRelation(
                objectLabel = it.objectLabel,
                desiredSide = if (anchorObject == null) 0 else if (
                    targets.firstOrNull()?.joints?.values?.map { p -> p.x }?.average()?.toFloat()
                        ?.let { center -> center < anchorObject.bbox.centerX() } == true
                ) -1 else 1,
                desiredDistanceNorm = it.distance,
                weight = it.weight
            )
        }

        return targets to relation
    }

    private fun shouldMirror(
        template: ShotTemplate,
        live: PoseState,
        anchorObject: EntityState?,
        index: Int
    ): Boolean {
        if (!template.pose.mirrorAllowed) return false
        if (anchorObject != null) {
            val freeOnLeft = anchorObject.bbox.centerX() > 0.5f
            return !freeOnLeft
        }
        if (index % 2 == 1) return true
        return live.bbox.centerX() > 0.62f
    }

    private fun groupOffset(index: Int, count: Int): Float {
        if (count <= 1) return 0f
        val span = min(0.48f, 0.19f * (count - 1))
        return -span / 2f + span * index.toFloat() / (count - 1).toFloat()
    }

    private fun retarget(
        template: ShotTemplate,
        live: PoseState,
        profile: BodyProfile?,
        mirror: Boolean,
        groupOffset: Float,
        anchorObject: EntityState?
    ): Map<Int, Point2> {
        val canonical = template.pose.joints
        if (canonical.isEmpty()) return emptyMap()

        val crop = profile?.crop ?: inferLegacyCrop(live)
        val allowed = allowedJointSet(crop)
        val availableCanonical = canonical.filterKeys { it in allowed }
        if (availableCanonical.isEmpty()) return emptyMap()

        val liveBox = live.bbox
        val liveCenterX = liveBox.centerX()
        val liveCenterY = liveBox.centerY()

        // Composition target starts close to where the real person already is; template center only
        // provides a bounded artistic offset instead of forcing a generic absolute placement.
        val templateOffsetX = (template.pose.centerX - 0.5f) * 0.42f
        val templateOffsetY = (template.pose.centerY - 0.5f) * 0.32f
        var targetCenterX = (liveCenterX + templateOffsetX + groupOffset).coerceIn(0.12f, 0.88f)
        var targetCenterY = (liveCenterY + templateOffsetY).coerceIn(0.12f, 0.88f)

        if (anchorObject != null && template.relation != null) {
            val objectCenter = anchorObject.bbox.centerX()
            val personHalfWidth = liveBox.width().coerceAtLeast(0.10f) * 0.5f
            val objectHalfWidth = anchorObject.bbox.width() * 0.5f
            val gap = template.relation.distance
            targetCenterX = if (objectCenter >= 0.5f) {
                (objectCenter - objectHalfWidth - gap - personHalfWidth).coerceIn(0.12f, 0.76f)
            } else {
                (objectCenter + objectHalfWidth + gap + personHalfWidth).coerceIn(0.24f, 0.88f)
            }
        }

        val profileScale = profile?.projectedScale ?: liveBox.height()
        val desiredScale = (profileScale * 0.96f + template.pose.scale * 0.04f).coerceIn(0.10f, 0.94f)

        // First construct a directionally transformed canonical skeleton around its own pelvis/torso.
        val mirroredCanonical = availableCanonical.mapValues { (_, p) ->
            Point2(if (mirror) 1f - p.x else p.x, p.y, p.confidence)
        }

        val root = canonicalRoot(mirroredCanonical)
        val canonicalBox = bboxOf(mirroredCanonical.values.toList())
        val canonicalHeight = canonicalBox.second.coerceAtLeast(0.20f)
        val initialScale = desiredScale / canonicalHeight
        val transformed = mirroredCanonical.mapValues { (_, p) ->
            Point2(
                targetCenterX + (p.x - root.x) * initialScale,
                targetCenterY + (p.y - root.y) * initialScale,
                1f
            )
        }.toMutableMap()

        // Reconstruct connected limbs using the user's stable segment lengths while preserving the
        // canonical pose direction. Torso/root geometry remains the composition anchor.
        if (profile != null) {
            retargetSegment(transformed, mirroredCanonical, 11, 13, profile.leftUpperArm)
            retargetSegment(transformed, mirroredCanonical, 13, 15, profile.leftForearm)
            retargetSegment(transformed, mirroredCanonical, 12, 14, profile.rightUpperArm)
            retargetSegment(transformed, mirroredCanonical, 14, 16, profile.rightForearm)
            retargetSegment(transformed, mirroredCanonical, 23, 25, profile.leftThigh)
            retargetSegment(transformed, mirroredCanonical, 25, 27, profile.leftShin)
            retargetSegment(transformed, mirroredCanonical, 24, 26, profile.rightThigh)
            retargetSegment(transformed, mirroredCanonical, 26, 28, profile.rightShin)

            // Preserve actual shoulder/hip breadth rather than template breadth.
            resizePair(transformed, 11, 12, profile.shoulderWidth)
            resizePair(transformed, 23, 24, profile.hipWidth)
        }

        // Final frame fit: translate only, never non-uniformly stretch the person's body.
        val fitted = translateIntoFrame(transformed)

        // Only joints that are sensible for the current crop remain. Confidence intentionally
        // reflects calibration confidence so UI/alignment can down-weight uncertain targets.
        val confidence = profile?.confidence?.coerceIn(0.45f, 1f) ?: 0.65f
        return fitted
            .filterKeys { it in allowed }
            .mapValues { (_, p) -> Point2(p.x, p.y, confidence) }
    }

    private fun retargetSegment(
        out: MutableMap<Int, Point2>,
        canonical: Map<Int, Point2>,
        parent: Int,
        child: Int,
        length: Float
    ) {
        val p = out[parent] ?: return
        val cp = canonical[parent] ?: return
        val cc = canonical[child] ?: return
        val angle = atan2((cc.y - cp.y).toDouble(), (cc.x - cp.x).toDouble())
        out[child] = Point2(
            p.x + cos(angle).toFloat() * length,
            p.y + sin(angle).toFloat() * length,
            1f
        )
    }

    private fun resizePair(out: MutableMap<Int, Point2>, a: Int, b: Int, targetDistance: Float) {
        val pa = out[a] ?: return
        val pb = out[b] ?: return
        val cx = (pa.x + pb.x) * 0.5f
        val cy = (pa.y + pb.y) * 0.5f
        val angle = atan2((pb.y - pa.y).toDouble(), (pb.x - pa.x).toDouble())
        val hx = cos(angle).toFloat() * targetDistance * 0.5f
        val hy = sin(angle).toFloat() * targetDistance * 0.5f
        out[a] = Point2(cx - hx, cy - hy, 1f)
        out[b] = Point2(cx + hx, cy + hy, 1f)
    }

    private fun translateIntoFrame(points: Map<Int, Point2>): Map<Int, Point2> {
        if (points.isEmpty()) return points
        val xs = points.values.map { it.x }
        val ys = points.values.map { it.y }
        val left = xs.minOrNull() ?: 0f
        val right = xs.maxOrNull() ?: 1f
        val top = ys.minOrNull() ?: 0f
        val bottom = ys.maxOrNull() ?: 1f
        val margin = 0.035f

        var dx = 0f
        var dy = 0f
        if (left < margin) dx += margin - left
        if (right + dx > 1f - margin) dx -= right + dx - (1f - margin)
        if (top < margin) dy += margin - top
        if (bottom + dy > 1f - margin) dy -= bottom + dy - (1f - margin)

        return points.mapValues { (_, p) ->
            Point2((p.x + dx).coerceIn(margin, 1f - margin), (p.y + dy).coerceIn(margin, 1f - margin), p.confidence)
        }
    }

    private fun canonicalRoot(points: Map<Int, Point2>): Point2 {
        val lHip = points[23]
        val rHip = points[24]
        if (lHip != null && rHip != null) return Point2((lHip.x + rHip.x) * 0.5f, (lHip.y + rHip.y) * 0.5f)
        val lShoulder = points[11]
        val rShoulder = points[12]
        if (lShoulder != null && rShoulder != null) return Point2((lShoulder.x + rShoulder.x) * 0.5f, (lShoulder.y + rShoulder.y) * 0.5f)
        return points.values.first()
    }

    private fun bboxOf(points: List<Point2>): Pair<Float, Float> {
        if (points.isEmpty()) return 0f to 1f
        val minY = points.minOf { it.y }
        val maxY = points.maxOf { it.y }
        return minY to (maxY - minY)
    }

    private fun allowedJointSet(crop: CropClass): Set<Int> = when (crop) {
        CropClass.FACE -> setOf(0)
        CropClass.HEAD_SHOULDERS -> setOf(0, 11, 12)
        CropClass.CHEST_UP -> setOf(0, 11, 12, 13, 14, 15, 16)
        CropClass.WAIST_UP -> setOf(0, 11, 12, 13, 14, 15, 16, 23, 24)
        CropClass.THREE_QUARTER -> setOf(0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26)
        CropClass.FULL_BODY, CropClass.ENVIRONMENTAL -> setOf(0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
    }

    private fun inferLegacyCrop(live: PoseState): CropClass = when {
        live.landmarks.getOrNull(27)?.confidence ?: 0f > 0.42f && live.landmarks.getOrNull(28)?.confidence ?: 0f > 0.42f -> CropClass.FULL_BODY
        live.landmarks.getOrNull(25)?.confidence ?: 0f > 0.42f || live.landmarks.getOrNull(26)?.confidence ?: 0f > 0.42f -> CropClass.THREE_QUARTER
        live.landmarks.getOrNull(23)?.confidence ?: 0f > 0.42f && live.landmarks.getOrNull(24)?.confidence ?: 0f > 0.42f -> CropClass.WAIST_UP
        else -> CropClass.CHEST_UP
    }
}
