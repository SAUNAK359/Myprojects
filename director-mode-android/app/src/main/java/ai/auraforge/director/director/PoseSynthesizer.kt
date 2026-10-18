package ai.auraforge.director.director

import ai.auraforge.director.core.EntityState
import ai.auraforge.director.core.Point2
import ai.auraforge.director.core.PoseState
import ai.auraforge.director.core.TargetPose
import ai.auraforge.director.core.TargetRelation
import kotlin.math.max
import kotlin.math.min

class PoseSynthesizer {

    fun synthesize(
        template: ShotTemplate,
        livePoses: List<PoseState>,
        anchorObject: EntityState?
    ): Pair<List<TargetPose>, TargetRelation?> {
        if (livePoses.isEmpty()) return emptyList<TargetPose>() to null

        val count = min(max(1, template.minimumPeople), livePoses.size)
        val targets = mutableListOf<TargetPose>()

        for (index in 0 until count) {
            val live = livePoses[index]
            val mirrored = shouldMirror(template, live, anchorObject, index)
            val personOffset = groupOffset(index, count)
            val joints = retarget(template, live, mirrored, personOffset, anchorObject)
            targets += TargetPose(index, joints, template.name)
        }

        val relation = template.relation?.let {
            TargetRelation(
                objectLabel = it.objectLabel,
                desiredSide = if (anchorObject == null) 0 else if (targets.firstOrNull()?.joints?.values?.map { p -> p.x }?.average()?.toFloat()
                        ?.let { center -> center < anchorObject.bbox.centerX() } == true) -1 else 1,
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
        mirror: Boolean,
        groupOffset: Float,
        anchorObject: EntityState?
    ): Map<Int, Point2> {
        val liveBox = live.bbox
        val bodyHeight = liveBox.height().coerceAtLeast(0.30f)
        val bodyWidth = liveBox.width().coerceAtLeast(bodyHeight * 0.26f)

        var targetCenterX = template.pose.centerX + groupOffset
        val targetCenterY = template.pose.centerY

        if (anchorObject != null && template.relation != null) {
            val objectCenter = anchorObject.bbox.centerX()
            val personHalfWidth = bodyWidth * 0.5f
            val objectHalfWidth = anchorObject.bbox.width() * 0.5f
            val gap = template.relation.distance
            targetCenterX = if (objectCenter >= 0.5f) {
                (objectCenter - objectHalfWidth - gap - personHalfWidth).coerceIn(0.16f, 0.72f)
            } else {
                (objectCenter + objectHalfWidth + gap + personHalfWidth).coerceIn(0.28f, 0.84f)
            }
        }

        val targetHeight = (bodyHeight * 0.92f + template.pose.scale * 0.12f).coerceIn(0.45f, 0.90f)
        val targetWidth = (bodyWidth * 0.88f).coerceIn(0.14f, 0.38f)

        return template.pose.joints.mapValues { (_, p) ->
            val localX = if (mirror) 1f - p.x else p.x
            val x = targetCenterX + (localX - 0.5f) * targetWidth
            val y = targetCenterY + (p.y - 0.5f) * targetHeight
            Point2(x.coerceIn(0.03f, 0.97f), y.coerceIn(0.03f, 0.97f), 1f)
        }
    }
}
