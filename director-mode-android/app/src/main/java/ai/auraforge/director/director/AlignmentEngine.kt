package ai.auraforge.director.director

import ai.auraforge.director.core.AlignmentState
import ai.auraforge.director.core.DirectorLockState
import ai.auraforge.director.core.DirectorPlan
import ai.auraforge.director.core.Geometry
import ai.auraforge.director.core.Point2
import ai.auraforge.director.core.SceneState
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max

class AlignmentEngine(private val config: DirectorConfig) {
    private var state: DirectorLockState = DirectorLockState.SEARCHING
    private var candidateSince: Long = 0L
    private var lastCaptureAt: Long = 0L

    fun reset() {
        state = DirectorLockState.SEARCHING
        candidateSince = 0L
    }

    fun markCaptured(now: Long) {
        lastCaptureAt = now
        state = DirectorLockState.CAPTURING
    }

    fun canCaptureAgain(now: Long): Boolean = now - lastCaptureAt >= config.lock.captureCooldownMs

    fun evaluate(scene: SceneState, plan: DirectorPlan?): AlignmentState {
        if (plan == null || scene.poses.isEmpty() || plan.targetPoses.isEmpty()) {
            reset()
            return AlignmentState(state = DirectorLockState.SEARCHING, technicalScore = scene.quality.technicalScore)
        }

        val poseScores = mutableListOf<Float>()
        val frameScores = mutableListOf<Float>()
        val headScores = mutableListOf<Float>()

        plan.targetPoses.forEach { target ->
            val live = scene.poses.getOrNull(target.personIndex) ?: return@forEach
            poseScores += poseScore(live.landmarks, target.joints)
            frameScores += framingScore(live.bbox, target.joints.values.toList())
            headScores += headScore(live.landmarks, target.joints)
        }

        val poseScore = poseScores.averageOrZero()
        val framingScore = frameScores.averageOrZero()
        val headScore = headScores.averageOrZero()
        val relationScore = relationScore(scene, plan)
        val compositionScore = compositionScore(scene, plan)

        val w = config.weights
        val total = (
            w.pose * poseScore +
                w.framing * framingScore +
                w.relation * relationScore +
                w.head * headScore +
                w.composition * compositionScore
            ).coerceIn(0f, 1f)

        val moment = momentScore(scene)
        val technical = scene.quality.technicalScore
        val now = scene.timestampMs

        state = nextState(now, total, technical, moment)
        val stable = if (candidateSince > 0) now - candidateSince else 0L

        return AlignmentState(
            poseScore = poseScore,
            framingScore = framingScore,
            relationScore = relationScore,
            headScore = headScore,
            compositionScore = compositionScore,
            totalScore = total,
            technicalScore = technical,
            momentScore = moment,
            state = state,
            stableMs = stable
        )
    }

    private fun nextState(now: Long, total: Float, technical: Float, moment: Float): DirectorLockState {
        val l = config.lock
        if (state == DirectorLockState.LOCKED && total >= l.unlock && technical >= l.technicalMin) return DirectorLockState.LOCKED

        if (total < l.aligning) {
            candidateSince = 0L
            return DirectorLockState.ALIGNING
        }
        if (total < l.nearLock) {
            candidateSince = 0L
            return DirectorLockState.ALIGNING
        }
        if (total < l.candidate) {
            candidateSince = 0L
            return DirectorLockState.NEAR_LOCK
        }

        if (technical < l.technicalMin || moment < 0.54f) {
            candidateSince = 0L
            return DirectorLockState.LOCK_CANDIDATE
        }

        if (candidateSince == 0L) candidateSince = now
        return if (now - candidateSince >= l.candidateHoldMs) DirectorLockState.LOCKED else DirectorLockState.LOCK_CANDIDATE
    }

    private fun poseScore(live: List<Point2>, target: Map<Int, Point2>): Float {
        if (live.isEmpty()) return 0f
        val indices = target.keys.filter { it < live.size }
        if (indices.isEmpty()) return 0f

        var sum = 0f
        var weight = 0f
        indices.forEach { idx ->
            val l = live[idx]
            if (l.confidence < 0.30f) return@forEach
            val t = target[idx] ?: return@forEach
            val importance = when (idx) {
                11, 12, 23, 24 -> 1.35f
                13, 14, 25, 26 -> 1.10f
                15, 16, 27, 28 -> 0.95f
                0 -> 0.90f
                else -> 0.65f
            }
            val d = l.distance(t)
            sum += importance * exp((-7.5f * d).toDouble()).toFloat()
            weight += importance
        }
        return if (weight <= 0f) 0f else (sum / weight).coerceIn(0f, 1f)
    }

    private fun framingScore(liveBox: android.graphics.RectF, targetPoints: List<Point2>): Float {
        if (targetPoints.isEmpty()) return 0f
        val targetBox = Geometry.bbox(targetPoints, 0f)
        val centerDx = abs(liveBox.centerX() - targetBox.centerX())
        val centerDy = abs(liveBox.centerY() - targetBox.centerY())
        val scaleError = abs(liveBox.height() - targetBox.height())
        return exp((-(4.4f * centerDx + 3.6f * centerDy + 2.5f * scaleError)).toDouble()).toFloat().coerceIn(0f, 1f)
    }

    private fun headScore(live: List<Point2>, target: Map<Int, Point2>): Float {
        if (live.size <= 12 || target[0] == null || target[11] == null || target[12] == null) return 1f
        val lNose = live[0]
        val lCenter = Point2((live[11].x + live[12].x) / 2f, (live[11].y + live[12].y) / 2f)
        val tNose = target[0]!!
        val tCenter = Point2((target[11]!!.x + target[12]!!.x) / 2f, (target[11]!!.y + target[12]!!.y) / 2f)
        val liveVecX = lNose.x - lCenter.x
        val targetVecX = tNose.x - tCenter.x
        return (1f - abs(liveVecX - targetVecX) * 8f).coerceIn(0f, 1f)
    }

    private fun relationScore(scene: SceneState, plan: DirectorPlan): Float {
        val relation = plan.relation ?: return 1f
        val obj = scene.primary(relation.objectLabel) ?: return 0f
        val person = scene.poses.firstOrNull() ?: return 0f
        val gap = kotlin.math.abs(person.bbox.centerX() - obj.bbox.centerX()) - (person.bbox.width() + obj.bbox.width()) / 2f
        val distanceScore = exp((-4.5f * abs(gap - relation.desiredDistanceNorm)).toDouble()).toFloat()
        val currentSide = if (person.bbox.centerX() < obj.bbox.centerX()) -1 else 1
        val sideScore = if (relation.desiredSide == 0 || currentSide == relation.desiredSide) 1f else 0.35f
        val overlapPenalty = Geometry.intersectionOverUnion(person.bbox, obj.bbox)
        return (distanceScore * sideScore * (1f - 0.85f * overlapPenalty)).coerceIn(0f, 1f)
    }

    private fun compositionScore(scene: SceneState, plan: DirectorPlan): Float {
        val pose = scene.poses.firstOrNull() ?: return 0f
        val thirdX = if (plan.relation != null) if (pose.bbox.centerX() < 0.5f) 1f / 3f else 2f / 3f else 0.5f
        val horizontal = (1f - abs(pose.bbox.centerX() - thirdX) * 2.1f).coerceIn(0f, 1f)
        val headroom = pose.bbox.top
        val headroomScore = (1f - abs(headroom - 0.08f) * 5f).coerceIn(0f, 1f)
        return 0.72f * horizontal + 0.28f * headroomScore
    }

    private fun momentScore(scene: SceneState): Float {
        val eyes = scene.face?.eyesOpenScore ?: 0.84f
        val smile = scene.face?.smile ?: 0.55f
        val stability = scene.quality.stabilityScore
        return (0.42f * eyes + 0.18f * smile + 0.40f * stability).coerceIn(0f, 1f)
    }
}

private fun List<Float>.averageOrZero(): Float = if (isEmpty()) 0f else average().toFloat()
