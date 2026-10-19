package ai.auraforge.director.body

import ai.auraforge.director.core.Point2
import ai.auraforge.director.core.PoseState
import kotlin.math.abs
import kotlin.math.max

/**
 * Crop is inferred from actually reliable landmarks, never from a fixed screen-size threshold.
 * This is deliberately independent of photographic style so every recipe can declare what crops
 * it can operate on.
 */
enum class CropClass {
    FACE,
    HEAD_SHOULDERS,
    CHEST_UP,
    WAIST_UP,
    THREE_QUARTER,
    FULL_BODY,
    ENVIRONMENTAL
}

data class BodyProfile(
    val personId: Int,
    val crop: CropClass,
    val confidence: Float,
    val visibleJoints: Set<Int>,
    val shoulderWidth: Float,
    val hipWidth: Float,
    val torsoLength: Float,
    val leftUpperArm: Float,
    val leftForearm: Float,
    val rightUpperArm: Float,
    val rightForearm: Float,
    val leftThigh: Float,
    val leftShin: Float,
    val rightThigh: Float,
    val rightShin: Float,
    val projectedCenter: Point2,
    val projectedScale: Float
) {
    fun segmentLength(a: Int, b: Int): Float? = when (ordered(a, b)) {
        ordered(11, 12) -> shoulderWidth
        ordered(23, 24) -> hipWidth
        ordered(11, 13) -> leftUpperArm
        ordered(13, 15) -> leftForearm
        ordered(12, 14) -> rightUpperArm
        ordered(14, 16) -> rightForearm
        ordered(23, 25) -> leftThigh
        ordered(25, 27) -> leftShin
        ordered(24, 26) -> rightThigh
        ordered(26, 28) -> rightShin
        else -> null
    }

    private fun ordered(a: Int, b: Int): Long {
        val lo = minOf(a, b)
        val hi = maxOf(a, b)
        return (lo.toLong() shl 32) or hi.toLong()
    }
}

/**
 * Stabilises proportions over time. The user's proportions are intentionally estimated as ratios
 * in image space; no claim is made about real-world height or physical measurements.
 */
class BodyCalibrator(
    private val alpha: Float = 0.16f,
    private val minJointConfidence: Float = 0.42f
) {
    private data class Running(
        var shoulder: Float = Float.NaN,
        var hip: Float = Float.NaN,
        var torso: Float = Float.NaN,
        var lua: Float = Float.NaN,
        var lfa: Float = Float.NaN,
        var rua: Float = Float.NaN,
        var rfa: Float = Float.NaN,
        var lthigh: Float = Float.NaN,
        var lshin: Float = Float.NaN,
        var rthigh: Float = Float.NaN,
        var rshin: Float = Float.NaN,
        var observations: Int = 0
    )

    private val running = mutableMapOf<Int, Running>()

    fun reset() = running.clear()

    fun update(pose: PoseState): BodyProfile {
        val r = running.getOrPut(pose.id) { Running() }
        val p = pose.landmarks
        val visible = p.indices.filterTo(mutableSetOf()) { p[it].confidence >= minJointConfidence }

        fun observed(a: Int, b: Int): Float? {
            if (a !in p.indices || b !in p.indices) return null
            if (p[a].confidence < minJointConfidence || p[b].confidence < minJointConfidence) return null
            val d = p[a].distance(p[b])
            return d.takeIf { it in 0.006f..0.75f }
        }

        fun smooth(old: Float, value: Float?): Float {
            if (value == null || !value.isFinite()) return old
            if (!old.isFinite()) return value
            // Reject violent single-frame scale changes caused by mistracked landmarks.
            val ratio = value / old.coerceAtLeast(0.0001f)
            if (ratio !in 0.62f..1.62f) return old
            return old + alpha * (value - old)
        }

        r.shoulder = smooth(r.shoulder, observed(11, 12))
        r.hip = smooth(r.hip, observed(23, 24))
        val shoulderCenter = midpointOrNull(p, 11, 12, minJointConfidence)
        val hipCenter = midpointOrNull(p, 23, 24, minJointConfidence)
        r.torso = smooth(r.torso, if (shoulderCenter != null && hipCenter != null) shoulderCenter.distance(hipCenter) else null)
        r.lua = smooth(r.lua, observed(11, 13))
        r.lfa = smooth(r.lfa, observed(13, 15))
        r.rua = smooth(r.rua, observed(12, 14))
        r.rfa = smooth(r.rfa, observed(14, 16))
        r.lthigh = smooth(r.lthigh, observed(23, 25))
        r.lshin = smooth(r.lshin, observed(25, 27))
        r.rthigh = smooth(r.rthigh, observed(24, 26))
        r.rshin = smooth(r.rshin, observed(26, 28))
        r.observations++

        val shoulder = fallback(r.shoulder, pose.bbox.width() * 0.42f, 0.08f)
        val torso = fallback(r.torso, max(shoulder * 1.28f, pose.bbox.height() * 0.28f), 0.10f)
        val hip = fallback(r.hip, shoulder * 0.86f, shoulder * 0.72f)
        val upperArm = shoulder * 0.86f
        val forearm = shoulder * 0.80f
        val thigh = shoulder * 1.35f
        val shin = shoulder * 1.30f

        val crop = classifyCrop(pose, visible)
        val confidence = ((visible.size / 13f) * 0.60f + (r.observations.coerceAtMost(12) / 12f) * 0.40f)
            .coerceIn(0f, 1f)

        return BodyProfile(
            personId = pose.id,
            crop = crop,
            confidence = confidence,
            visibleJoints = visible,
            shoulderWidth = shoulder,
            hipWidth = fallback(r.hip, hip, hip),
            torsoLength = torso,
            leftUpperArm = fallback(r.lua, upperArm, upperArm),
            leftForearm = fallback(r.lfa, forearm, forearm),
            rightUpperArm = fallback(r.rua, upperArm, upperArm),
            rightForearm = fallback(r.rfa, forearm, forearm),
            leftThigh = fallback(r.lthigh, thigh, thigh),
            leftShin = fallback(r.lshin, shin, shin),
            rightThigh = fallback(r.rthigh, thigh, thigh),
            rightShin = fallback(r.rshin, shin, shin),
            projectedCenter = Point2(pose.bbox.centerX(), pose.bbox.centerY(), pose.confidence),
            projectedScale = pose.bbox.height().coerceIn(0.05f, 1f)
        )
    }

    private fun classifyCrop(pose: PoseState, visible: Set<Int>): CropClass {
        fun visibleAny(vararg ids: Int) = ids.any { it in visible }
        fun visibleBoth(a: Int, b: Int) = a in visible && b in visible

        val shoulders = visibleBoth(11, 12)
        val hips = visibleBoth(23, 24)
        val knees = visibleAny(25, 26)
        val ankles = visibleAny(27, 28)

        return when {
            ankles && knees && hips -> FullBodyOrEnvironmental(pose)
            knees && hips -> CropClass.THREE_QUARTER
            hips -> CropClass.WAIST_UP
            shoulders && visibleAny(13, 14, 15, 16) -> CropClass.CHEST_UP
            shoulders -> CropClass.HEAD_SHOULDERS
            else -> CropClass.FACE
        }
    }

    private fun FullBodyOrEnvironmental(pose: PoseState): CropClass {
        val margin = minOf(pose.bbox.left, 1f - pose.bbox.right, pose.bbox.top, 1f - pose.bbox.bottom)
        return if (pose.bbox.height() < 0.56f && margin > 0.05f) CropClass.ENVIRONMENTAL else CropClass.FULL_BODY
    }

    private fun fallback(value: Float, first: Float, minimum: Float): Float {
        val v = if (value.isFinite()) value else first
        return max(v, minimum)
    }

    private fun midpointOrNull(points: List<Point2>, a: Int, b: Int, minConfidence: Float): Point2? {
        if (a !in points.indices || b !in points.indices) return null
        val pa = points[a]
        val pb = points[b]
        if (pa.confidence < minConfidence || pb.confidence < minConfidence) return null
        return Point2((pa.x + pb.x) * 0.5f, (pa.y + pb.y) * 0.5f, minOf(pa.confidence, pb.confidence))
    }
}
