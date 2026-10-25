package com.saunak.director.engine.subject

import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/** Coordinates in the oriented analysis frame, normalized to [0,1]. */
data class NormalizedPoint(val x: Double, val y: Double) {
    init { require(x.isFinite() && y.isFinite()) }
    fun clamped() = NormalizedPoint(x.coerceIn(0.0, 1.0), y.coerceIn(0.0, 1.0))
    fun distanceTo(other: NormalizedPoint): Double = hypot(x - other.x, y - other.y)
}

data class NormalizedRect(
    val left: Double,
    val top: Double,
    val right: Double,
    val bottom: Double
) {
    init {
        require(listOf(left, top, right, bottom).all(Double::isFinite))
        require(right >= left && bottom >= top)
    }

    val width: Double get() = (right - left).coerceAtLeast(0.0)
    val height: Double get() = (bottom - top).coerceAtLeast(0.0)
    val area: Double get() = width * height
    val center: NormalizedPoint get() = NormalizedPoint((left + right) * 0.5, (top + bottom) * 0.5)

    fun contains(x: Double, y: Double): Boolean =
        x.isFinite() && y.isFinite() && x >= left && x <= right && y >= top && y <= bottom

    fun contains(point: NormalizedPoint): Boolean = contains(point.x, point.y)

    fun clamped(): NormalizedRect {
        val l = left.coerceIn(0.0, 1.0)
        val t = top.coerceIn(0.0, 1.0)
        val r = right.coerceIn(l, 1.0)
        val b = bottom.coerceIn(t, 1.0)
        return NormalizedRect(l, t, r, b)
    }

    fun expanded(scaleX: Double, scaleY: Double): NormalizedRect {
        require(scaleX > 0.0 && scaleY > 0.0)
        val c = center
        val halfW = width * scaleX * 0.5
        val halfH = height * scaleY * 0.5
        return NormalizedRect(c.x - halfW, c.y - halfH, c.x + halfW, c.y + halfH).clamped()
    }

    fun iou(other: NormalizedRect): Double {
        val x1 = max(left, other.left)
        val y1 = max(top, other.top)
        val x2 = min(right, other.right)
        val y2 = min(bottom, other.bottom)
        val intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        val union = area + other.area - intersection
        return if (union <= 1e-9) 0.0 else (intersection / union).coerceIn(0.0, 1.0)
    }

    companion object {
        fun fromPixels(x: Double, y: Double, width: Double, height: Double, frameWidth: Int, frameHeight: Int): NormalizedRect {
            require(frameWidth > 0 && frameHeight > 0)
            require(width >= 0.0 && height >= 0.0)
            return NormalizedRect(
                x / frameWidth,
                y / frameHeight,
                (x + width) / frameWidth,
                (y + height) / frameHeight
            ).clamped()
        }
    }
}

data class PoseKeypoint(
    val index: Int,
    val position: NormalizedPoint,
    val confidence: Double
) {
    init { require(index >= 0); require(confidence in 0.0..1.0) }
}

data class SubjectPoseState(
    val keypoints: List<PoseKeypoint>,
    val bounds: NormalizedRect,
    val confidence: Double,
    val source: String
) {
    init {
        require(keypoints.isNotEmpty())
        require(confidence in 0.0..1.0)
        require(source.isNotBlank())
    }
}

data class SubjectMaskState(
    val width: Int,
    val height: Int,
    val probabilities: FloatArray,
    val bounds: NormalizedRect,
    val centroid: NormalizedPoint,
    val coverage: Double,
    val confidence: Double,
    val source: String
) {
    init {
        require(width > 0 && height > 0)
        require(probabilities.size == width * height)
        require(probabilities.all { it.isFinite() && it in 0f..1f })
        require(coverage in 0.0..1.0)
        require(confidence in 0.0..1.0)
        require(source.isNotBlank())
    }
}

data class SubjectDepthState(
    /** Relative inverse depth only: larger means closer. Never metres. */
    val relativeInverseDepth: Double,
    val foregroundSeparation: Double,
    val confidence: Double,
    val source: String
) {
    init {
        require(relativeInverseDepth in 0.0..1.0)
        require(foregroundSeparation in -1.0..1.0)
        require(confidence in 0.0..1.0)
        require(source.isNotBlank())
    }
}

data class SubjectObservation(
    val observedAtNs: Long,
    val cameraId: String?,
    val bounds: NormalizedRect,
    val confidence: Double,
    val faceConfidence: Double? = null,
    val mask: SubjectMaskState? = null,
    val pose: SubjectPoseState? = null,
    val depth: SubjectDepthState? = null,
    val source: String
) {
    init {
        require(observedAtNs > 0)
        require(confidence in 0.0..1.0)
        require(faceConfidence == null || faceConfidence in 0.0..1.0)
        require(source.isNotBlank())
    }
}

data class SubjectKinematics(
    val velocityXPerSec: Double = 0.0,
    val velocityYPerSec: Double = 0.0,
    val accelerationXPerSec2: Double = 0.0,
    val accelerationYPerSec2: Double = 0.0,
    val predictedCenter: NormalizedPoint,
    val predictionHorizonMs: Long
) {
    init { require(predictionHorizonMs >= 0) }
    val speedPerSec: Double get() = hypot(velocityXPerSec, velocityYPerSec)
    val accelerationPerSec2: Double get() = hypot(accelerationXPerSec2, accelerationYPerSec2)
}

data class SubjectTrack(
    val id: Long,
    val cameraId: String?,
    val bounds: NormalizedRect,
    val center: NormalizedPoint,
    val kinematics: SubjectKinematics,
    val confidence: Double,
    val uncertainty: Double,
    val firstSeenNs: Long,
    val lastSeenNs: Long,
    val updateCount: Int,
    val missedUpdates: Int,
    val mask: SubjectMaskState?,
    val pose: SubjectPoseState?,
    val depth: SubjectDepthState?,
    val source: String
) {
    init {
        require(id > 0)
        require(confidence in 0.0..1.0)
        require(uncertainty in 0.0..1.0)
        require(firstSeenNs > 0 && lastSeenNs >= firstSeenNs)
        require(updateCount > 0 && missedUpdates >= 0)
        require(source.isNotBlank())
    }

    val predictedCenter: NormalizedPoint get() = kinematics.predictedCenter
}
