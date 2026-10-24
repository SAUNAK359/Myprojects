package com.saunak.director.engine.experts

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

data class SubjectDepthEvidence(
    val subjectInverseDepth: Double,
    val backgroundInverseDepth: Double,
    val separation: Double,
    val confidence: Double,
    val source: String = "depth_anything_v2_small_relative"
) {
    init {
        require(subjectInverseDepth in 0.0..1.0)
        require(backgroundInverseDepth in 0.0..1.0)
        require(separation in 0.0..1.0)
        require(confidence in 0.0..1.0)
    }
}

/** Converts relative depth into one safe photographic signal: foreground/background separation. */
object DepthSceneInterpreter {
    fun subjectSeparation(
        depth: DepthAnythingV2Expert.RelativeDepthMap,
        faces: List<FaceDetection>
    ): SubjectDepthEvidence? {
        if (faces.isEmpty() || depth.confidence < 0.25) return null
        val primary = faces.maxByOrNull { it.area * it.score } ?: return null

        val scaleX = depth.width.toDouble() / depth.sourceWidth.toDouble()
        val scaleY = depth.height.toDouble() / depth.sourceHeight.toDouble()
        val left = (primary.x * scaleX).toInt().coerceIn(0, depth.width - 1)
        val top = (primary.y * scaleY).toInt().coerceIn(0, depth.height - 1)
        val right = ((primary.x + primary.width) * scaleX).toInt().coerceIn(left + 1, depth.width)
        val bottom = ((primary.y + primary.height) * scaleY).toInt().coerceIn(top + 1, depth.height)

        val subject = robustMean(depth, left, top, right, bottom)
        val marginX = max(2, (right - left) / 2)
        val marginY = max(2, (bottom - top) / 2)
        val outerLeft = max(0, left - marginX)
        val outerTop = max(0, top - marginY)
        val outerRight = min(depth.width, right + marginX)
        val outerBottom = min(depth.height, bottom + marginY)
        val background = robustRingMean(depth, outerLeft, outerTop, outerRight, outerBottom, left, top, right, bottom)
            ?: return null

        val separation = (subject - background).coerceIn(0.0, 1.0)
        // Only positive foreground separation is photographically useful for portrait depth cues.
        val geometricStrength = (separation / 0.28).coerceIn(0.0, 1.0)
        val faceSupport = (primary.score.toDouble() * (primary.area /
            (depth.sourceWidth.toFloat() * depth.sourceHeight.toFloat())).toDouble().coerceAtMost(0.25) * 4.0)
            .coerceIn(0.0, 1.0)
        val confidence = (depth.confidence * (0.65 * geometricStrength + 0.35 * faceSupport)).coerceIn(0.0, 1.0)

        return SubjectDepthEvidence(subject, background, separation, confidence)
    }

    private fun robustMean(
        depth: DepthAnythingV2Expert.RelativeDepthMap,
        left: Int,
        top: Int,
        right: Int,
        bottom: Int
    ): Double {
        var sum = 0.0
        var count = 0
        val xInset = max(0, (right - left) / 6)
        val yInset = max(0, (bottom - top) / 6)
        for (y in top + yInset until max(top + yInset + 1, bottom - yInset)) {
            for (x in left + xInset until max(left + xInset + 1, right - xInset)) {
                sum += depth.inverseDepth[y * depth.width + x]
                count++
            }
        }
        return if (count > 0) (sum / count).coerceIn(0.0, 1.0) else 0.5
    }

    private fun robustRingMean(
        depth: DepthAnythingV2Expert.RelativeDepthMap,
        outerLeft: Int,
        outerTop: Int,
        outerRight: Int,
        outerBottom: Int,
        innerLeft: Int,
        innerTop: Int,
        innerRight: Int,
        innerBottom: Int
    ): Double? {
        var sum = 0.0
        var count = 0
        val stride = 2
        var y = outerTop
        while (y < outerBottom) {
            var x = outerLeft
            while (x < outerRight) {
                val inside = x in innerLeft until innerRight && y in innerTop until innerBottom
                if (!inside) {
                    sum += depth.inverseDepth[y * depth.width + x]
                    count++
                }
                x += stride
            }
            y += stride
        }
        return if (count >= 16) (sum / count).coerceIn(0.0, 1.0) else null
    }
}
