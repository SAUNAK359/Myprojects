package com.saunak.director.engine.reconstruction

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/** Translation applied as sample(x + dx, y + dy) to map an observation into reference coordinates. */
data class CfaTranslation(
    val dx: Int,
    val dy: Int,
    val confidence: Float,
    val meanAbsoluteResidual: Float
) {
    init {
        require(dx % 2 == 0 && dy % 2 == 0) { "Bayer-preserving translations must move by whole 2x2 CFA cells" }
        require(confidence in 0f..1f)
        require(meanAbsoluteResidual >= 0f)
    }
}

data class CfaAlignedBurst(
    val observations: List<RawObservation>,
    val translations: List<CfaTranslation>,
    val referenceCropX: Int,
    val referenceCropY: Int
) {
    init {
        require(observations.isNotEmpty())
        require(observations.size == translations.size)
        require(referenceCropX >= 0 && referenceCropY >= 0)
    }
}

/**
 * Portable CFA-safe alignment baseline. It deliberately estimates only even-pixel global
 * translations. Dense/subpixel flow belongs after a geometry stage that explicitly preserves CFA
 * sampling; this baseline never performs arbitrary interpolation on the Bayer mosaic.
 */
object CfaBurstAlignment {
    fun estimate(reference: RawPlane, candidate: RawPlane, searchRadiusPx: Int = 12): CfaTranslation {
        require(reference.width == candidate.width && reference.height == candidate.height)
        require(reference.pattern == candidate.pattern)
        require(searchRadiusPx in 0..64)

        val ref = RawImageFormation.normalize(reference)
        val cand = RawImageFormation.normalize(candidate)
        val radius = searchRadiusPx - (searchRadiusPx and 1)
        var bestDx = 0
        var bestDy = 0
        var best = Double.POSITIVE_INFINITY
        var second = Double.POSITIVE_INFINITY

        for (dy in -radius..radius step 2) {
            for (dx in -radius..radius step 2) {
                val residual = residual(ref, cand, reference.width, reference.height, dx, dy)
                if (residual < best) {
                    second = best
                    best = residual
                    bestDx = dx
                    bestDy = dy
                } else if (residual < second) {
                    second = residual
                }
            }
        }

        val separation = if (second.isFinite() && second > 0.0) ((second - best) / second).coerceIn(0.0, 1.0) else 0.0
        val absoluteQuality = exp(-best * 18.0).coerceIn(0.0, 1.0)
        val confidence = (0.55 * absoluteQuality + 0.45 * separation).toFloat().coerceIn(0f..1f)
        return CfaTranslation(bestDx, bestDy, confidence, best.toFloat().coerceAtLeast(0f))
    }

    fun align(observations: List<RawObservation>, searchRadiusPx: Int = 12): CfaAlignedBurst {
        require(observations.isNotEmpty())
        val reference = observations.first().raw
        require(observations.all {
            it.raw.width == reference.width && it.raw.height == reference.height && it.raw.pattern == reference.pattern
        })

        val translations = observations.mapIndexed { index, observation ->
            if (index == 0) CfaTranslation(0, 0, 1f, 0f)
            else estimate(reference, observation.raw, searchRadiusPx)
        }

        var left = 0
        var top = 0
        var right = reference.width
        var bottom = reference.height
        for (t in translations) {
            left = max(left, -t.dx)
            top = max(top, -t.dy)
            right = min(right, reference.width - t.dx)
            bottom = min(bottom, reference.height - t.dy)
        }

        if ((left and 1) != 0) left++
        if ((top and 1) != 0) top++
        if (((right - left) and 1) != 0) right--
        if (((bottom - top) and 1) != 0) bottom--
        require(right - left >= 4 && bottom - top >= 4) { "Burst overlap too small after alignment" }

        val width = right - left
        val height = bottom - top
        val aligned = observations.mapIndexed { index, observation ->
            val t = translations[index]
            val values = FloatArray(width * height)
            for (y in 0 until height) {
                val sy = y + top + t.dy
                for (x in 0 until width) {
                    val sx = x + left + t.dx
                    values[y * width + x] = observation.raw[sx, sy]
                }
            }
            observation.copy(
                raw = RawPlane(
                    width = width,
                    height = height,
                    values = values,
                    pattern = reference.pattern,
                    blackLevel = observation.raw.blackLevel,
                    whiteLevel = observation.raw.whiteLevel,
                    blackLevelPattern = observation.raw.blackLevelPattern
                ),
                alignmentConfidence = observation.alignmentConfidence * t.confidence
            )
        }
        return CfaAlignedBurst(aligned, translations, left, top)
    }

    private fun residual(
        ref: FloatArray,
        candidate: FloatArray,
        width: Int,
        height: Int,
        dx: Int,
        dy: Int
    ): Double {
        val border = 16
        val x0 = max(border, -dx)
        val y0 = max(border, -dy)
        val x1 = min(width - border, width - dx)
        val y1 = min(height - border, height - dy)
        if (x1 <= x0 || y1 <= y0) return Double.POSITIVE_INFINITY

        var error = 0.0
        var count = 0
        val stride = 8
        var y = y0 + (y0 and 1)
        while (y < y1) {
            var x = x0 + (x0 and 1)
            while (x < x1) {
                val a = ref[y * width + x]
                val b = candidate[(y + dy) * width + x + dx]
                error += min(abs(a - b).toDouble(), 0.20)
                count++
                x += stride
            }
            y += stride
        }
        return if (count > 0) error / count else Double.POSITIVE_INFINITY
    }
}
