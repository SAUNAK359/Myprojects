package com.saunak.director.engine.motion

import kotlin.math.abs
import kotlin.math.hypot

class VisualMotionTracker {
    private var previous: LumaMotionFrame? = null

    @Synchronized
    fun push(frame: LumaMotionFrame): VisualMotionObservation? {
        val prior = previous
        previous = frame
        if (prior == null || prior.width != frame.width || prior.height != frame.height) return null
        val dtNs = frame.sensorTimestampNs - prior.sensorTimestampNs
        if (dtNs <= 1_000_000L || dtNs > 250_000_000L) return null
        val dt = dtNs / 1_000_000_000.0

        val local = ArrayList<Shift>(GRID_X * GRID_Y)
        val cellW = frame.width / GRID_X
        val cellH = frame.height / GRID_Y
        for (gy in 0 until GRID_Y) for (gx in 0 until GRID_X) {
            val cx = gx * cellW + cellW / 2
            val cy = gy * cellH + cellH / 2
            bestShift(prior, frame, cx, cy)?.let(local::add)
        }
        if (local.size < MIN_TRACKS) return null

        val globalDx = medianInt(local.map { it.dx }.sorted()).toDouble()
        val globalDy = medianInt(local.map { it.dy }.sorted()).toDouble()
        val globalSpeed = hypot(globalDx, globalDy) / dt
        val residualSpeeds = local.map { hypot(it.dx.toDouble() - globalDx, it.dy.toDouble() - globalDy) / dt }.sorted()
        val subjectSpeed = percentile(residualSpeeds, 0.82)
        val subjectShift = local.maxByOrNull { hypot(it.dx.toDouble() - globalDx, it.dy.toDouble() - globalDy) }
        val subjectVx = (((subjectShift?.dx?.toDouble()) ?: globalDx) - globalDx) / dt
        val subjectVy = (((subjectShift?.dy?.toDouble()) ?: globalDy) - globalDy) / dt
        val scoreMedian = medianDouble(local.map { it.score }.sorted())
        val textureConfidence = (1.0 - scoreMedian / 48.0).coerceIn(0.0, 1.0)
        val trackConfidence = (local.size / (GRID_X * GRID_Y).toDouble()).coerceIn(0.0, 1.0)

        return VisualMotionObservation(
            MotionVector(globalDx / dt, globalDy / dt),
            MotionVector(subjectVx, subjectVy),
            speedToRisk(globalSpeed),
            speedToRisk(subjectSpeed),
            (0.58 * trackConfidence + 0.42 * textureConfidence).coerceIn(0.0, 1.0),
            frame.sensorTimestampNs,
            local.size
        )
    }

    @Synchronized fun reset() { previous = null }
    private data class Shift(val dx: Int, val dy: Int, val score: Double)

    private fun bestShift(a: LumaMotionFrame, b: LumaMotionFrame, cx: Int, cy: Int): Shift? {
        var best: Shift? = null
        for (dy in -SEARCH_RADIUS..SEARCH_RADIUS) for (dx in -SEARCH_RADIUS..SEARCH_RADIUS) {
            var sad = 0L
            var n = 0
            var y = -PATCH_RADIUS
            while (y <= PATCH_RADIUS) {
                var x = -PATCH_RADIUS
                while (x <= PATCH_RADIUS) {
                    val ax = cx + x; val ay = cy + y; val bx = ax + dx; val by = ay + dy
                    if (ax in 0 until a.width && ay in 0 until a.height && bx in 0 until b.width && by in 0 until b.height) {
                        val av = a.luma[ay * a.width + ax].toInt() and 0xff
                        val bv = b.luma[by * b.width + bx].toInt() and 0xff
                        sad += abs(av - bv); n++
                    }
                    x += SAMPLE_STEP
                }
                y += SAMPLE_STEP
            }
            if (n >= 9) {
                val score = sad.toDouble() / n
                if (best == null || score < best.score) best = Shift(dx, dy, score)
            }
        }
        return best
    }

    private fun speedToRisk(pxPerSecond: Double): Double {
        val normalized = (pxPerSecond / HIGH_MOTION_PX_PER_SECOND).coerceAtLeast(0.0)
        return (normalized / (1.0 + normalized)).coerceIn(0.0, 1.0)
    }
    private fun medianInt(values: List<Int>) = values[values.size / 2]
    private fun medianDouble(values: List<Double>) = values[values.size / 2]
    private fun percentile(values: List<Double>, q: Double): Double = values[((values.size - 1) * q.coerceIn(0.0, 1.0)).toInt()]

    companion object {
        private const val GRID_X = 7
        private const val GRID_Y = 5
        private const val MIN_TRACKS = 12
        private const val SEARCH_RADIUS = 4
        private const val PATCH_RADIUS = 6
        private const val SAMPLE_STEP = 3
        private const val HIGH_MOTION_PX_PER_SECOND = 420.0
    }
}
