package com.saunak.director.engine.motion

import com.saunak.director.engine.subject.NormalizedRect
import kotlin.math.abs
import kotlin.math.hypot

/**
 * Allocation-bounded translational motion estimator for the preview stream. It estimates robust
 * global image motion first, then measures residual motion preferentially inside the persistent
 * primary-subject region so unrelated moving objects do not masquerade as subject motion.
 */
class VisualMotionTracker {
    private var previous: LumaMotionFrame? = null

    @Synchronized
    fun push(frame: LumaMotionFrame, subjectBounds: NormalizedRect? = null): VisualMotionObservation? {
        val prior = previous
        previous = frame
        if (prior == null || prior.width != frame.width || prior.height != frame.height) return null
        val dtNs = frame.sensorTimestampNs - prior.sensorTimestampNs
        if (dtNs <= 1_000_000L || dtNs > 250_000_000L) return null
        val dt = dtNs / 1_000_000_000.0

        val local = ArrayList<Shift>(GRID_X * GRID_Y)
        val cellW = frame.width / GRID_X
        val cellH = frame.height / GRID_Y
        for (gy in 0 until GRID_Y) {
            for (gx in 0 until GRID_X) {
                val cx = gx * cellW + cellW / 2
                val cy = gy * cellH + cellH / 2
                bestShift(prior, frame, cx, cy)?.let(local::add)
            }
        }
        if (local.size < MIN_TRACKS) return null

        val dxs = local.map { it.dx }.sorted()
        val dys = local.map { it.dy }.sorted()
        val globalDx = medianInt(dxs).toDouble()
        val globalDy = medianInt(dys).toDouble()
        val globalSpeed = hypot(globalDx, globalDy) / dt

        val subjectCandidates = subjectBounds?.let { bounds ->
            local.filter { shift ->
                val nx = shift.cx.toDouble() / frame.width.toDouble()
                val ny = shift.cy.toDouble() / frame.height.toDouble()
                nx in bounds.left..bounds.right && ny in bounds.top..bounds.bottom
            }
        }.orEmpty()
        val roiSupported = subjectCandidates.size >= MIN_SUBJECT_TRACKS
        val subjectLocal = if (roiSupported) subjectCandidates else local

        val residualSpeeds = subjectLocal.map { shift ->
            hypot(shift.dx.toDouble() - globalDx, shift.dy.toDouble() - globalDy) / dt
        }.sorted()
        val subjectSpeed = percentile(residualSpeeds, if (roiSupported) 0.68 else 0.82)
        val subjectShift = subjectLocal.maxByOrNull { shift ->
            hypot(shift.dx.toDouble() - globalDx, shift.dy.toDouble() - globalDy)
        }
        val subjectDx = subjectShift?.dx?.toDouble() ?: globalDx
        val subjectDy = subjectShift?.dy?.toDouble() ?: globalDy
        val subjectVx = (subjectDx - globalDx) / dt
        val subjectVy = (subjectDy - globalDy) / dt

        val scoreMedian = medianDouble(local.map { it.score }.sorted())
        val textureConfidence = (1.0 - scoreMedian / 48.0).coerceIn(0.0, 1.0)
        val trackConfidence = (local.size / (GRID_X * GRID_Y).toDouble()).coerceIn(0.0, 1.0)
        val confidence = (0.58 * trackConfidence + 0.42 * textureConfidence).coerceIn(0.0, 1.0)
        val roiCoverage = if (subjectBounds != null) {
            (subjectCandidates.size / maxOf(1.0, local.size.toDouble() * subjectBounds.area.coerceAtLeast(0.08))).coerceIn(0.0, 1.0)
        } else 0.0
        val subjectConfidence = if (roiSupported) {
            (confidence * (0.68 + 0.32 * roiCoverage)).coerceIn(0.0, 1.0)
        } else {
            // Whole-frame residual remains a useful fallback but cannot claim subject identity.
            (confidence * 0.42).coerceIn(0.0, 1.0)
        }

        return VisualMotionObservation(
            globalVector = MotionVector(globalDx / dt, globalDy / dt),
            residualSubjectVector = MotionVector(subjectVx, subjectVy),
            globalRisk = speedToRisk(globalSpeed),
            subjectRisk = speedToRisk(subjectSpeed),
            confidence = confidence,
            frameTimestampNs = frame.sensorTimestampNs,
            trackCount = local.size,
            subjectConfidence = subjectConfidence
        )
    }

    @Synchronized fun reset() { previous = null }

    private data class Shift(val dx: Int, val dy: Int, val score: Double, val cx: Int, val cy: Int)

    private fun bestShift(a: LumaMotionFrame, b: LumaMotionFrame, cx: Int, cy: Int): Shift? {
        var best: Shift? = null
        for (dy in -SEARCH_RADIUS..SEARCH_RADIUS) {
            for (dx in -SEARCH_RADIUS..SEARCH_RADIUS) {
                var sad = 0L
                var n = 0
                var y = -PATCH_RADIUS
                while (y <= PATCH_RADIUS) {
                    var x = -PATCH_RADIUS
                    while (x <= PATCH_RADIUS) {
                        val ax = cx + x
                        val ay = cy + y
                        val bx = ax + dx
                        val by = ay + dy
                        if (ax in 0 until a.width && ay in 0 until a.height && bx in 0 until b.width && by in 0 until b.height) {
                            val av = a.luma[ay * a.width + ax].toInt() and 0xff
                            val bv = b.luma[by * b.width + bx].toInt() and 0xff
                            sad += abs(av - bv)
                            n++
                        }
                        x += SAMPLE_STEP
                    }
                    y += SAMPLE_STEP
                }
                if (n < 9) continue
                val score = sad.toDouble() / n
                if (best == null || score < best.score) best = Shift(dx, dy, score, cx, cy)
            }
        }
        return best
    }

    private fun speedToRisk(pxPerSecond: Double): Double {
        val normalized = (pxPerSecond / HIGH_MOTION_PX_PER_SECOND).coerceAtLeast(0.0)
        return (normalized / (1.0 + normalized)).coerceIn(0.0, 1.0)
    }

    private fun medianInt(values: List<Int>): Int = values[values.size / 2]
    private fun medianDouble(values: List<Double>): Double = values[values.size / 2]
    private fun percentile(values: List<Double>, q: Double): Double {
        if (values.isEmpty()) return 0.0
        val index = ((values.size - 1) * q.coerceIn(0.0, 1.0)).toInt()
        return values[index]
    }

    companion object {
        private const val GRID_X = 7
        private const val GRID_Y = 5
        private const val MIN_TRACKS = 12
        private const val MIN_SUBJECT_TRACKS = 2
        private const val SEARCH_RADIUS = 4
        private const val PATCH_RADIUS = 6
        private const val SAMPLE_STEP = 3
        private const val HIGH_MOTION_PX_PER_SECOND = 420.0
    }
}
