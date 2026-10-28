package com.saunak.director.director.live

import android.graphics.Bitmap
import com.saunak.director.engine.subject.NormalizedRect
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Deterministic generic primary-entity anchor for non-human and unclassified subjects.
 * Gemini identifies the entity once by current_box. The runtime stores a compact luminance template
 * and tracks that exact region locally with bounded multi-scale search. It never changes semantic identity.
 */
class VisualEntityAnchor {
    data class Observation(
        val bounds: NormalizedRect,
        val confidence: Double,
        val visibility: Double,
        val source: String = "director_template_anchor"
    )

    private data class Template(val width: Int, val height: Int, val luma: FloatArray)
    private data class Candidate(val bounds: NormalizedRect, val score: Double)

    private var template: Template? = null
    private var lastBounds: NormalizedRect? = null
    private var missCount: Int = 0

    fun initialize(frame: Bitmap, box: DirectorRect): Observation {
        require(!frame.isRecycled && frame.width > 0 && frame.height > 0)
        val bounds = box.normalizedRect().clamped()
        template = extractTemplate(frame, bounds)
        lastBounds = bounds
        missCount = 0
        return Observation(bounds, confidence = 1.0, visibility = visibility(bounds))
    }

    fun track(frame: Bitmap): Observation? {
        val t = template ?: return null
        val previous = lastBounds ?: return null
        if (frame.isRecycled || frame.width <= 0 || frame.height <= 0) return null

        val local = search(
            frame = frame,
            template = t,
            previous = previous,
            scales = LOCAL_SCALES,
            searchSteps = LOCAL_SEARCH_STEPS,
            stepFraction = LOCAL_STEP_FRACTION
        )
        val selected = if (local != null && adjustedConfidence(previous, local) >= MIN_ACCEPT_CONFIDENCE) {
            local
        } else {
            // Camera controls can alter apparent scale immediately after Gemini's planning frame.
            // A bounded coarse pass is used only after local tracking loses confidence, so normal
            // frames stay cheap while zoom/reframe transitions remain recoverable.
            search(
                frame = frame,
                template = t,
                previous = previous,
                scales = REACQUIRE_SCALES,
                searchSteps = REACQUIRE_SEARCH_STEPS,
                stepFraction = REACQUIRE_STEP_FRACTION
            )
        } ?: return miss()

        val confidence = adjustedConfidence(previous, selected)
        if (confidence < REACQUIRE_MIN_CONFIDENCE) return miss()

        missCount = 0
        val alpha = if (confidence >= 0.72) SMOOTH_ALPHA else RECOVERY_SMOOTH_ALPHA
        lastBounds = smooth(previous, selected.bounds, alpha)
        return Observation(
            bounds = lastBounds!!,
            confidence = confidence,
            visibility = visibility(lastBounds!!),
            source = if (local === selected) "director_template_anchor" else "director_template_anchor_reacquired"
        )
    }

    fun reset() {
        template = null
        lastBounds = null
        missCount = 0
    }

    fun latestBounds(): NormalizedRect? = lastBounds

    private fun search(
        frame: Bitmap,
        template: Template,
        previous: NormalizedRect,
        scales: DoubleArray,
        searchSteps: Int,
        stepFraction: Double
    ): Candidate? {
        var best: Candidate? = null
        for (scale in scales) {
            val w = (previous.width * scale).coerceIn(MIN_BOX_FRACTION, 0.95)
            val h = (previous.height * scale).coerceIn(MIN_BOX_FRACTION, 0.95)
            val cx0 = previous.center.x
            val cy0 = previous.center.y
            for (dy in -searchSteps..searchSteps) {
                for (dx in -searchSteps..searchSteps) {
                    val cx = cx0 + dx * previous.width * stepFraction
                    val cy = cy0 + dy * previous.height * stepFraction
                    val rect = NormalizedRect(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0).clamped()
                    if (rect.width < MIN_BOX_FRACTION || rect.height < MIN_BOX_FRACTION) continue
                    val score = similarity(frame, rect, template)
                    if (best == null || score > best!!.score) best = Candidate(rect, score)
                }
            }
        }
        return best
    }

    private fun adjustedConfidence(previous: NormalizedRect, candidate: Candidate): Double {
        val displacement = previous.center.distanceTo(candidate.bounds.center)
        val motionPenalty = (displacement / MAX_EXPECTED_REACQUIRE_DISTANCE).coerceIn(0.0, 1.0)
        val scaleRatio = max(
            candidate.bounds.width / max(previous.width, 1e-6),
            previous.width / max(candidate.bounds.width, 1e-6)
        )
        val scalePenalty = ((scaleRatio - 1.0) / 1.2).coerceIn(0.0, 0.45)
        return (candidate.score * (1.0 - 0.22 * motionPenalty - 0.18 * scalePenalty)).coerceIn(0.0, 1.0)
    }

    private fun miss(): Observation? {
        missCount++
        if (missCount > MAX_MISSES) {
            lastBounds = null
            return null
        }
        val bounds = lastBounds ?: return null
        val confidence = (0.42 * (1.0 - missCount.toDouble() / (MAX_MISSES + 1))).coerceIn(0.0, 1.0)
        return Observation(bounds, confidence, visibility(bounds), source = "director_template_anchor_predicted")
    }

    private fun extractTemplate(frame: Bitmap, bounds: NormalizedRect): Template {
        val width = TEMPLATE_WIDTH
        val height = ((bounds.height / max(bounds.width, 1e-6)) * width).toInt().coerceIn(MIN_TEMPLATE_HEIGHT, MAX_TEMPLATE_HEIGHT)
        return Template(width, height, sampleLuma(frame, bounds, width, height))
    }

    private fun similarity(frame: Bitmap, bounds: NormalizedRect, template: Template): Double {
        val sample = sampleLuma(frame, bounds, template.width, template.height)
        var meanA = 0.0
        var meanB = 0.0
        for (i in sample.indices) {
            meanA += sample[i]
            meanB += template.luma[i]
        }
        meanA /= sample.size
        meanB /= sample.size

        var numerator = 0.0
        var denomA = 0.0
        var denomB = 0.0
        for (i in sample.indices) {
            val a = sample[i] - meanA
            val b = template.luma[i] - meanB
            numerator += a * b
            denomA += a * a
            denomB += b * b
        }
        if (denomA <= 1e-6 || denomB <= 1e-6) {
            var mad = 0.0
            for (i in sample.indices) mad += abs(sample[i] - template.luma[i])
            return (1.0 - mad / sample.size / 255.0).coerceIn(0.0, 1.0)
        }
        val ncc = numerator / sqrt(denomA * denomB)
        return ((ncc + 1.0) * 0.5).coerceIn(0.0, 1.0)
    }

    private fun sampleLuma(frame: Bitmap, bounds: NormalizedRect, outW: Int, outH: Int): FloatArray {
        val output = FloatArray(outW * outH)
        val x0 = bounds.left * (frame.width - 1)
        val y0 = bounds.top * (frame.height - 1)
        val xSpan = max(1.0, bounds.width * (frame.width - 1))
        val ySpan = max(1.0, bounds.height * (frame.height - 1))
        var idx = 0
        for (y in 0 until outH) {
            val sy = (y0 + (y + 0.5) / outH * ySpan).toInt().coerceIn(0, frame.height - 1)
            for (x in 0 until outW) {
                val sx = (x0 + (x + 0.5) / outW * xSpan).toInt().coerceIn(0, frame.width - 1)
                val p = frame.getPixel(sx, sy)
                val r = (p shr 16) and 0xff
                val g = (p shr 8) and 0xff
                val b = p and 0xff
                output[idx++] = ((77 * r + 150 * g + 29 * b) shr 8).toFloat()
            }
        }
        return output
    }

    private fun smooth(a: NormalizedRect, b: NormalizedRect, alpha: Double): NormalizedRect {
        fun blend(x: Double, y: Double) = x * (1.0 - alpha) + y * alpha
        return NormalizedRect(blend(a.left,b.left), blend(a.top,b.top), blend(a.right,b.right), blend(a.bottom,b.bottom)).clamped()
    }

    private fun visibility(bounds: NormalizedRect): Double {
        val marginLoss = listOf(bounds.left, bounds.top, 1.0 - bounds.right, 1.0 - bounds.bottom)
            .sumOf { if (it >= EDGE_SAFE_MARGIN) 0.0 else (EDGE_SAFE_MARGIN - it) / EDGE_SAFE_MARGIN }
        return (1.0 - marginLoss * 0.18).coerceIn(0.0, 1.0)
    }

    companion object {
        private const val TEMPLATE_WIDTH = 40
        private const val MIN_TEMPLATE_HEIGHT = 24
        private const val MAX_TEMPLATE_HEIGHT = 56
        private const val LOCAL_SEARCH_STEPS = 2
        private const val LOCAL_STEP_FRACTION = 0.12
        private const val REACQUIRE_SEARCH_STEPS = 3
        private const val REACQUIRE_STEP_FRACTION = 0.19
        private const val MIN_ACCEPT_CONFIDENCE = 0.44
        private const val REACQUIRE_MIN_CONFIDENCE = 0.48
        private const val MAX_EXPECTED_REACQUIRE_DISTANCE = 0.42
        private const val MAX_MISSES = 6
        private const val SMOOTH_ALPHA = 0.68
        private const val RECOVERY_SMOOTH_ALPHA = 0.82
        private const val MIN_BOX_FRACTION = 0.025
        private const val EDGE_SAFE_MARGIN = 0.018
        private val LOCAL_SCALES = doubleArrayOf(0.90, 1.0, 1.10)
        private val REACQUIRE_SCALES = doubleArrayOf(0.65, 0.80, 1.0, 1.25, 1.50)
    }
}
