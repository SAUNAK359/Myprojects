package com.saunak.director.processing

import android.graphics.Bitmap
import kotlin.math.max

/**
 * Purely technical post-render release gate. This never judges style or creative strength; Gemini's
 * Render Spec remains the creative authority. It only catches catastrophic implementation/output
 * failures before pixels are committed to MediaStore.
 */
data class RenderSafetyDecision(
    val accepted: Boolean,
    val code: String,
    val before: TechnicalQuality,
    val after: TechnicalQuality
) {
    fun compactDescription(): String = buildString {
        append(if (accepted) "render_verified" else "render_rejected")
        append(':').append(code)
        append(";before=").append("%.3f".format(java.util.Locale.US, before.heuristicScore))
        append(";after=").append("%.3f".format(java.util.Locale.US, after.heuristicScore))
    }
}

object RenderSafetyGate {
    fun evaluate(beforeBitmap: Bitmap, renderedBitmap: Bitmap): RenderSafetyDecision {
        require(!beforeBitmap.isRecycled && !renderedBitmap.isRecycled)
        val before = PhotoQuality.analyze(beforeBitmap)
        val after = PhotoQuality.analyze(renderedBitmap)
        val code = failureCode(beforeBitmap, renderedBitmap, before, after)
        return RenderSafetyDecision(code == null, code ?: "pass", before, after)
    }

    private fun failureCode(
        beforeBitmap: Bitmap,
        afterBitmap: Bitmap,
        before: TechnicalQuality,
        after: TechnicalQuality
    ): String? {
        if (afterBitmap.width <= 0 || afterBitmap.height <= 0) return "invalid_dimensions"
        if (beforeBitmap.width != afterBitmap.width || beforeBitmap.height != afterBitmap.height) return "geometry_changed"
        if (after.catastrophic) return "catastrophic_output"

        // These are deliberately extreme technical limits. Intentional low-key/high-key photography
        // remains valid; only near-total clipping introduced by rendering is rejected.
        if (after.darkClipFraction > 0.985 && after.darkClipFraction > before.darkClipFraction + 0.35) return "black_collapse"
        if (after.brightClipFraction > 0.985 && after.brightClipFraction > before.brightClipFraction + 0.35) return "white_collapse"

        if (before.medianLuma >= 18.0 && after.medianLuma < 3.0) return "luminance_collapse_dark"
        if (before.medianLuma <= 237.0 && after.medianLuma > 252.0) return "luminance_collapse_bright"

        // Catch implementation failures that destroy essentially all spatial information. This is
        // not a sharpness preference: the threshold requires an extreme loss from an image that had
        // measurable structure before rendering.
        if (before.edgeEnergy >= 2.0 && after.edgeEnergy < max(0.20, before.edgeEnergy * 0.10)) return "detail_collapse"

        return null
    }
}
