package com.saunak.director.processing

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import kotlin.math.max

/**
 * Final perceptual contrast guard for Sierra output.
 *
 * ANIP may legitimately recover shadows and compress highlights; on some scenes that can leave the
 * display-referred JPEG looking flatter than the recovered evidence deserves. This finisher measures
 * the rendered image and expands tone separation only when percentile span / luma deviation say the
 * image is flat. The transform pivots around the image median and is suppressed when clipping risk is
 * already high, so it does not blindly crush blacks or blow highlights.
 */
object AdaptiveContrastFinisher {
    fun apply(source: Bitmap): Bitmap {
        if (source.isRecycled) return source
        val q = runCatching { PhotoQuality.analyze(source) }.getOrNull() ?: return source
        val span = (q.highlightP95 - q.shadowP05).coerceAtLeast(1.0)
        val spanNeed = ((TARGET_P95_P05_SPAN - span) / TARGET_P95_P05_SPAN).coerceIn(0.0, 1.0)
        val stdNeed = ((TARGET_LUMA_STD - q.lumaStdDev) / TARGET_LUMA_STD).coerceIn(0.0, 1.0)
        val need = max(spanNeed, stdNeed)
        if (need < MIN_NEED) return source

        val clipRisk = max(q.darkClipFraction, q.brightClipFraction)
        val headroom = (1.0 - clipRisk * 12.0).coerceIn(0.0, 1.0)
        val gain = (1.0 + MAX_GAIN * need * headroom).coerceIn(1.0, 1.16)
        if (gain <= 1.005) return source

        val pivot = q.medianLuma.coerceIn(42.0, 214.0).toFloat()
        val offset = (pivot * (1.0 - gain)).toFloat()
        val matrix = ColorMatrix(
            floatArrayOf(
                gain.toFloat(), 0f, 0f, 0f, offset,
                0f, gain.toFloat(), 0f, 0f, offset,
                0f, 0f, gain.toFloat(), 0f, offset,
                0f, 0f, 0f, 1f, 0f
            )
        )

        val out = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
        Canvas(out).drawBitmap(
            source,
            0f,
            0f,
            Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
                colorFilter = ColorMatrixColorFilter(matrix)
            }
        )
        return out
    }

    internal fun recommendedGain(q: TechnicalQuality): Double {
        val span = (q.highlightP95 - q.shadowP05).coerceAtLeast(1.0)
        val spanNeed = ((TARGET_P95_P05_SPAN - span) / TARGET_P95_P05_SPAN).coerceIn(0.0, 1.0)
        val stdNeed = ((TARGET_LUMA_STD - q.lumaStdDev) / TARGET_LUMA_STD).coerceIn(0.0, 1.0)
        val need = max(spanNeed, stdNeed)
        if (need < MIN_NEED) return 1.0
        val clipRisk = max(q.darkClipFraction, q.brightClipFraction)
        val headroom = (1.0 - clipRisk * 12.0).coerceIn(0.0, 1.0)
        return (1.0 + MAX_GAIN * need * headroom).coerceIn(1.0, 1.16)
    }

    private const val TARGET_P95_P05_SPAN = 154.0
    private const val TARGET_LUMA_STD = 44.0
    private const val MIN_NEED = 0.08
    private const val MAX_GAIN = 0.16
}
