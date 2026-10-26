package com.saunak.director.processing

import android.graphics.Bitmap
import com.saunak.director.director.FinalRenderAdjustment
import com.saunak.director.director.PhotographicDirection
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Applies the semantic/artistic part of Sierra's latched PhotographicDirection after evidence-safe
 * reconstruction/ANIP. This is deliberately deterministic and bounded: Gemini chooses intent, while
 * Sierra converts it into a reproducible tone/color transform with clipping protection.
 */
object PhotographicDirectionFinisher {
    fun apply(
        source: Bitmap,
        direction: PhotographicDirection?,
        adjustment: FinalRenderAdjustment? = null
    ): Bitmap {
        if (source.isRecycled) return source
        val d = direction ?: return AdaptiveContrastFinisher.apply(source)
        val a = adjustment ?: FinalRenderAdjustment()
        val q = runCatching { PhotoQuality.analyze(source) }.getOrNull() ?: return source

        val themeStrength = (d.themeStrength + a.themeStrengthDelta).coerceIn(0.0, 1.0) * d.themeConfidence.coerceIn(0.0, 1.0)
        val midtoneIntent = (d.midtoneContrast + a.midtoneContrastDelta).coerceIn(0.0, 1.0)
        val localIntent = (d.localContrast + a.localContrastDelta).coerceIn(0.0, 1.0)
        val shadowDepth = (d.shadowDepth + a.shadowDepthDelta).coerceIn(0.0, 1.0)
        val highlightProtection = (d.highlightProtection + a.highlightProtectionDelta).coerceIn(0.0, 1.0)
        val warmth = (d.warmthBias + a.warmthDelta).coerceIn(-0.25, 0.25)
        val color = (d.colorIntensity + a.colorIntensityDelta).coerceIn(0.0, 1.0)

        val span = (q.highlightP95 - q.shadowP05).coerceAtLeast(1.0)
        val flatness = max(
            ((156.0 - span) / 156.0).coerceIn(0.0, 1.0),
            ((45.0 - q.lumaStdDev) / 45.0).coerceIn(0.0, 1.0)
        )
        val clipRisk = max(q.darkClipFraction, q.brightClipFraction)
        val headroom = (1.0 - clipRisk * 14.0).coerceIn(0.0, 1.0)
        val styleWeight = if (d.abstain) 0.28 else (0.45 + 0.55 * themeStrength)
        val contrastPower = ((0.10 + 0.24 * midtoneIntent + 0.10 * localIntent + 0.14 * flatness) * headroom * styleWeight)
            .coerceIn(0.0, 0.34)
        val toePower = ((shadowDepth - 0.5) * 0.14 * styleWeight).coerceIn(-0.055, 0.075)
        val shoulderPower = (0.06 + highlightProtection * 0.16).coerceIn(0.06, 0.22)
        val targetSaturation = if (d.abstain) 1.0 else (0.88 + color * 0.28 * styleWeight).coerceIn(0.86, 1.16)
        val pivot = (q.medianLuma / 255.0).coerceIn(0.24, 0.72)

        if (contrastPower < 0.008 && kotlin.math.abs(warmth) < 0.008 && kotlin.math.abs(targetSaturation - 1.0) < 0.008) {
            return source
        }

        val out = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
        val stripeHeight = 72
        var y0 = 0
        while (y0 < source.height) {
            val h = min(stripeHeight, source.height - y0)
            val pixels = IntArray(source.width * h)
            source.getPixels(pixels, 0, source.width, 0, y0, source.width, h)
            for (i in pixels.indices) {
                val c = pixels[i]
                var r = ((c ushr 16) and 255) / 255.0
                var g = ((c ushr 8) and 255) / 255.0
                var b = (c and 255) / 255.0
                val y = (0.2126 * r + 0.7152 * g + 0.0722 * b).coerceIn(0.0, 1.0)

                // Midtone-weighted S-curve. Strongest around midtones, fades near black/white.
                val midWeight = (4.0 * y * (1.0 - y)).coerceIn(0.0, 1.0)
                var targetY = pivot + (y - pivot) * (1.0 + contrastPower * midWeight)

                // Theme shadow philosophy: positive value adds density, negative value opens shadows.
                val shadowMask = ((0.55 - y) / 0.55).coerceIn(0.0, 1.0).let { it * it }
                targetY -= toePower * shadowMask

                // Smooth shoulder avoids turning the requested contrast into clipped highlights.
                if (targetY > 0.70) {
                    val over = targetY - 0.70
                    targetY = 0.70 + over / (1.0 + shoulderPower * over * 3.2)
                }
                targetY = targetY.coerceIn(0.0, 1.0)

                if (y > 1e-5) {
                    val scale = (targetY / y).coerceIn(0.55, 1.65)
                    r *= scale; g *= scale; b *= scale
                }

                val lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).coerceAtLeast(0.0)
                r = lum + (r - lum) * targetSaturation
                g = lum + (g - lum) * targetSaturation
                b = lum + (b - lum) * targetSaturation

                r *= 1.0 + warmth * 0.55 * styleWeight
                g *= 1.0 + warmth * 0.10 * styleWeight
                b *= 1.0 - warmth * 0.60 * styleWeight

                val rr = (r.coerceIn(0.0, 1.0) * 255.0 + 0.5).toInt()
                val gg = (g.coerceIn(0.0, 1.0) * 255.0 + 0.5).toInt()
                val bb = (b.coerceIn(0.0, 1.0) * 255.0 + 0.5).toInt()
                pixels[i] = (0xff shl 24) or (rr shl 16) or (gg shl 8) or bb
            }
            out.setPixels(pixels, 0, source.width, 0, y0, source.width, h)
            y0 += h
        }
        return out
    }

    fun applyRelightIntent(
        direction: PhotographicDirection,
        adjustment: FinalRenderAdjustment? = null
    ): Pair<Double, Double> {
        val a = adjustment ?: FinalRenderAdjustment()
        return (
            direction.subjectRelightEv + a.subjectRelightDeltaEv
        ).coerceIn(-0.4, 0.8) to (
            direction.backgroundSuppressionEv + a.backgroundSuppressionDeltaEv
        ).coerceIn(-0.6, 0.1)
    }
}
