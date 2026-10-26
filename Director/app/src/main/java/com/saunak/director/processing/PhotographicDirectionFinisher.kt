package com.saunak.director.processing

import android.graphics.Bitmap
import com.saunak.director.director.FinalRenderAdjustment
import com.saunak.director.director.PhotographicDirection
import com.saunak.director.engine.render.AnipSubjectContext
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/** Deterministic bounded executor for Gemini/Sierra photographic direction. */
object PhotographicDirectionFinisher {
    fun apply(
        source: Bitmap,
        direction: PhotographicDirection?,
        adjustment: FinalRenderAdjustment? = null,
        subject: AnipSubjectContext? = null
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
        val subjectEv = (d.subjectRelightEv + a.subjectRelightDeltaEv).coerceIn(-0.4, 0.8)
        val backgroundEv = (d.backgroundSuppressionEv + a.backgroundSuppressionDeltaEv).coerceIn(-0.6, 0.1)
        val grainStrength = d.grain.coerceIn(0.0, 0.20) * themeStrength
        val vignetteStrength = d.vignette.coerceIn(0.0, 0.35) * themeStrength

        val span = (q.highlightP95 - q.shadowP05).coerceAtLeast(1.0)
        val flatness = max(((156.0 - span) / 156.0).coerceIn(0.0, 1.0), ((45.0 - q.lumaStdDev) / 45.0).coerceIn(0.0, 1.0))
        val clipRisk = max(q.darkClipFraction, q.brightClipFraction)
        val headroom = (1.0 - clipRisk * 14.0).coerceIn(0.0, 1.0)
        val styleWeight = if (d.abstain) 0.28 else (0.45 + 0.55 * themeStrength)
        val contrastPower = ((0.10 + 0.24 * midtoneIntent + 0.10 * localIntent + 0.14 * flatness) * headroom * styleWeight).coerceIn(0.0, 0.34)
        val toePower = ((shadowDepth - 0.5) * 0.14 * styleWeight).coerceIn(-0.055, 0.075)
        val shoulderPower = (0.06 + highlightProtection * 0.16).coerceIn(0.06, 0.22)
        val targetSaturation = if (d.abstain) 1.0 else (0.88 + color * 0.28 * styleWeight).coerceIn(0.86, 1.16)
        val pivot = (q.medianLuma / 255.0).coerceIn(0.24, 0.72)

        val out = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
        val stripeHeight = 72
        var y0 = 0
        while (y0 < source.height) {
            val h = min(stripeHeight, source.height - y0)
            val pixels = IntArray(source.width * h)
            source.getPixels(pixels, 0, source.width, 0, y0, source.width, h)
            for (i in pixels.indices) {
                val px = i % source.width
                val py = y0 + i / source.width
                val nx = (px + 0.5) / source.width.toDouble()
                val ny = (py + 0.5) / source.height.toDouble()
                val c = pixels[i]
                var r = ((c ushr 16) and 255) / 255.0
                var g = ((c ushr 8) and 255) / 255.0
                var b = (c and 255) / 255.0
                val y = (0.2126 * r + 0.7152 * g + 0.0722 * b).coerceIn(0.0, 1.0)

                val subjectWeight = subjectWeight(subject, nx, ny)
                val spatialEv = (subjectWeight * subjectEv + (1.0 - subjectWeight) * backgroundEv) * styleWeight
                val exposureGain = 2.0.pow(spatialEv)
                r *= exposureGain; g *= exposureGain; b *= exposureGain

                val exposedY = (0.2126 * r + 0.7152 * g + 0.0722 * b).coerceIn(0.0, 1.0)
                val midWeight = (4.0 * exposedY * (1.0 - exposedY)).coerceIn(0.0, 1.0)
                var targetY = pivot + (exposedY - pivot) * (1.0 + contrastPower * midWeight)
                val shadowMask = ((0.55 - exposedY) / 0.55).coerceIn(0.0, 1.0).let { it * it }
                targetY -= toePower * shadowMask
                if (targetY > 0.70) {
                    val over = targetY - 0.70
                    targetY = 0.70 + over / (1.0 + shoulderPower * over * 3.2)
                }
                targetY = targetY.coerceIn(0.0, 1.0)
                if (exposedY > 1e-5) {
                    val scale = (targetY / exposedY).coerceIn(0.55, 1.65)
                    r *= scale; g *= scale; b *= scale
                }

                val lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).coerceAtLeast(0.0)
                r = lum + (r - lum) * targetSaturation
                g = lum + (g - lum) * targetSaturation
                b = lum + (b - lum) * targetSaturation
                r *= 1.0 + warmth * 0.55 * styleWeight
                g *= 1.0 + warmth * 0.10 * styleWeight
                b *= 1.0 - warmth * 0.60 * styleWeight

                if (vignetteStrength > 0.001) {
                    val dx = nx - 0.5; val dy = ny - 0.5
                    val radial = (sqrt(dx * dx + dy * dy) / 0.70710678).coerceIn(0.0, 1.0)
                    val v = 1.0 - vignetteStrength * radial.pow(1.85) * 0.62
                    r *= v; g *= v; b *= v
                }
                if (grainStrength > 0.001) {
                    val grain = deterministicNoise(px, py) * grainStrength * 0.025
                    r += grain; g += grain; b += grain
                }

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

    private fun subjectWeight(subject: AnipSubjectContext?, nx: Double, ny: Double): Double {
        subject ?: return 0.0
        val mask = subject.mask
        if (mask != null && subject.maskWidth > 0 && subject.maskHeight > 0) {
            val x = (nx * (subject.maskWidth - 1)).toInt().coerceIn(0, subject.maskWidth - 1)
            val y = (ny * (subject.maskHeight - 1)).toInt().coerceIn(0, subject.maskHeight - 1)
            val m = mask[y * subject.maskWidth + x].toDouble().coerceIn(0.0, 1.0)
            if (subject.maskConfidence >= 0.45) return (m * subject.maskConfidence).coerceIn(0.0, 1.0)
        }
        val b = subject.bounds
        val sx = max(0.10, b.width * 0.70); val sy = max(0.14, b.height * 0.68)
        val dx = (nx - b.center.x) / sx; val dy = (ny - b.center.y) / sy
        return (exp(-0.5 * (dx * dx + dy * dy)) * subject.trackConfidence).coerceIn(0.0, 1.0)
    }

    private fun deterministicNoise(x: Int, y: Int): Double {
        var n = x * 374761393 + y * 668265263
        n = (n xor (n ushr 13)) * 1274126177
        return ((n xor (n ushr 16)) and 0xffff) / 32767.5 - 1.0
    }
}
