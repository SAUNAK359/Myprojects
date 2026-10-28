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

/**
 * Bounded residual finisher for Sierra v11.
 *
 * ANIP already consumes the full PhotographicDirection. This stage therefore applies only the
 * final-review deltas plus theme details ANIP does not carry explicitly (grain/vignette).
 *
 * Memory ownership rule: mutate the ANIP output in-place whenever it is mutable. This avoids a
 * second full-resolution output bitmap at the hottest point of the finishing pipeline. If a caller
 * ever supplies an immutable bitmap, one mutable copy is created and returned as the new owner.
 */
object PhotographicDirectionFinisher {
    fun apply(
        source: Bitmap,
        direction: PhotographicDirection?,
        adjustment: FinalRenderAdjustment? = null,
        subject: AnipSubjectContext? = null
    ): Bitmap {
        if (source.isRecycled) return source

        val d = direction
        val a = adjustment
        val themeConfidence = d?.takeUnless { it.abstain }?.themeConfidence?.coerceIn(0.0, 1.0) ?: 0.0
        val grainStrength = (d?.grain ?: 0.0).coerceIn(0.0, 0.20) * themeConfidence * 0.55
        val vignetteStrength = (d?.vignette ?: 0.0).coerceIn(0.0, 0.35) * themeConfidence * 0.55
        val hasReviewDelta = a?.let(::hasMeaningfulDelta) == true
        if (!hasReviewDelta && grainStrength <= 0.001 && vignetteStrength <= 0.001) return source

        val trust = when {
            d == null -> 0.55
            d.abstain -> 0.45
            else -> 0.60 + 0.40 * themeConfidence
        }
        val safe = a ?: FinalRenderAdjustment()

        val contrastDelta = ((safe.midtoneContrastDelta * 0.16) +
            (safe.localContrastDelta * 0.12) +
            (safe.themeStrengthDelta * 0.06)).times(trust).coerceIn(-0.075, 0.075)
        val toeDelta = (safe.shadowDepthDelta * 0.10 * trust).coerceIn(-0.025, 0.025)
        val shoulderDelta = (safe.highlightProtectionDelta * 0.12 * trust).coerceIn(-0.025, 0.025)
        val warmthDelta = (safe.warmthDelta * 0.55 * trust).coerceIn(-0.045, 0.045)
        val saturationScale = (1.0 + safe.colorIntensityDelta * 0.34 * trust).coerceIn(0.95, 1.05)
        val subjectEv = (safe.subjectRelightDeltaEv * trust).coerceIn(-0.18, 0.18)
        val backgroundEv = (safe.backgroundSuppressionDeltaEv * trust).coerceIn(-0.14, 0.10)

        // Preserve the v10 tonal pivot. This analysis does not create another full-resolution bitmap.
        val pivot = runCatching { (PhotoQuality.analyze(source).medianLuma / 255.0).coerceIn(0.24, 0.72) }
            .getOrDefault(0.46)

        val target = if (source.isMutable) source
        else source.copy(Bitmap.Config.ARGB_8888, true) ?: error("Could not create mutable Sierra finishing bitmap")
        require(!target.isRecycled && target.width > 0 && target.height > 0) { "Invalid Sierra finishing bitmap" }

        val stripeHeight = 72
        var y0 = 0
        while (y0 < target.height) {
            val h = min(stripeHeight, target.height - y0)
            val pixels = IntArray(target.width * h)
            target.getPixels(pixels, 0, target.width, 0, y0, target.width, h)
            for (i in pixels.indices) {
                val px = i % target.width
                val py = y0 + i / target.width
                val nx = (px + 0.5) / target.width.toDouble()
                val ny = (py + 0.5) / target.height.toDouble()
                val c = pixels[i]
                var r = ((c ushr 16) and 255) / 255.0
                var g = ((c ushr 8) and 255) / 255.0
                var b = (c and 255) / 255.0

                val subjectWeight = subjectWeight(subject, nx, ny)
                val spatialEv = subjectWeight * subjectEv + (1.0 - subjectWeight) * backgroundEv
                val exposureGain = 2.0.pow(spatialEv)
                r *= exposureGain; g *= exposureGain; b *= exposureGain

                val exposedY = (0.2126 * r + 0.7152 * g + 0.0722 * b).coerceIn(0.0, 1.0)
                val midWeight = (4.0 * exposedY * (1.0 - exposedY)).coerceIn(0.0, 1.0)
                var targetY = pivot + (exposedY - pivot) * (1.0 + contrastDelta * midWeight)

                if (targetY < 0.34 && toeDelta != 0.0) {
                    val toeMask = ((0.34 - targetY) / 0.34).coerceIn(0.0, 1.0)
                    targetY *= (1.0 - toeDelta * toeMask).coerceIn(0.96, 1.04)
                }
                if (targetY > 0.72 && shoulderDelta > 0.0) {
                    val over = targetY - 0.72
                    targetY = 0.72 + over / (1.0 + shoulderDelta * over * 4.0)
                }
                targetY = targetY.coerceIn(0.0, 1.0)
                if (exposedY > 1e-5) {
                    val scale = (targetY / exposedY).coerceIn(0.88, 1.12)
                    r *= scale; g *= scale; b *= scale
                }

                val lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).coerceAtLeast(0.0)
                r = lum + (r - lum) * saturationScale
                g = lum + (g - lum) * saturationScale
                b = lum + (b - lum) * saturationScale
                r *= 1.0 + warmthDelta * 0.48
                g *= 1.0 + warmthDelta * 0.08
                b *= 1.0 - warmthDelta * 0.52

                if (vignetteStrength > 0.001) {
                    val dx = nx - 0.5; val dy = ny - 0.5
                    val radial = (sqrt(dx * dx + dy * dy) / 0.70710678).coerceIn(0.0, 1.0)
                    val v = 1.0 - vignetteStrength * radial.pow(1.90) * 0.42
                    r *= v; g *= v; b *= v
                }
                if (grainStrength > 0.001) {
                    val grain = deterministicNoise(px, py) * grainStrength * 0.018
                    r += grain; g += grain; b += grain
                }

                val rr = (r.coerceIn(0.0, 1.0) * 255.0 + 0.5).toInt()
                val gg = (g.coerceIn(0.0, 1.0) * 255.0 + 0.5).toInt()
                val bb = (b.coerceIn(0.0, 1.0) * 255.0 + 0.5).toInt()
                pixels[i] = (0xff shl 24) or (rr shl 16) or (gg shl 8) or bb
            }
            target.setPixels(pixels, 0, target.width, 0, y0, target.width, h)
            y0 += h
        }

        return target
    }

    private fun hasMeaningfulDelta(a: FinalRenderAdjustment): Boolean =
        listOf(
            a.themeStrengthDelta,
            a.midtoneContrastDelta,
            a.localContrastDelta,
            a.shadowDepthDelta,
            a.highlightProtectionDelta,
            a.subjectRelightDeltaEv,
            a.backgroundSuppressionDeltaEv,
            a.warmthDelta,
            a.colorIntensityDelta
        ).any { kotlin.math.abs(it) > 0.002 }

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
