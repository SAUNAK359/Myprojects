package com.saunak.director.processing

import android.graphics.Bitmap
import com.saunak.director.director.SierraRenderSpec
import com.saunak.director.engine.subject.NormalizedRect
import com.saunak.director.engine.subject.SubjectMaskState
import com.saunak.director.engine.subject.SubjectTrack
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/** Deterministic executor for Sierra's final Render Spec with bounded pixel-buffer memory. */
object SierraRenderExecutor {
    data class Result(val bitmap: Bitmap, val operations: List<String>)

    fun render(source: Bitmap, spec: SierraRenderSpec, subject: SubjectTrack?, protectSource: Boolean = false): Result {
        require(!source.isRecycled && source.width > 0 && source.height > 0)
        require(spec.execution.precision == "linear_srgb_argb8888")
        val working = when {
            protectSource -> source.copy(Bitmap.Config.ARGB_8888, true)
            source.isMutable && source.config == Bitmap.Config.ARGB_8888 -> source
            else -> source.copy(Bitmap.Config.ARGB_8888, true)
        } ?: error("Could not allocate Sierra render buffer")
        val width = working.width
        val height = working.height
        val tileRows = (TARGET_TILE_PIXELS / width.coerceAtLeast(1)).coerceIn(1, MAX_TILE_ROWS)
        val pixels = IntArray(width * tileRows)
        val mask = subject?.mask?.takeIf { spec.spatial.useSubjectMask && it.confidence >= 0.25 }
        val bounds = subject?.bounds?.takeIf { spec.spatial.useSubjectMask }
        val depthSeparation = subject?.depth?.takeIf { spec.spatial.useDepth && it.confidence >= 0.25 }
            ?.foregroundSeparation?.coerceIn(-1.0, 1.0) ?: 0.0
        val materialDetail = listOf(
            spec.materials.skinTexture, spec.materials.hairDetail, spec.materials.fabricDetail,
            spec.materials.foliageDetail, spec.materials.metalSpecular, spec.materials.glassSpecular
        ).average().coerceIn(0.0, 1.0)
        val backgroundPresence = spec.hierarchy.backgroundPresence.coerceIn(0.0, 1.0)
        val grainCell = (1.0 + spec.film.grainSize.coerceIn(0.0, 1.0) * 3.0).toInt().coerceIn(1, 4)

        var tileTop = 0
        while (tileTop < height) {
            val rows = min(tileRows, height - tileTop)
            working.getPixels(pixels, 0, width, 0, tileTop, width, rows)
            for (localY in 0 until rows) {
                val y = tileTop + localY
                val ny = if (height <= 1) 0.5 else y.toDouble() / (height - 1)
                for (x in 0 until width) {
                    val nx = if (width <= 1) 0.5 else x.toDouble() / (width - 1)
                    val index = localY * width + x
                    val p = pixels[index]
                    var r = srgbToLinear(((p shr 16) and 255) / 255.0)
                    var g = srgbToLinear(((p shr 8) and 255) / 255.0)
                    var b = srgbToLinear((p and 255) / 255.0)

                    val sw = subjectWeight(nx, ny, mask, bounds)
                    val effectiveBackgroundEv = spec.localLight.backgroundEv * (1.0 - 0.45 * backgroundPresence)
                    val exposure = 2.0.pow(spec.tonal.exposureEv + sw * spec.localLight.subjectEv + (1.0 - sw) * effectiveBackgroundEv)
                    r *= exposure; g *= exposure; b *= exposure

                    val initialLuma = luma(r, g, b)
                    val tone = toneMap(initialLuma, spec)
                    val scale = if (initialLuma > 1e-8) tone / initialLuma else 1.0
                    r *= scale; g *= scale; b *= scale

                    val warmth = spec.color.temperature
                    r *= (1.0 + warmth * 0.34).coerceIn(0.65, 1.45)
                    b *= (1.0 - warmth * 0.34).coerceIn(0.65, 1.45)
                    val tint = spec.color.tint
                    g *= (1.0 - tint * 0.18).coerceIn(0.72, 1.28)
                    r *= (1.0 + tint * 0.08).coerceIn(0.86, 1.14)
                    b *= (1.0 + tint * 0.08).coerceIn(0.86, 1.14)

                    var lum = luma(r, g, b)
                    val sat = (spec.color.saturation *
                        (1.0 + spec.color.vibrance * (1.0 - saturationEstimate(r, g, b)) * 0.45) *
                        spec.finishing.outputSaturation * (1.0 + spec.film.colorResponse * 0.08)).coerceIn(0.0, 2.2)
                    r = lum + (r - lum) * sat; g = lum + (g - lum) * sat; b = lum + (b - lum) * sat

                    val depthGain = depthSeparation.coerceAtLeast(0.0) * sw * 0.10
                    val hierarchy = 1.0 + spec.hierarchy.subjectSeparation * sw * 0.12 + depthGain + spec.localLight.dodgeBurnStrength * (sw - 0.5) * 0.12
                    r *= hierarchy; g *= hierarchy; b *= hierarchy

                    val subjectDetailProtection = 0.65 + 0.35 * spec.color.skinProtection.coerceIn(0.0, 1.0)
                    val detailGain = if (sw > 0.0) materialDetail * subjectDetailProtection else materialDetail
                    val contrast = 1.0 + (spec.texture.microcontrast + spec.texture.detail * detailGain + spec.finishing.outputSharpen) * 0.16 +
                        spec.atmosphere.localContrast * 0.12 + spec.finishing.outputContrast * 0.18
                    r = 0.5 + (r - 0.5) * contrast; g = 0.5 + (g - 0.5) * contrast; b = 0.5 + (b - 0.5) * contrast

                    if (spec.atmosphere.haze > 0.0) {
                        val h = spec.atmosphere.haze * 0.10
                        r = r * (1.0 - h) + h; g = g * (1.0 - h) + h; b = b * (1.0 - h) + h
                    }
                    if (spec.atmosphere.dehaze > 0.0) {
                        val c = 1.0 + spec.atmosphere.dehaze * 0.18
                        r = 0.5 + (r - 0.5) * c; g = 0.5 + (g - 0.5) * c; b = 0.5 + (b - 0.5) * c
                    }

                    val dist = sqrt((nx - 0.5) * (nx - 0.5) + (ny - 0.5) * (ny - 0.5)) / 0.70710678
                    val vignetteAmount = max(spec.optical.vignette, spec.optical.lensFalloff)
                    val vignette = 1.0 - vignetteAmount * smoothstep(0.45, 1.0, dist) * 0.45
                    r *= vignette; g *= vignette; b *= vignette

                    val hi = smoothstep(0.72, 1.0, max(r, max(g, b)))
                    val glow = (spec.highlights.bloom + spec.highlights.halation + spec.atmosphere.glow) * hi * 0.035
                    r += glow * 1.08; g += glow; b += glow * 0.90

                    val grain = deterministicNoise(x / grainCell, y / grainCell, spec.intentSummary.hashCode()) * spec.film.grainAmount * 0.045
                    r += grain; g += grain; b += grain

                    lum = luma(r, g, b)
                    if (spec.finishing.preserveBlacks && lum < 0.06) {
                        val anchor = (1.0 - spec.shadows.blackAnchor * 0.20).coerceIn(0.75, 1.0)
                        r *= anchor; g *= anchor; b *= anchor
                    }

                    val rr = (linearToSrgb(r.coerceIn(0.0, 1.0)) * 255.0 + 0.5).toInt().coerceIn(0, 255)
                    val gg = (linearToSrgb(g.coerceIn(0.0, 1.0)) * 255.0 + 0.5).toInt().coerceIn(0, 255)
                    val bb = (linearToSrgb(b.coerceIn(0.0, 1.0)) * 255.0 + 0.5).toInt().coerceIn(0, 255)
                    pixels[index] = (0xff shl 24) or (rr shl 16) or (gg shl 8) or bb
                }
            }
            working.setPixels(pixels, 0, width, 0, tileTop, width, rows)
            tileTop += rows
        }
        return Result(working, spec.execution.operations)
    }

    private fun toneMap(input: Double, spec: SierraRenderSpec): Double {
        var y = input.coerceAtLeast(0.0)
        y = (y + spec.tonal.blackPoint).coerceAtLeast(0.0)
        val toe = (spec.tonal.toeStrength + spec.shadows.depth * 0.4 + spec.shadows.blackAnchor * 0.25).coerceIn(0.0, 1.4)
        y = y.pow((1.0 + toe * 0.35).coerceAtLeast(0.55))
        y += spec.shadows.recovery * (1.0 - smoothstep(0.08, 0.45, y)) * 0.16
        val filmContrast = spec.film.curveStrength * 0.18
        y = 0.5 + (y - 0.5) * (1.0 + spec.tonal.midtoneContrast * 0.55 + filmContrast)
        val preservation = max(spec.highlights.practicalLightPreservation, spec.hierarchy.luminousSourcePriority).coerceIn(0.0, 1.0)
        val shoulder = (spec.tonal.shoulderStrength + spec.highlights.rolloff + spec.highlights.compression + spec.tonal.dynamicRangeCompression)
            .coerceIn(0.0, 3.0) * (1.0 - preservation * 0.28)
        if (y > 0.62) {
            val t = (y - 0.62) / 0.38
            y = 0.62 + 0.38 * t / (1.0 + shoulder * 0.42 * t)
        }
        return y.coerceIn(0.0, 1.0)
    }

    private fun subjectWeight(nx: Double, ny: Double, mask: SubjectMaskState?, bounds: NormalizedRect?): Double {
        if (mask != null) {
            val mx = (nx * (mask.width - 1)).toInt().coerceIn(0, mask.width - 1)
            val my = (ny * (mask.height - 1)).toInt().coerceIn(0, mask.height - 1)
            return mask.probabilities[my * mask.width + mx].toDouble().coerceIn(0.0, 1.0)
        }
        if (bounds != null) {
            val c = bounds.center
            val dx = abs(nx - c.x) / max(0.08, bounds.width * 0.70)
            val dy = abs(ny - c.y) / max(0.08, bounds.height * 0.70)
            return (1.0 - sqrt(dx * dx + dy * dy)).coerceIn(0.0, 1.0)
        }
        return 0.0
    }

    private fun saturationEstimate(r: Double, g: Double, b: Double): Double {
        val hi = max(r, max(g, b)); val lo = min(r, min(g, b))
        return if (hi <= 1e-8) 0.0 else ((hi - lo) / hi).coerceIn(0.0, 1.0)
    }
    private fun luma(r: Double, g: Double, b: Double) = 0.2126 * r + 0.7152 * g + 0.0722 * b
    private fun srgbToLinear(v: Double): Double = if (v <= 0.04045) v / 12.92 else ((v + 0.055) / 1.055).pow(2.4)
    private fun linearToSrgb(v: Double): Double = if (v <= 0.0031308) v * 12.92 else 1.055 * v.pow(1.0 / 2.4) - 0.055
    private fun smoothstep(a: Double, b: Double, x: Double): Double {
        val t = ((x - a) / (b - a)).coerceIn(0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    }
    private fun deterministicNoise(x: Int, y: Int, seed: Int): Double {
        var n = x * 374761393 + y * 668265263 + seed * 69069
        n = (n xor (n ushr 13)) * 1274126177
        return ((n xor (n ushr 16)) and 0xffff) / 32767.5 - 1.0
    }

    private const val TARGET_TILE_PIXELS = 262_144
    private const val MAX_TILE_ROWS = 64
}
