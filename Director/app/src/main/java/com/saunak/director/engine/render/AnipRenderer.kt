package com.saunak.director.engine.render

import android.graphics.Bitmap
import com.saunak.director.processing.Look
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

object AnipRenderer {
    fun render(source: Bitmap, recipe: RenderRecipe, field: RenderField): Bitmap {
        require(!source.isRecycled)
        val out = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
        val stripe = 96
        val sampled = DoubleArray(8)
        var y0 = 0
        while (y0 < source.height) {
            val coreH = min(stripe, source.height - y0)
            val readY = max(0, y0 - 1)
            val readBottom = min(source.height, y0 + coreH + 1)
            val readH = readBottom - readY
            val src = IntArray(source.width * readH)
            val linearLuma = FloatArray(src.size)
            source.getPixels(src, 0, source.width, 0, readY, source.width, readH)
            for (i in src.indices) {
                val c = src[i]
                val r = srgbToLinear(((c ushr 16) and 255) / 255.0)
                val g = srgbToLinear(((c ushr 8) and 255) / 255.0)
                val b = srgbToLinear((c and 255) / 255.0)
                linearLuma[i] = luma(r, g, b).toFloat()
            }
            val dst = IntArray(source.width * coreH)
            for (y in 0 until coreH) {
                val sy = y0 + y
                val ry = sy - readY
                for (x in 0 until source.width) {
                    val index = ry * source.width + x
                    val c = src[index]
                    var r = srgbToLinear(((c ushr 16) and 255) / 255.0)
                    var g = srgbToLinear(((c ushr 8) and 255) / 255.0)
                    var b = srgbToLinear((c and 255) / 255.0)
                    val originalLuma = linearLuma[index].toDouble()
                    val local = localMeanLuma(linearLuma, source.width, readH, x, ry)
                    val nx = (x + 0.5) / source.width.toDouble()
                    val ny = (sy + 0.5) / source.height.toDouble()
                    sample(field, nx, ny, sampled)

                    val exposureEv = sampled[0]
                    val contrast = sampled[1]
                    val saturation = sampled[2]
                    val warmth = sampled[3]
                    val denoise = sampled[4]
                    val detail = sampled[5]
                    val dehaze = sampled[6]
                    val subjectWeight = sampled[7]

                    val residual = originalLuma - local
                    val edgeMagnitude = abs(residual)
                    val smoothWeight = (1.0 - edgeMagnitude / 0.055).coerceIn(0.0, 1.0)
                    val denoiseBlend = (denoise * smoothWeight * 0.58).coerceIn(0.0, 0.66)
                    if (originalLuma > 1e-6) {
                        val denoisedLuma = originalLuma * (1.0 - denoiseBlend) + local * denoiseBlend
                        val denoiseScale = denoisedLuma / originalLuma
                        r *= denoiseScale; g *= denoiseScale; b *= denoiseScale
                    }

                    val exposureGain = 2.0.pow(exposureEv)
                    r *= exposureGain; g *= exposureGain; b *= exposureGain
                    var yLin = luma(r, g, b)

                    // Zero-preserving photographic toe. Unlike an additive shadow lift this can never turn
                    // literal black into grey. Detail-bearing shadows can open while intentional blacks stay anchored.
                    val shadowMask = ((0.48 - yLin) / 0.48).coerceIn(0.0, 1.0).pow(1.45)
                    val recoveryGain = 1.0 + shadowMask * recipe.shadowRecovery * (0.72 - 0.34 * recipe.blackAnchor)
                    var tonal = yLin * recoveryGain
                    val deepToe = ((0.16 - tonal) / 0.16).coerceIn(0.0, 1.0).pow(1.35)
                    tonal *= 1.0 - deepToe * recipe.toeStrength * recipe.blackAnchor * 0.30

                    // Preserve luminous sources (neon, sun, stage lights) while compressing true clipping risk.
                    val highlightMask = ((tonal - 0.66) / 0.34).coerceIn(0.0, 1.0)
                    val effectiveCompression = recipe.highlightCompression * (1.0 - 0.48 * recipe.highlightGlowPreservation)
                    val compressed = tonal / (1.0 + effectiveCompression * highlightMask * max(0.0, tonal - 0.58) * 1.72)
                    if (yLin > 1e-6) {
                        val scale = compressed / yLin
                        r *= scale; g *= scale; b *= scale
                    }
                    yLin = luma(r, g, b)

                    val contrastStrength = (0.30 * contrast + 0.26 * dehaze).coerceIn(0.0, 0.44)
                    val contrastTarget = (yLin + (yLin - local) * contrastStrength).coerceAtLeast(0.0)
                    if (yLin > 1e-6) {
                        val scale = contrastTarget / yLin
                        r *= scale; g *= scale; b *= scale
                    }

                    val subjectTextureGuard = 1.0 - subjectWeight * recipe.skinProtection * 0.28
                    val detailStrength = ((0.44 * detail + 0.28 * recipe.routing.microDeblur +
                        0.16 * recipe.routing.opticalCompensation) * recipe.textureProtection * subjectTextureGuard)
                        .coerceIn(0.0, 0.72)
                    val detailDelta = residual.coerceIn(-0.085, 0.085) * detailStrength
                    val afterContrast = luma(r, g, b)
                    if (afterContrast > 1e-6) {
                        val target = (afterContrast + detailDelta).coerceAtLeast(0.0)
                        val scale = target / afterContrast
                        r *= scale; g *= scale; b *= scale
                    }

                    val lum = luma(r, g, b)
                    r = lum + (r - lum) * saturation
                    g = lum + (g - lum) * saturation
                    b = lum + (b - lum) * saturation
                    r *= 1.0 + warmth * 0.78
                    g *= 1.0 + warmth * 0.16
                    b *= 1.0 - warmth * 0.84

                    if (subjectWeight > 0.001) {
                        val subjectPop = 1.0 + subjectWeight * recipe.localContrast * 0.038 * (1.0 - recipe.skinProtection * 0.32)
                        r *= subjectPop; g *= subjectPop; b *= subjectPop
                    }

                    val dx = nx - 0.5
                    val dy = ny - 0.5
                    val radial = sqrt(dx * dx + dy * dy) / 0.70710678
                    val vignette = 1.0 - recipe.vignette * radial.pow(1.8) * 0.64
                    r *= vignette; g *= vignette; b *= vignette
                    if (recipe.grain > 0.001) {
                        val grain = deterministicNoise(x, sy) * recipe.grain * 0.030
                        r += grain; g += grain; b += grain
                    }

                    val rr = (linearToSrgb(r.coerceIn(0.0, 1.0)) * 255.0 + 0.5).toInt().coerceIn(0, 255)
                    val gg = (linearToSrgb(g.coerceIn(0.0, 1.0)) * 255.0 + 0.5).toInt().coerceIn(0, 255)
                    val bb = (linearToSrgb(b.coerceIn(0.0, 1.0)) * 255.0 + 0.5).toInt().coerceIn(0, 255)
                    dst[y * source.width + x] = (0xff shl 24) or (rr shl 16) or (gg shl 8) or bb
                }
            }
            out.setPixels(dst, 0, source.width, 0, y0, source.width, coreH)
            y0 += coreH
        }
        return out
    }

    private fun sample(field: RenderField, nx: Double, ny: Double, out: DoubleArray) {
        val fx = (nx.coerceIn(0.0, 1.0) * (field.width - 1)).coerceIn(0.0, (field.width - 1).toDouble())
        val fy = (ny.coerceIn(0.0, 1.0) * (field.height - 1)).coerceIn(0.0, (field.height - 1).toDouble())
        val x0 = fx.toInt(); val y0 = fy.toInt()
        val x1 = min(field.width - 1, x0 + 1); val y1 = min(field.height - 1, y0 + 1)
        val tx = fx - x0; val ty = fy - y0
        out[0] = bilerp(field.exposureEv, field.width, x0, x1, y0, y1, tx, ty)
        out[1] = bilerp(field.contrast, field.width, x0, x1, y0, y1, tx, ty)
        out[2] = bilerp(field.saturation, field.width, x0, x1, y0, y1, tx, ty)
        out[3] = bilerp(field.warmth, field.width, x0, x1, y0, y1, tx, ty)
        out[4] = bilerp(field.denoise, field.width, x0, x1, y0, y1, tx, ty)
        out[5] = bilerp(field.detail, field.width, x0, x1, y0, y1, tx, ty)
        out[6] = bilerp(field.dehaze, field.width, x0, x1, y0, y1, tx, ty)
        out[7] = bilerp(field.subjectWeight, field.width, x0, x1, y0, y1, tx, ty)
    }

    private fun bilerp(values: FloatArray, width: Int, x0: Int, x1: Int, y0: Int, y1: Int, tx: Double, ty: Double): Double {
        val p00 = values[y0 * width + x0].toDouble(); val p10 = values[y0 * width + x1].toDouble()
        val p01 = values[y1 * width + x0].toDouble(); val p11 = values[y1 * width + x1].toDouble()
        val top = p00 * (1.0 - tx) + p10 * tx
        val bottom = p01 * (1.0 - tx) + p11 * tx
        return top * (1.0 - ty) + bottom * ty
    }

    private fun localMeanLuma(luma: FloatArray, width: Int, height: Int, x: Int, y: Int): Double {
        var sum = 0.0; var count = 0
        for (dy in -1..1) for (dx in -1..1) {
            val xx = (x + dx).coerceIn(0, width - 1); val yy = (y + dy).coerceIn(0, height - 1)
            sum += luma[yy * width + xx]; count++
        }
        return sum / count.coerceAtLeast(1)
    }

    private fun luma(r: Double, g: Double, b: Double) = 0.2126 * r + 0.7152 * g + 0.0722 * b
    private fun srgbToLinear(v: Double): Double = if (v <= 0.04045) v / 12.92 else ((v + 0.055) / 1.055).pow(2.4)
    private fun linearToSrgb(v: Double): Double = if (v <= 0.0031308) 12.92 * v else 1.055 * v.pow(1.0 / 2.4) - 0.055
    private fun deterministicNoise(x: Int, y: Int): Double {
        var n = x * 374761393 + y * 668265263
        n = (n xor (n ushr 13)) * 1274126177
        return ((n xor (n ushr 16)) and 0xffff) / 32767.5 - 1.0
    }
}

object AnipCandidateCritic {
    fun score(source: ImageStatistics, output: ImageStatistics, request: AnipRequest): RenderCandidateScore {
        fun closeness(v: Double, target: Double, tolerance: Double) = (1.0 - abs(v - target) / tolerance).coerceIn(0.0, 1.0)
        val situation = AnipSituationResolver.resolve(request, source)
        val target = AnipSituationResolver.exposureTarget(situation)
        val exposure = 0.62 * closeness(output.p50Luma, target, if (situation == AnipSituation.DOCUMENT) 0.24 else 0.28) +
            0.38 * (1.0 - output.highlightFraction * 4.0).coerceIn(0.0, 1.0)

        val sourceSpan = (source.p95Luma - source.p05Luma).coerceAtLeast(0.04)
        val outputSpan = (output.p95Luma - output.p05Luma).coerceAtLeast(0.0)
        val spanRatio = outputSpan / sourceSpan
        val contrast = when {
            spanRatio < 0.72 -> (spanRatio / 0.72).coerceIn(0.0, 1.0)
            spanRatio <= 1.32 -> 1.0
            else -> (1.0 - (spanRatio - 1.32) / 0.75).coerceIn(0.0, 1.0)
        }

        val saturationTarget = when {
            request.look == Look.MONOCHROME -> 0.0
            situation in setOf(AnipSituation.PORTRAIT, AnipSituation.PORTRAIT_LOW_LIGHT, AnipSituation.SELFIE, AnipSituation.SELFIE_LOW_LIGHT) -> 0.30
            situation == AnipSituation.LANDSCAPE -> 0.37
            situation in setOf(AnipSituation.NIGHT_NEON, AnipSituation.STAGE, AnipSituation.SUNSET) -> max(0.30, source.meanSaturation)
            situation == AnipSituation.DOCUMENT -> 0.20
            else -> 0.33
        }
        val colorTolerance = if (request.look == Look.MONOCHROME) 0.08 else 0.28
        val color = closeness(output.meanSaturation, saturationTarget, colorTolerance)
        val edgeRatio = output.edgeEnergy.safeRatio(source.edgeEnergy.coerceAtLeast(0.006))
        val detail = when {
            edgeRatio < 0.86 -> (edgeRatio / 0.86).coerceIn(0.0, 1.0)
            edgeRatio <= 1.52 -> 1.0
            else -> (1.0 - (edgeRatio - 1.52) / 1.0).coerceIn(0.0, 1.0)
        }
        val subject = if (request.subjectPresent && output.subjectMeanLuma != null && output.backgroundMeanLuma != null) {
            val separation = output.subjectMeanLuma - output.backgroundMeanLuma
            (0.62 + separation * 1.45).coerceIn(0.0, 1.0)
        } else 0.80

        val darkScene = situation in setOf(AnipSituation.NIGHT, AnipSituation.NIGHT_NEON, AnipSituation.STAGE, AnipSituation.SUNSET,
            AnipSituation.PORTRAIT_LOW_LIGHT, AnipSituation.SELFIE_LOW_LIGHT)
        val allowedBlackRise = if (darkScene) 0.018 else 0.035
        val raisedBlackPenalty = if (source.p05Luma < 0.16) max(0.0, output.p05Luma - source.p05Luma - allowedBlackRise) * 7.5 else 0.0
        val flatHdrPenalty = max(0.0, 0.78 - spanRatio) * if (darkScene) 2.1 else 1.25
        val crushedShadowPenalty = if (!darkScene) max(0.0, source.p05Luma - output.p05Luma - 0.055) * 3.0 else 0.0
        val luminousHighlightLoss = if (situation in setOf(AnipSituation.NIGHT_NEON, AnipSituation.STAGE, AnipSituation.SUNSET)) {
            max(0.0, source.p95Luma - output.p95Luma - 0.10) * 2.6
        } else 0.0
        val clipPenalty = (output.highlightFraction * 5.0).coerceIn(0.0, 1.0)
        val oversharpenPenalty = max(0.0, edgeRatio - 1.68).coerceIn(0.0, 1.0)
        val saturationPenalty = if (request.look == Look.MONOCHROME) output.meanSaturation * 2.0 else max(0.0, output.meanSaturation - 0.60) * 1.8
        val artifact = (0.28 * clipPenalty + 0.20 * oversharpenPenalty + 0.12 * saturationPenalty +
            0.20 * raisedBlackPenalty + 0.12 * flatHdrPenalty + 0.04 * crushedShadowPenalty + 0.04 * luminousHighlightLoss)
            .coerceIn(0.0, 1.0)
        val total = (0.22 * exposure + 0.21 * contrast + 0.14 * color + 0.18 * detail + 0.25 * subject - 0.42 * artifact)
            .coerceIn(0.0, 1.0)
        return RenderCandidateScore(total, exposure, contrast, color, detail, subject, artifact)
    }
}

class AnipEngine {
    fun render(source: Bitmap, request: AnipRequest, subject: AnipSubjectContext?): Pair<Bitmap, AnipRenderResult> {
        val bounds = subject?.bounds
        val sourceStats = AnipImageAnalyzer.analyze(source, bounds)
        val base = AnipRecipePlanner.base(request, sourceStats)
        val candidates = AnipRecipePlanner.candidates(base)
        val maxEdge = 512
        val scale = min(1.0, maxEdge.toDouble() / max(source.width, source.height).toDouble())
        val preview = if (scale < 0.999) Bitmap.createScaledBitmap(source, max(32, (source.width * scale).toInt()), max(32, (source.height * scale).toInt()), true) else source
        try {
            var bestRecipe = candidates.first()
            var bestScore = RenderCandidateScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            for (candidate in candidates) {
                val field = AnipRenderFieldBuilder.build(candidate, subject)
                val rendered = AnipRenderer.render(preview, candidate, field)
                try {
                    val stats = AnipImageAnalyzer.analyze(rendered, bounds)
                    val score = AnipCandidateCritic.score(sourceStats, stats, request)
                    if (score.total > bestScore.total) { bestRecipe = candidate; bestScore = score }
                } finally { if (!rendered.isRecycled) rendered.recycle() }
            }
            val finalField = AnipRenderFieldBuilder.build(bestRecipe, subject)
            val finalBitmap = AnipRenderer.render(source, bestRecipe, finalField)
            val finalStats = AnipImageAnalyzer.analyze(finalBitmap, bounds)
            val finalScore = AnipCandidateCritic.score(sourceStats, finalStats, request)
            return finalBitmap to AnipRenderResult(bestRecipe, finalScore, sourceStats, finalStats, candidates.size)
        } finally {
            if (preview !== source && !preview.isRecycled) preview.recycle()
        }
    }
}
