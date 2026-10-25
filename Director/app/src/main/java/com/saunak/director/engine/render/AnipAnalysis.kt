package com.saunak.director.engine.render

import android.graphics.Bitmap
import com.saunak.director.engine.subject.NormalizedRect
import com.saunak.director.processing.Look
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

object AnipImageAnalyzer {
    fun analyze(source: Bitmap, subject: NormalizedRect? = null, targetLongEdge: Int = 320): ImageStatistics {
        require(!source.isRecycled)
        val scale = min(1.0, targetLongEdge.toDouble() / max(source.width, source.height).toDouble())
        val w = max(32, (source.width * scale).toInt())
        val h = max(32, (source.height * scale).toInt())
        val small = if (w == source.width && h == source.height) source else Bitmap.createScaledBitmap(source, w, h, true)
        try {
            val pixels = IntArray(w * h)
            small.getPixels(pixels, 0, w, 0, 0, w, h)
            val hist = IntArray(256)
            var lumaSum = 0.0
            var satSum = 0.0
            var shadow = 0
            var highlight = 0
            var edgeSum = 0.0
            var noiseSum = 0.0
            var subjectLuma = 0.0
            var subjectCount = 0
            var backgroundLuma = 0.0
            var backgroundCount = 0
            val luma = FloatArray(pixels.size)

            for (i in pixels.indices) {
                val c = pixels[i]
                val r = ((c ushr 16) and 255) / 255.0
                val g = ((c ushr 8) and 255) / 255.0
                val b = (c and 255) / 255.0
                val y = 0.2126 * r + 0.7152 * g + 0.0722 * b
                luma[i] = y.toFloat()
                val mx = max(r, max(g, b)); val mn = min(r, min(g, b))
                val s = if (mx <= 1e-6) 0.0 else (mx - mn) / mx
                lumaSum += y; satSum += s
                hist[(y * 255.0 + 0.5).toInt().coerceIn(0, 255)]++
                if (y < 0.12) shadow++
                if (y > 0.92) highlight++
                val x = i % w; val yy = i / w
                val nx = (x + 0.5) / w.toDouble(); val ny = (yy + 0.5) / h.toDouble()
                if (subject != null && subject.contains(nx, ny)) { subjectLuma += y; subjectCount++ }
                else { backgroundLuma += y; backgroundCount++ }
            }

            for (y in 1 until h - 1) for (x in 1 until w - 1) {
                val i = y * w + x
                val dx = (luma[i + 1] - luma[i - 1]).toDouble()
                val dy = (luma[i + w] - luma[i - w]).toDouble()
                val grad = sqrt(dx * dx + dy * dy)
                edgeSum += grad
                val localMean = (luma[i - 1] + luma[i + 1] + luma[i - w] + luma[i + w]) * 0.25f
                val residual = abs(luma[i] - localMean).toDouble()
                if (grad < 0.08) noiseSum += residual
            }
            val total = pixels.size.toDouble()
            return ImageStatistics(
                meanLuma = lumaSum / total,
                p05Luma = percentile(hist, 0.05),
                p50Luma = percentile(hist, 0.50),
                p95Luma = percentile(hist, 0.95),
                shadowFraction = shadow / total,
                highlightFraction = highlight / total,
                meanSaturation = satSum / total,
                edgeEnergy = (edgeSum / max(1, (w - 2) * (h - 2))).coerceIn(0.0, 1.0),
                noiseProxy = (noiseSum / max(1, (w - 2) * (h - 2))).coerceIn(0.0, 0.25),
                subjectMeanLuma = if (subjectCount > 0) subjectLuma / subjectCount else null,
                backgroundMeanLuma = if (backgroundCount > 0) backgroundLuma / backgroundCount else null
            )
        } finally {
            if (small !== source && !small.isRecycled) small.recycle()
        }
    }

    private fun percentile(hist: IntArray, q: Double): Double {
        val total = hist.sum().coerceAtLeast(1)
        val target = (total * q).toInt().coerceIn(0, total - 1)
        var running = 0
        for (i in hist.indices) { running += hist[i]; if (running > target) return i / 255.0 }
        return 1.0
    }
}

object AnipRecipePlanner {
    fun base(request: AnipRequest, stats: ImageStatistics): RenderRecipe {
        val scene = (request.sceneLabel + " " + request.objective).lowercase()
        val portrait = request.subjectPresent || scene.contains("portrait") || scene.contains("person") || scene.contains("selfie")
        val night = scene.contains("night") || stats.p50Luma < 0.30
        val landscape = scene.contains("landscape") || scene.contains("nature") || scene.contains("sky")
        val food = scene.contains("food") || scene.contains("product")
        val dynamicRange = (stats.p95Luma - stats.p05Luma).coerceIn(0.0, 1.0)
        val darkNeed = ((0.48 - stats.p50Luma) / 0.48).coerceIn(0.0, 1.0)
        val highlightNeed = (stats.highlightFraction * 7.5 + max(0.0, stats.p95Luma - 0.86) * 3.0).coerceIn(0.0, 1.0)
        val shadowNeed = (stats.shadowFraction * 2.8 + darkNeed * 0.55).coerceIn(0.0, 1.0)
        val noiseNeed = (stats.noiseProxy / 0.035).coerceIn(0.0, 1.0)
        val detailNeed = (1.0 - stats.edgeEnergy / 0.11).coerceIn(0.15, 1.0)
        val subjectUnder = if (portrait && stats.subjectMeanLuma != null) {
            ((stats.backgroundMeanLuma ?: stats.meanLuma) - stats.subjectMeanLuma).coerceAtLeast(0.0)
        } else 0.0
        val integrity = request.reconstructionIntegrity
        val motionPenalty = request.motionRisk

        val style = when {
            request.look in setOf(Look.CINEMATIC_COOL, Look.CINEMATIC_WARM, Look.FILM_TEAL_ORANGE, Look.MOODY, Look.NIGHT_NEON) -> AnipStyle.CINEMATIC
            request.look in setOf(Look.LANDSCAPE_VIVID, Look.SOCIAL_POP, Look.GOLDEN_HOUR) -> AnipStyle.IMPACT
            else -> AnipStyle.NATURAL_PRO
        }
        val lookBoost = 1.0 + request.lookStrength * 0.22
        val exposure = ((0.43 - stats.p50Luma) * 1.1).coerceIn(-0.28, 0.52)
        val subjectRelight = if (portrait) (0.18 + subjectUnder * 1.9 + darkNeed * 0.22).coerceIn(0.12, 0.68) else 0.0
        val backgroundSuppression = if (portrait) (-0.10 - subjectUnder * 0.45).coerceIn(-0.36, -0.06) else 0.0

        val baseSat = when {
            request.look == Look.MONOCHROME -> 0.0
            food -> 1.12
            landscape -> 1.10
            night -> 1.08
            portrait -> 1.035
            else -> 1.06
        }
        val warmth = when {
            request.look == Look.MONOCHROME -> 0.0
            request.look == Look.COOL || request.look == Look.CINEMATIC_COOL -> -0.10
            request.look == Look.WARM || request.look == Look.GOLDEN_HOUR || request.look == Look.CINEMATIC_WARM -> 0.11
            portrait -> 0.035
            else -> 0.0
        }
        val cinematic = style == AnipStyle.CINEMATIC
        return RenderRecipe(
            style = style,
            globalExposureEv = exposure,
            shadowRecovery = (0.28 + shadowNeed * 0.60).coerceIn(0.18, 0.92),
            highlightCompression = (0.30 + highlightNeed * 0.62 + dynamicRange * 0.15).coerceIn(0.22, 0.96),
            localContrast = (0.28 + detailNeed * 0.30 + if (landscape || cinematic) 0.12 else 0.0).coerceIn(0.18, 0.78),
            saturation = if (request.look == Look.MONOCHROME) 0.0 else (1.0 + (baseSat - 1.0) * lookBoost).coerceIn(0.88, 1.24),
            warmth = warmth.coerceIn(-0.18, 0.18),
            coolShadowBias = if (cinematic || night) 0.08 else 0.02,
            subjectRelightEv = subjectRelight,
            backgroundSuppressionEv = backgroundSuppression,
            dehaze = (if (landscape) 0.20 else 0.08 + dynamicRange * 0.07).coerceIn(0.04, 0.30),
            vignette = (if (portrait || cinematic) 0.10 else 0.035).coerceIn(0.0, 0.22),
            grain = if (cinematic || request.look == Look.MONOCHROME) 0.035 else 0.0,
            routing = RestorationRouting(
                denoise = (0.18 + noiseNeed * 0.70 + if (night) 0.12 else 0.0).coerceIn(0.12, 0.90),
                microDeblur = ((detailNeed * 0.48) * (1.0 - motionPenalty * 0.45) * integrity).coerceIn(0.05, 0.55),
                chromaRepair = (0.16 + noiseNeed * 0.52).coerceIn(0.10, 0.76),
                detailRecovery = (0.24 + detailNeed * 0.46 * integrity).coerceIn(0.18, 0.78),
                opticalCompensation = (0.12 + detailNeed * 0.22).coerceIn(0.10, 0.40)
            ),
            impact = (1.02 + request.lookStrength * 0.14 + if (style == AnipStyle.IMPACT) 0.08 else 0.0).coerceIn(0.92, 1.28)
        )
    }

    fun candidates(base: RenderRecipe): List<RenderRecipe> = listOf(
        base.scaled(0.86, AnipStyle.NATURAL_PRO),
        base.scaled(1.00, base.style),
        base.scaled(1.14, AnipStyle.IMPACT),
        base.scaled(1.27, if (base.style == AnipStyle.CINEMATIC) AnipStyle.CINEMATIC else AnipStyle.IMPACT)
    )
}

object AnipRenderFieldBuilder {
    fun build(recipe: RenderRecipe, subject: NormalizedRect?, width: Int = 48, height: Int = 48): RenderField =
        build(recipe, subject?.let { AnipSubjectContext(it) }, width, height)

    fun build(recipe: RenderRecipe, subject: AnipSubjectContext?, width: Int = 48, height: Int = 48): RenderField {
        val size = width * height
        val exposure = FloatArray(size); val contrast = FloatArray(size); val saturation = FloatArray(size)
        val warmth = FloatArray(size); val denoise = FloatArray(size); val detail = FloatArray(size)
        val dehaze = FloatArray(size); val subjectWeight = FloatArray(size)
        val bounds = subject?.bounds
        val cx = bounds?.center?.x ?: 0.5; val cy = bounds?.center?.y ?: 0.5
        val sx = max(0.10, (bounds?.width ?: 0.34) * 0.72); val sy = max(0.14, (bounds?.height ?: 0.52) * 0.70)
        val depthBoost = (1.0 + max(0.0, subject?.depthSeparation ?: 0.0) * 0.30).coerceIn(1.0, 1.30)
        for (y in 0 until height) for (x in 0 until width) {
            val nx = (x + 0.5) / width; val ny = (y + 0.5) / height
            val dx = (nx - cx) / sx; val dy = (ny - cy) / sy
            val gaussian = if (subject == null) 0.0 else exp(-0.5 * (dx * dx + dy * dy)).coerceIn(0.0, 1.0)
            val maskValue = subject?.let { sampleSubjectMask(it, nx, ny) }
            val semantic = if (maskValue == null) gaussian else {
                val mc = subject.maskConfidence
                (maskValue * mc + gaussian * (1.0 - mc)).coerceIn(0.0, 1.0)
            }
            val sw = (semantic * (0.55 + 0.45 * (subject?.trackConfidence ?: 1.0))).coerceIn(0.0, 1.0)
            val i = y * width + x
            subjectWeight[i] = sw.toFloat()
            exposure[i] = (recipe.globalExposureEv + sw * recipe.subjectRelightEv * depthBoost + (1.0 - sw) * recipe.backgroundSuppressionEv).toFloat()
            contrast[i] = (recipe.localContrast * (0.70 + 0.30 * (1.0 - sw) + sw * 0.08 * depthBoost)).coerceIn(0.0, 1.0).toFloat()
            saturation[i] = if (recipe.saturation < 0.20) recipe.saturation.toFloat() else (1.0 + (recipe.saturation - 1.0) * (0.87 + 0.13 * sw)).toFloat()
            warmth[i] = (recipe.warmth + sw * 0.020 - (1.0 - sw) * recipe.coolShadowBias * 0.34).toFloat()
            denoise[i] = (recipe.routing.denoise * (0.90 + 0.14 * (1.0 - sw))).coerceIn(0.0, 1.0).toFloat()
            detail[i] = (recipe.routing.detailRecovery * (0.76 + 0.34 * sw * depthBoost)).coerceIn(0.0, 1.0).toFloat()
            dehaze[i] = (recipe.dehaze * (0.70 + 0.30 * (1.0 - sw))).toFloat()
        }
        return RenderField(width, height, exposure, contrast, saturation, warmth, denoise, detail, dehaze, subjectWeight)
    }

    private fun sampleSubjectMask(subject: AnipSubjectContext, nx: Double, ny: Double): Double? {
        val mask = subject.mask ?: return null
        if (subject.maskWidth <= 0 || subject.maskHeight <= 0) return null
        val fx = (nx.coerceIn(0.0, 1.0) * (subject.maskWidth - 1)).coerceIn(0.0, (subject.maskWidth - 1).toDouble())
        val fy = (ny.coerceIn(0.0, 1.0) * (subject.maskHeight - 1)).coerceIn(0.0, (subject.maskHeight - 1).toDouble())
        val x0 = fx.toInt(); val y0 = fy.toInt(); val x1 = min(subject.maskWidth - 1, x0 + 1); val y1 = min(subject.maskHeight - 1, y0 + 1)
        val tx = fx - x0; val ty = fy - y0
        val p00 = mask[y0 * subject.maskWidth + x0].toDouble(); val p10 = mask[y0 * subject.maskWidth + x1].toDouble()
        val p01 = mask[y1 * subject.maskWidth + x0].toDouble(); val p11 = mask[y1 * subject.maskWidth + x1].toDouble()
        return ((p00 * (1.0 - tx) + p10 * tx) * (1.0 - ty) + (p01 * (1.0 - tx) + p11 * tx) * ty).coerceIn(0.0, 1.0)
    }
}
