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

object AnipSituationResolver {
    fun resolve(request: AnipRequest, stats: ImageStatistics): AnipSituation {
        val scene = request.semanticText()
        val dark = stats.p50Luma < 0.30
        val selfie = scene.hasAny("selfie", "front camera", "self portrait")
        val human = request.subjectPresent || scene.hasAny("portrait", "person", "people", "human", "face")
        return when {
            selfie && dark -> AnipSituation.SELFIE_LOW_LIGHT
            selfie -> AnipSituation.SELFIE
            scene.hasAny("concert", "stage", "performance", "spotlight") -> AnipSituation.STAGE
            scene.hasAny("neon", "city lights", "nightlife") && dark -> AnipSituation.NIGHT_NEON
            scene.hasAny("sunset", "sunrise", "golden hour", "dusk", "twilight") -> AnipSituation.SUNSET
            human && dark -> AnipSituation.PORTRAIT_LOW_LIGHT
            human -> AnipSituation.PORTRAIT
            scene.hasAny("architecture", "building", "facade", "cityscape") -> AnipSituation.ARCHITECTURE
            scene.hasAny("landscape", "nature", "mountain", "sea", "ocean", "forest", "sky") -> AnipSituation.LANDSCAPE
            scene.hasAny("food", "dish", "meal", "drink", "coffee") -> AnipSituation.FOOD
            scene.hasAny("product", "object", "merchandise", "packaging") -> AnipSituation.PRODUCT
            scene.hasAny("macro", "close-up", "closeup", "flower", "insect") -> AnipSituation.MACRO
            scene.hasAny("document", "receipt", "whiteboard", "page", "text") -> AnipSituation.DOCUMENT
            scene.hasAny("interior", "room", "indoor", "home", "office") -> AnipSituation.INTERIOR
            scene.contains("night") || dark -> AnipSituation.NIGHT
            else -> AnipSituation.GENERAL
        }
    }

    fun exposureTarget(situation: AnipSituation): Double = when (situation) {
        AnipSituation.NIGHT_NEON -> 0.29
        AnipSituation.NIGHT -> 0.31
        AnipSituation.STAGE -> 0.31
        AnipSituation.SUNSET -> 0.39
        AnipSituation.PORTRAIT_LOW_LIGHT, AnipSituation.SELFIE_LOW_LIGHT -> 0.39
        AnipSituation.PORTRAIT, AnipSituation.SELFIE -> 0.45
        AnipSituation.INTERIOR -> 0.44
        AnipSituation.LANDSCAPE, AnipSituation.ARCHITECTURE -> 0.44
        AnipSituation.FOOD, AnipSituation.PRODUCT, AnipSituation.MACRO -> 0.47
        AnipSituation.DOCUMENT -> 0.57
        AnipSituation.GENERAL -> 0.45
    }

    private fun String.hasAny(vararg tokens: String): Boolean = tokens.any(::contains)
}

object AnipRecipePlanner {
    fun base(request: AnipRequest, stats: ImageStatistics): RenderRecipe {
        val situation = AnipSituationResolver.resolve(request, stats)
        val portrait = situation in setOf(AnipSituation.PORTRAIT, AnipSituation.PORTRAIT_LOW_LIGHT, AnipSituation.SELFIE, AnipSituation.SELFIE_LOW_LIGHT)
        val night = situation in setOf(AnipSituation.NIGHT, AnipSituation.NIGHT_NEON, AnipSituation.STAGE, AnipSituation.PORTRAIT_LOW_LIGHT, AnipSituation.SELFIE_LOW_LIGHT)
        val landscape = situation == AnipSituation.LANDSCAPE
        val highTexture = situation in setOf(AnipSituation.ARCHITECTURE, AnipSituation.PRODUCT, AnipSituation.MACRO, AnipSituation.DOCUMENT, AnipSituation.LANDSCAPE)
        val dynamicRange = (stats.p95Luma - stats.p05Luma).coerceIn(0.0, 1.0)
        val target = AnipSituationResolver.exposureTarget(situation)
        val darkNeed = ((target - stats.p50Luma) / max(0.28, target)).coerceIn(0.0, 1.0)
        val highlightNeed = (stats.highlightFraction * 6.8 + max(0.0, stats.p95Luma - 0.88) * 2.6).coerceIn(0.0, 1.0)
        val shadowNeed = (stats.shadowFraction * 1.55 + darkNeed * 0.38).coerceIn(0.0, 1.0)
        val noiseNeed = (stats.noiseProxy / 0.035).coerceIn(0.0, 1.0)
        val detailNeed = (1.0 - stats.edgeEnergy / 0.11).coerceIn(0.15, 1.0)
        val subjectUnder = if (portrait && stats.subjectMeanLuma != null) {
            ((stats.backgroundMeanLuma ?: stats.meanLuma) - stats.subjectMeanLuma).coerceAtLeast(0.0)
        } else 0.0
        val integrity = request.reconstructionIntegrity
        val motionPenalty = request.motionRisk
        val directionWeight = request.directionConfidence.coerceIn(0.0, 1.0)

        val style = when {
            request.look in setOf(Look.CINEMATIC_COOL, Look.CINEMATIC_WARM, Look.FILM_TEAL_ORANGE, Look.MOODY, Look.NIGHT_NEON) -> AnipStyle.CINEMATIC
            request.look in setOf(Look.LANDSCAPE_VIVID, Look.SOCIAL_POP, Look.GOLDEN_HOUR) -> AnipStyle.IMPACT
            else -> AnipStyle.NATURAL_PRO
        }
        val exposureLimit = if (night) 0.30 else 0.46
        val exposure = ((target - stats.p50Luma) * 0.78).coerceIn(-0.30, exposureLimit)

        val automaticSubjectRelight = if (portrait) (0.10 + subjectUnder * 1.45 + darkNeed * 0.15).coerceIn(0.06, 0.48) else 0.0
        val subjectRelight = lerp(automaticSubjectRelight, request.subjectRelightIntentEv, directionWeight * 0.72).coerceIn(-0.25, 0.62)
        val automaticBackground = if (portrait) (-0.06 - subjectUnder * 0.30).coerceIn(-0.28, -0.03) else 0.0
        val backgroundSuppression = lerp(automaticBackground, request.backgroundSuppressionIntentEv, directionWeight * 0.68).coerceIn(-0.36, 0.05)

        val sceneShadowCap = when (situation) {
            AnipSituation.NIGHT_NEON, AnipSituation.STAGE -> 0.30
            AnipSituation.NIGHT -> 0.34
            AnipSituation.SUNSET -> 0.40
            AnipSituation.PORTRAIT_LOW_LIGHT, AnipSituation.SELFIE_LOW_LIGHT -> 0.48
            else -> 0.68
        }
        val autoShadowRecovery = (0.12 + shadowNeed * 0.48).coerceIn(0.10, sceneShadowCap)
        val intentRecovery = 1.0 - request.shadowDepthIntent
        val shadowRecovery = lerp(autoShadowRecovery, intentRecovery * sceneShadowCap, directionWeight * 0.58).coerceIn(0.08, sceneShadowCap)

        val blackAnchor = when (situation) {
            AnipSituation.NIGHT_NEON, AnipSituation.STAGE -> 0.92
            AnipSituation.NIGHT -> 0.88
            AnipSituation.SUNSET -> 0.78
            AnipSituation.PORTRAIT_LOW_LIGHT, AnipSituation.SELFIE_LOW_LIGHT -> 0.72
            AnipSituation.CINEMATIC_COMPAT -> 0.76
            AnipSituation.DOCUMENT -> 0.28
            else -> if (style == AnipStyle.CINEMATIC) 0.76 else 0.56
        }
        val toeStrength = (0.18 + request.shadowDepthIntent * 0.62 * (0.45 + 0.55 * directionWeight)).coerceIn(0.15, 0.82)
        val glowPreservation = when (situation) {
            AnipSituation.NIGHT_NEON, AnipSituation.STAGE, AnipSituation.SUNSET -> 0.88
            AnipSituation.NIGHT -> 0.68
            else -> 0.46 + request.atmospherePriority * 0.24
        }.coerceIn(0.0, 1.0)

        val autoHighlight = (0.25 + highlightNeed * 0.54 + dynamicRange * 0.10).coerceIn(0.20, 0.84)
        val highlightCompression = lerp(autoHighlight, 0.24 + request.highlightProtectionIntent * 0.62, directionWeight * 0.64).coerceIn(0.18, 0.90)
        val autoLocalContrast = (0.26 + detailNeed * 0.22 + if (landscape || highTexture || style == AnipStyle.CINEMATIC) 0.10 else 0.0).coerceIn(0.18, 0.68)
        val localContrast = lerp(autoLocalContrast, 0.18 + request.localContrastIntent * 0.58, directionWeight * 0.58).coerceIn(0.16, 0.74)

        val baseSat = when (situation) {
            AnipSituation.NIGHT_NEON, AnipSituation.STAGE -> 1.08
            AnipSituation.SUNSET -> 1.07
            AnipSituation.FOOD -> 1.09
            AnipSituation.LANDSCAPE -> 1.07
            AnipSituation.PORTRAIT, AnipSituation.PORTRAIT_LOW_LIGHT, AnipSituation.SELFIE, AnipSituation.SELFIE_LOW_LIGHT -> 1.015
            AnipSituation.DOCUMENT -> 0.98
            else -> 1.035
        }
        val intentSat = 0.90 + request.colorIntensityIntent * 0.22
        val saturation = if (request.look == Look.MONOCHROME) 0.0 else lerp(baseSat, intentSat, directionWeight * 0.66).coerceIn(0.88, 1.18)
        val automaticWarmth = when {
            request.look == Look.MONOCHROME -> 0.0
            request.look == Look.COOL || request.look == Look.CINEMATIC_COOL -> -0.10
            request.look == Look.WARM || request.look == Look.GOLDEN_HOUR || request.look == Look.CINEMATIC_WARM -> 0.10
            situation == AnipSituation.SUNSET -> 0.065
            portrait -> 0.025
            else -> 0.0
        }
        val warmth = lerp(automaticWarmth, request.warmthBiasIntent, directionWeight * 0.70).coerceIn(-0.16, 0.16)
        val textureProtection = when {
            highTexture -> 0.88
            portrait -> 0.58
            else -> 0.68
        }
        val skinProtection = if (portrait) max(0.72, request.skinNaturalness) else 0.42
        val cinematic = style == AnipStyle.CINEMATIC

        return RenderRecipe(
            style = style,
            globalExposureEv = exposure,
            shadowRecovery = shadowRecovery,
            highlightCompression = highlightCompression,
            localContrast = localContrast,
            saturation = saturation,
            warmth = warmth,
            coolShadowBias = if (cinematic || situation in setOf(AnipSituation.NIGHT, AnipSituation.NIGHT_NEON)) 0.07 else 0.015,
            subjectRelightEv = subjectRelight,
            backgroundSuppressionEv = backgroundSuppression,
            dehaze = when (situation) {
                AnipSituation.LANDSCAPE -> 0.18
                AnipSituation.ARCHITECTURE -> 0.14
                AnipSituation.NIGHT_NEON, AnipSituation.STAGE -> 0.05
                else -> (0.055 + dynamicRange * 0.06).coerceIn(0.04, 0.18)
            },
            vignette = (if (portrait || cinematic) 0.075 else 0.025).coerceIn(0.0, 0.18),
            grain = if (cinematic || request.look == Look.MONOCHROME) 0.025 else 0.0,
            routing = RestorationRouting(
                denoise = (0.14 + noiseNeed * 0.58 + if (night) 0.08 else 0.0).coerceIn(0.10, 0.78),
                microDeblur = ((detailNeed * 0.42) * (1.0 - motionPenalty * 0.50) * integrity * textureProtection).coerceIn(0.04, 0.48),
                chromaRepair = (0.12 + noiseNeed * 0.46).coerceIn(0.08, 0.66),
                detailRecovery = (0.22 + detailNeed * 0.40 * integrity * textureProtection).coerceIn(0.16, 0.70),
                opticalCompensation = (0.10 + detailNeed * 0.18).coerceIn(0.08, 0.32)
            ),
            impact = (1.0 + request.lookStrength * 0.10 + if (style == AnipStyle.IMPACT) 0.06 else 0.0).coerceIn(0.90, 1.20),
            situation = situation,
            blackAnchor = blackAnchor,
            toeStrength = toeStrength,
            highlightGlowPreservation = glowPreservation,
            textureProtection = textureProtection,
            skinProtection = skinProtection
        )
    }

    fun candidates(base: RenderRecipe): List<RenderRecipe> = listOf(
        base.scaled(0.88, AnipStyle.NATURAL_PRO),
        base.scaled(1.00, base.style),
        base.scaled(1.10, if (base.style == AnipStyle.CINEMATIC) AnipStyle.CINEMATIC else AnipStyle.IMPACT),
        base.scaled(1.18, if (base.style == AnipStyle.CINEMATIC) AnipStyle.CINEMATIC else AnipStyle.IMPACT)
    )

    private fun lerp(a: Double, b: Double, t: Double): Double = a + (b - a) * t.coerceIn(0.0, 1.0)
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
            contrast[i] = (recipe.localContrast * (0.72 + 0.28 * (1.0 - sw) + sw * 0.06 * depthBoost)).coerceIn(0.0, 1.0).toFloat()
            saturation[i] = if (recipe.saturation < 0.20) recipe.saturation.toFloat() else (1.0 + (recipe.saturation - 1.0) * (0.88 + 0.12 * sw)).toFloat()
            warmth[i] = (recipe.warmth + sw * 0.015 - (1.0 - sw) * recipe.coolShadowBias * 0.30).toFloat()
            val subjectDenoiseGuard = 1.0 - sw * recipe.skinProtection * 0.26
            denoise[i] = (recipe.routing.denoise * (0.92 + 0.10 * (1.0 - sw)) * subjectDenoiseGuard).coerceIn(0.0, 1.0).toFloat()
            val detailBoost = (0.76 + 0.24 * recipe.textureProtection + 0.18 * sw * depthBoost * (1.0 - recipe.skinProtection * 0.35))
            detail[i] = (recipe.routing.detailRecovery * detailBoost).coerceIn(0.0, 1.0).toFloat()
            dehaze[i] = (recipe.dehaze * (0.72 + 0.28 * (1.0 - sw))).toFloat()
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
