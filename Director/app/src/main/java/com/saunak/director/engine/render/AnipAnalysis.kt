package com.saunak.director.engine.render

import android.graphics.Bitmap
import com.saunak.director.engine.subject.NormalizedRect
import com.saunak.director.processing.Look
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
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
            val luma = FloatArray(pixels.size)
            var lumaSum = 0.0; var satSum = 0.0; var shadows = 0; var highlights = 0
            var edgeSum = 0.0; var noiseSum = 0.0
            var subjectLuma = 0.0; var subjectCount = 0; var backgroundLuma = 0.0; var backgroundCount = 0
            for (i in pixels.indices) {
                val c = pixels[i]
                val r = ((c ushr 16) and 255) / 255.0; val g = ((c ushr 8) and 255) / 255.0; val b = (c and 255) / 255.0
                val y = 0.2126 * r + 0.7152 * g + 0.0722 * b
                luma[i] = y.toFloat(); lumaSum += y
                val mx = max(r, max(g, b)); val mn = min(r, min(g, b)); satSum += if (mx <= 1e-6) 0.0 else (mx - mn) / mx
                hist[(y * 255.0 + 0.5).toInt().coerceIn(0, 255)]++
                if (y < 0.12) shadows++; if (y > 0.92) highlights++
                val x = i % w; val yy = i / w; val nx = (x + 0.5) / w; val ny = (yy + 0.5) / h
                if (subject != null && subject.contains(nx, ny)) { subjectLuma += y; subjectCount++ } else { backgroundLuma += y; backgroundCount++ }
            }
            for (y in 1 until h - 1) for (x in 1 until w - 1) {
                val i = y * w + x
                val dx = (luma[i + 1] - luma[i - 1]).toDouble(); val dy = (luma[i + w] - luma[i - w]).toDouble()
                val grad = sqrt(dx * dx + dy * dy); edgeSum += grad
                val local = (luma[i - 1] + luma[i + 1] + luma[i - w] + luma[i + w]) * 0.25f
                if (grad < 0.08) noiseSum += abs(luma[i] - local)
            }
            val total = pixels.size.toDouble(); val interior = max(1, (w - 2) * (h - 2)).toDouble()
            return ImageStatistics(lumaSum / total, percentile(hist, .05), percentile(hist, .50), percentile(hist, .95),
                shadows / total, highlights / total, satSum / total, (edgeSum / interior).coerceIn(0.0, 1.0),
                (noiseSum / interior).coerceIn(0.0, .25), if (subjectCount > 0) subjectLuma / subjectCount else null,
                if (backgroundCount > 0) backgroundLuma / backgroundCount else null)
        } finally { if (small !== source && !small.isRecycled) small.recycle() }
    }

    private fun percentile(hist: IntArray, q: Double): Double {
        val total = hist.sum().coerceAtLeast(1); val target = (total * q).toInt().coerceIn(0, total - 1); var running = 0
        for (i in hist.indices) { running += hist[i]; if (running > target) return i / 255.0 }
        return 1.0
    }
}

object AnipSituationResolver {
    fun resolve(request: AnipRequest, stats: ImageStatistics): AnipSituation {
        val s = request.semanticText(); val dark = stats.p50Luma < .30
        val selfie = s.hasAny("selfie", "front camera", "self portrait")
        val human = request.subjectPresent || s.hasAny("portrait", "person", "people", "human", "face")
        return when {
            selfie && dark -> AnipSituation.SELFIE_LOW_LIGHT
            selfie -> AnipSituation.SELFIE
            s.hasAny("concert", "stage", "performance", "spotlight") -> AnipSituation.STAGE
            s.hasAny("neon", "city lights", "nightlife") && dark -> AnipSituation.NIGHT_NEON
            s.hasAny("sunset", "sunrise", "golden hour", "dusk", "twilight") -> AnipSituation.SUNSET
            human && dark -> AnipSituation.PORTRAIT_LOW_LIGHT
            human -> AnipSituation.PORTRAIT
            s.hasAny("architecture", "building", "facade", "cityscape") -> AnipSituation.ARCHITECTURE
            s.hasAny("landscape", "nature", "mountain", "sea", "ocean", "forest", "sky") -> AnipSituation.LANDSCAPE
            s.hasAny("food", "dish", "meal", "drink", "coffee") -> AnipSituation.FOOD
            s.hasAny("product", "object", "merchandise", "packaging") -> AnipSituation.PRODUCT
            s.hasAny("macro", "close-up", "closeup", "flower", "insect") -> AnipSituation.MACRO
            s.hasAny("document", "receipt", "whiteboard", "page", "text") -> AnipSituation.DOCUMENT
            s.hasAny("interior", "room", "indoor", "home", "office") -> AnipSituation.INTERIOR
            dark || s.contains("night") -> AnipSituation.NIGHT
            else -> AnipSituation.GENERAL
        }
    }

    fun exposureTarget(s: AnipSituation): Double = when (s) {
        AnipSituation.NIGHT_NEON -> .29; AnipSituation.NIGHT, AnipSituation.STAGE -> .31; AnipSituation.SUNSET -> .39
        AnipSituation.PORTRAIT_LOW_LIGHT, AnipSituation.SELFIE_LOW_LIGHT -> .39
        AnipSituation.PORTRAIT, AnipSituation.SELFIE -> .45; AnipSituation.INTERIOR -> .44
        AnipSituation.LANDSCAPE, AnipSituation.ARCHITECTURE, AnipSituation.CINEMATIC_COMPAT -> .44
        AnipSituation.FOOD, AnipSituation.PRODUCT, AnipSituation.MACRO -> .47; AnipSituation.DOCUMENT -> .57
        AnipSituation.GENERAL -> .45
    }
    private fun String.hasAny(vararg tokens: String) = tokens.any(::contains)
}

object AnipRecipePlanner {
    fun base(request: AnipRequest, stats: ImageStatistics): RenderRecipe {
        val situation = AnipSituationResolver.resolve(request, stats)
        val portrait = situation in portraitSituations
        val night = situation in darkSituations
        val highTexture = situation in textureSituations
        val target = AnipSituationResolver.exposureTarget(situation)
        val dynamicRange = (stats.p95Luma - stats.p05Luma).coerceIn(0.0, 1.0)
        val darkNeed = ((target - stats.p50Luma) / max(.28, target)).coerceIn(0.0, 1.0)
        val highlightNeed = (stats.highlightFraction * 6.8 + max(0.0, stats.p95Luma - .88) * 2.6).coerceIn(0.0, 1.0)
        val shadowNeed = (stats.shadowFraction * 1.55 + darkNeed * .38).coerceIn(0.0, 1.0)
        val noiseNeed = (stats.noiseProxy / .035).coerceIn(0.0, 1.0)
        val detailNeed = (1.0 - stats.edgeEnergy / .11).coerceIn(.15, 1.0)
        val subjectUnder = if (portrait && stats.subjectMeanLuma != null) ((stats.backgroundMeanLuma ?: stats.meanLuma) - stats.subjectMeanLuma).coerceAtLeast(0.0) else 0.0
        val dw = request.directionConfidence.coerceIn(0.0, 1.0)
        val style = when {
            request.look in setOf(Look.CINEMATIC_COOL, Look.CINEMATIC_WARM, Look.FILM_TEAL_ORANGE, Look.MOODY, Look.NIGHT_NEON) -> AnipStyle.CINEMATIC
            request.look in setOf(Look.LANDSCAPE_VIVID, Look.SOCIAL_POP, Look.GOLDEN_HOUR) -> AnipStyle.IMPACT
            else -> AnipStyle.NATURAL_PRO
        }
        val exposure = ((target - stats.p50Luma) * .78).coerceIn(-.30, if (night) .30 else .46)
        val autoRelight = if (portrait) (.10 + subjectUnder * 1.45 + darkNeed * .15).coerceIn(.06, .48) else 0.0
        val subjectRelight = lerp(autoRelight, request.subjectRelightIntentEv, dw * .72).coerceIn(-.25, .62)
        val autoBackground = if (portrait) (-.06 - subjectUnder * .30).coerceIn(-.28, -.03) else 0.0
        val background = lerp(autoBackground, request.backgroundSuppressionIntentEv, dw * .68).coerceIn(-.36, .05)
        val shadowCap = when (situation) {
            AnipSituation.NIGHT_NEON, AnipSituation.STAGE -> .30; AnipSituation.NIGHT -> .34; AnipSituation.SUNSET -> .40
            AnipSituation.PORTRAIT_LOW_LIGHT, AnipSituation.SELFIE_LOW_LIGHT -> .48; else -> .68
        }
        val shadowRecovery = lerp((.12 + shadowNeed * .48).coerceIn(.10, shadowCap), (1.0 - request.shadowDepthIntent) * shadowCap, dw * .58).coerceIn(.08, shadowCap)
        val blackAnchor = when (situation) {
            AnipSituation.NIGHT_NEON, AnipSituation.STAGE -> .92; AnipSituation.NIGHT -> .88; AnipSituation.SUNSET -> .78
            AnipSituation.PORTRAIT_LOW_LIGHT, AnipSituation.SELFIE_LOW_LIGHT -> .72; AnipSituation.DOCUMENT -> .28
            else -> if (style == AnipStyle.CINEMATIC) .76 else .56
        }
        val glow = when (situation) {
            AnipSituation.NIGHT_NEON, AnipSituation.STAGE, AnipSituation.SUNSET -> .88; AnipSituation.NIGHT -> .68
            else -> .46 + request.atmospherePriority * .24
        }.coerceIn(0.0, 1.0)
        val highlight = lerp((.25 + highlightNeed * .54 + dynamicRange * .10).coerceIn(.20, .84), .24 + request.highlightProtectionIntent * .62, dw * .64).coerceIn(.18, .90)
        val autoContrast = (.26 + detailNeed * .22 + if (highTexture || style == AnipStyle.CINEMATIC) .10 else 0.0).coerceIn(.18, .68)
        val localContrast = lerp(autoContrast, .18 + request.localContrastIntent * .58, dw * .58).coerceIn(.16, .74)
        val baseSat = when (situation) {
            AnipSituation.NIGHT_NEON, AnipSituation.STAGE -> 1.08; AnipSituation.SUNSET, AnipSituation.LANDSCAPE -> 1.07
            AnipSituation.FOOD -> 1.09; AnipSituation.DOCUMENT -> .98
            AnipSituation.PORTRAIT, AnipSituation.PORTRAIT_LOW_LIGHT, AnipSituation.SELFIE, AnipSituation.SELFIE_LOW_LIGHT -> 1.015
            else -> 1.035
        }
        val saturation = if (request.look == Look.MONOCHROME) 0.0 else lerp(baseSat, .90 + request.colorIntensityIntent * .22, dw * .66).coerceIn(.88, 1.18)
        val autoWarmth = when {
            request.look == Look.MONOCHROME -> 0.0
            request.look == Look.COOL || request.look == Look.CINEMATIC_COOL -> -.10
            request.look == Look.WARM || request.look == Look.GOLDEN_HOUR || request.look == Look.CINEMATIC_WARM -> .10
            situation == AnipSituation.SUNSET -> .065; portrait -> .025; else -> 0.0
        }
        val textureProtection = when { highTexture -> .88; portrait -> .58; else -> .68 }
        val skinProtection = if (portrait) max(.72, request.skinNaturalness) else .42
        return RenderRecipe(
            style, exposure, shadowRecovery, highlight,
            localContrast, saturation, lerp(autoWarmth, request.warmthBiasIntent, dw * .70).coerceIn(-.16, .16),
            if (style == AnipStyle.CINEMATIC || situation in setOf(AnipSituation.NIGHT, AnipSituation.NIGHT_NEON)) .07 else .015,
            subjectRelight, background,
            when (situation) { AnipSituation.LANDSCAPE -> .18; AnipSituation.ARCHITECTURE -> .14; AnipSituation.NIGHT_NEON, AnipSituation.STAGE -> .05; else -> (.055 + dynamicRange * .06).coerceIn(.04, .18) },
            (if (portrait || style == AnipStyle.CINEMATIC) .075 else .025), if (style == AnipStyle.CINEMATIC || request.look == Look.MONOCHROME) .025 else 0.0,
            RestorationRouting((.14 + noiseNeed * .58 + if (night) .08 else 0.0).coerceIn(.10, .78),
                (detailNeed * .42 * (1.0 - request.motionRisk * .50) * request.reconstructionIntegrity * textureProtection).coerceIn(.04, .48),
                (.12 + noiseNeed * .46).coerceIn(.08, .66), (.22 + detailNeed * .40 * request.reconstructionIntegrity * textureProtection).coerceIn(.16, .70),
                (.10 + detailNeed * .18).coerceIn(.08, .32)),
            (1.0 + request.lookStrength * .10 + if (style == AnipStyle.IMPACT) .06 else 0.0).coerceIn(.90, 1.20),
            situation, blackAnchor, (.18 + request.shadowDepthIntent * .62 * (.45 + .55 * dw)).coerceIn(.15, .82), glow, textureProtection, skinProtection
        )
    }

    fun candidates(base: RenderRecipe) = listOf(base.scaled(.88, AnipStyle.NATURAL_PRO), base.scaled(1.0, base.style),
        base.scaled(1.10, if (base.style == AnipStyle.CINEMATIC) AnipStyle.CINEMATIC else AnipStyle.IMPACT),
        base.scaled(1.18, if (base.style == AnipStyle.CINEMATIC) AnipStyle.CINEMATIC else AnipStyle.IMPACT))

    private val portraitSituations = setOf(AnipSituation.PORTRAIT, AnipSituation.PORTRAIT_LOW_LIGHT, AnipSituation.SELFIE, AnipSituation.SELFIE_LOW_LIGHT)
    private val darkSituations = setOf(AnipSituation.NIGHT, AnipSituation.NIGHT_NEON, AnipSituation.STAGE, AnipSituation.PORTRAIT_LOW_LIGHT, AnipSituation.SELFIE_LOW_LIGHT)
    private val textureSituations = setOf(AnipSituation.ARCHITECTURE, AnipSituation.PRODUCT, AnipSituation.MACRO, AnipSituation.DOCUMENT, AnipSituation.LANDSCAPE)
    private fun lerp(a: Double, b: Double, t: Double) = a + (b - a) * t.coerceIn(0.0, 1.0)
}

object AnipRenderFieldBuilder {
    fun build(recipe: RenderRecipe, subject: NormalizedRect?, width: Int = 48, height: Int = 48) = build(recipe, subject?.let { AnipSubjectContext(it) }, width, height)
    fun build(recipe: RenderRecipe, subject: AnipSubjectContext?, width: Int = 48, height: Int = 48): RenderField {
        val size = width * height
        val exposure = FloatArray(size); val contrast = FloatArray(size); val saturation = FloatArray(size); val warmth = FloatArray(size)
        val denoise = FloatArray(size); val detail = FloatArray(size); val dehaze = FloatArray(size); val weights = FloatArray(size)
        val b = subject?.bounds; val cx = b?.center?.x ?: .5; val cy = b?.center?.y ?: .5
        val sx = max(.10, (b?.width ?: .34) * .72); val sy = max(.14, (b?.height ?: .52) * .70); val depthBoost = (1.0 + max(0.0, subject?.depthSeparation ?: 0.0) * .30).coerceIn(1.0, 1.30)
        for (y in 0 until height) for (x in 0 until width) {
            val nx = (x + .5) / width; val ny = (y + .5) / height; val dx = (nx - cx) / sx; val dy = (ny - cy) / sy
            val gaussian = if (subject == null) 0.0 else exp(-.5 * (dx * dx + dy * dy)).coerceIn(0.0, 1.0)
            val mask = subject?.let { sampleMask(it, nx, ny) }; val semantic = if (mask == null) gaussian else mask * subject.maskConfidence + gaussian * (1.0 - subject.maskConfidence)
            val sw = (semantic * (.55 + .45 * (subject?.trackConfidence ?: 1.0))).coerceIn(0.0, 1.0); val i = y * width + x; weights[i] = sw.toFloat()
            exposure[i] = (recipe.globalExposureEv + sw * recipe.subjectRelightEv * depthBoost + (1.0 - sw) * recipe.backgroundSuppressionEv).toFloat()
            contrast[i] = (recipe.localContrast * (.72 + .28 * (1.0 - sw) + sw * .06 * depthBoost)).coerceIn(0.0, 1.0).toFloat()
            saturation[i] = if (recipe.saturation < .20) recipe.saturation.toFloat() else (1.0 + (recipe.saturation - 1.0) * (.88 + .12 * sw)).toFloat()
            warmth[i] = (recipe.warmth + sw * .015 - (1.0 - sw) * recipe.coolShadowBias * .30).toFloat()
            denoise[i] = (recipe.routing.denoise * (.92 + .10 * (1.0 - sw)) * (1.0 - sw * recipe.skinProtection * .26)).coerceIn(0.0, 1.0).toFloat()
            detail[i] = (recipe.routing.detailRecovery * (.76 + .24 * recipe.textureProtection + .18 * sw * depthBoost * (1.0 - recipe.skinProtection * .35))).coerceIn(0.0, 1.0).toFloat()
            dehaze[i] = (recipe.dehaze * (.72 + .28 * (1.0 - sw))).toFloat()
        }
        return RenderField(width, height, exposure, contrast, saturation, warmth, denoise, detail, dehaze, weights)
    }

    private fun sampleMask(subject: AnipSubjectContext, nx: Double, ny: Double): Double? {
        val m = subject.mask ?: return null; if (subject.maskWidth <= 0 || subject.maskHeight <= 0) return null
        val fx = nx.coerceIn(0.0, 1.0) * (subject.maskWidth - 1); val fy = ny.coerceIn(0.0, 1.0) * (subject.maskHeight - 1)
        val x0 = fx.toInt(); val y0 = fy.toInt(); val x1 = min(subject.maskWidth - 1, x0 + 1); val y1 = min(subject.maskHeight - 1, y0 + 1); val tx = fx - x0; val ty = fy - y0
        val p00 = m[y0 * subject.maskWidth + x0]; val p10 = m[y0 * subject.maskWidth + x1]; val p01 = m[y1 * subject.maskWidth + x0]; val p11 = m[y1 * subject.maskWidth + x1]
        return ((p00 * (1.0 - tx) + p10 * tx) * (1.0 - ty) + (p01 * (1.0 - tx) + p11 * tx) * ty).coerceIn(0.0, 1.0)
    }
}
