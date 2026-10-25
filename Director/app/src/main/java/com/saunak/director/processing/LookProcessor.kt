package com.saunak.director.processing

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.RadialGradient
import android.graphics.Shader
import com.saunak.director.engine.SierraEngine
import com.saunak.director.engine.render.AnipEngine
import com.saunak.director.engine.render.AnipRequest
import com.saunak.director.engine.render.AnipSubjectContext

enum class Look {
    NATURAL,
    WARM,
    COOL,
    CINEMATIC_WARM,
    CINEMATIC_COOL,
    MONOCHROME,
    SOFT_PORTRAIT,
    LANDSCAPE_VIVID,
    SOCIAL_POP,
    FILM_TEAL_ORANGE,
    GOLDEN_HOUR,
    MOODY,
    CLEAN_PORTRAIT,
    NIGHT_NEON
}

data class LookSpec(val look: Look = Look.NATURAL, val strength: Float = 0f) {
    val normalizedStrength: Float get() = strength.coerceIn(0f, 1f)
    fun isIdentity(): Boolean = look == Look.NATURAL || normalizedStrength <= 0.001f
}

object LookProcessor {
    /**
     * Sierra v8 routes normal finishing through ANIP, even for NATURAL. The legacy color-matrix
     * implementation remains as a fail-safe so rendering can never make capture/save unavailable.
     */
    fun apply(source: Bitmap, spec: LookSpec): Bitmap {
        val engine = SierraEngine.getOrNull()
        if (engine != null && !source.isRecycled) {
            val rendered = runCatching {
                val track = engine.subjects.latest()
                val subject = track?.let {
                    val mask = it.mask
                    AnipSubjectContext(
                        bounds = it.bounds,
                        maskWidth = mask?.width ?: 0,
                        maskHeight = mask?.height ?: 0,
                        mask = mask?.probabilities,
                        maskConfidence = mask?.confidence ?: 0.0,
                        depthSeparation = it.depth?.foregroundSeparation ?: 0.0,
                        trackConfidence = it.confidence
                    )
                }
                val request = AnipRequest(
                    sceneLabel = "",
                    objective = "",
                    look = spec.look,
                    lookStrength = spec.normalizedStrength,
                    subjectPresent = subject != null,
                    motionRisk = engine.motion.latest().conservativeRisk,
                    reconstructionIntegrity = 0.78
                )
                AnipEngine().render(source, request, subject).first
            }.getOrNull()
            if (rendered != null) return rendered
        }
        return legacyApply(source, spec)
    }

    private fun legacyApply(source: Bitmap, spec: LookSpec): Bitmap {
        if (spec.isIdentity()) return source
        val s = spec.normalizedStrength
        val target = when (spec.look) {
            Look.WARM -> matrix(sat = 1.03f, r = 1.06f, g = 1.01f, b = 0.95f)
            Look.COOL -> matrix(sat = 1.02f, r = 0.96f, g = 1.0f, b = 1.06f)
            Look.CINEMATIC_WARM -> matrix(sat = 0.96f, contrast = 1.06f, r = 1.06f, g = 1.0f, b = 0.93f, lift = -2f)
            Look.CINEMATIC_COOL -> matrix(sat = 0.93f, contrast = 1.07f, r = 0.94f, g = 1.01f, b = 1.07f, lift = -3f)
            Look.MONOCHROME -> ColorMatrix().apply { setSaturation(0f) }
            Look.SOFT_PORTRAIT -> matrix(sat = 0.96f, contrast = 0.97f, r = 1.025f, g = 1.005f, b = 0.985f, lift = 2f)
            Look.LANDSCAPE_VIVID -> matrix(sat = 1.07f, contrast = 1.045f, g = 1.008f, b = 1.005f, lift = -1f)
            Look.SOCIAL_POP -> matrix(sat = 1.06f, contrast = 1.035f, r = 1.01f, g = 1.008f, b = 1.004f, lift = 1f)
            Look.FILM_TEAL_ORANGE -> matrix(sat = 0.98f, contrast = 1.075f, r = 1.045f, g = 0.995f, b = 1.035f, lift = -4f)
            Look.GOLDEN_HOUR -> matrix(sat = 1.04f, contrast = 1.025f, r = 1.08f, g = 1.025f, b = 0.91f, lift = 3f)
            Look.MOODY -> matrix(sat = 0.88f, contrast = 1.10f, r = 0.97f, g = 0.99f, b = 1.015f, lift = -7f)
            Look.CLEAN_PORTRAIT -> matrix(sat = 0.99f, contrast = 1.015f, r = 1.018f, g = 1.012f, b = 0.995f, lift = 2f)
            Look.NIGHT_NEON -> matrix(sat = 1.12f, contrast = 1.09f, r = 1.02f, g = 1.0f, b = 1.08f, lift = -6f)
            Look.NATURAL -> ColorMatrix()
        }

        val a = target.array
        val blended = FloatArray(20) { i ->
            val identity = when (i) { 0, 6, 12, 18 -> 1f; else -> 0f }
            identity + (a[i] - identity) * s
        }

        val out = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawBitmap(
            source,
            0f,
            0f,
            Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
                colorFilter = ColorMatrixColorFilter(ColorMatrix(blended))
            }
        )

        val vignette = when (spec.look) {
            Look.CINEMATIC_WARM, Look.CINEMATIC_COOL, Look.FILM_TEAL_ORANGE -> 0.15f * s
            Look.MOODY -> 0.22f * s
            Look.NIGHT_NEON -> 0.18f * s
            else -> 0f
        }
        if (vignette > 0.001f) {
            val radius = maxOf(out.width, out.height) * 0.72f
            val shader = RadialGradient(
                out.width / 2f,
                out.height / 2f,
                radius,
                intArrayOf(Color.TRANSPARENT, Color.TRANSPARENT, Color.argb((255 * vignette).toInt(), 0, 0, 0)),
                floatArrayOf(0f, 0.62f, 1f),
                Shader.TileMode.CLAMP
            )
            canvas.drawRect(0f, 0f, out.width.toFloat(), out.height.toFloat(), Paint(Paint.ANTI_ALIAS_FLAG).apply { this.shader = shader })
        }
        return out
    }

    private fun matrix(
        sat: Float,
        contrast: Float = 1f,
        r: Float = 1f,
        g: Float = 1f,
        b: Float = 1f,
        lift: Float = 0f
    ): ColorMatrix {
        val m = ColorMatrix().apply { setSaturation(sat) }
        val t = (1f - contrast) * 128f + lift
        m.postConcat(
            ColorMatrix(
                floatArrayOf(
                    contrast * r, 0f, 0f, 0f, t,
                    0f, contrast * g, 0f, 0f, t,
                    0f, 0f, contrast * b, 0f, t,
                    0f, 0f, 0f, 1f, 0f
                )
            )
        )
        return m
    }
}
