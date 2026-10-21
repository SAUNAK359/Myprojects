package com.saunak.director.processing

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint

enum class Look { NATURAL, WARM, COOL, CINEMATIC_WARM, CINEMATIC_COOL, MONOCHROME, SOFT_PORTRAIT, LANDSCAPE_VIVID }
data class LookSpec(val look: Look = Look.NATURAL, val strength: Float = 0f) {
    val normalizedStrength: Float get() = strength.coerceIn(0f, 1f)
    fun isIdentity(): Boolean = look == Look.NATURAL || normalizedStrength <= 0.001f
}

object LookProcessor {
    fun apply(source: Bitmap, spec: LookSpec): Bitmap {
        if (spec.isIdentity()) return source
        val target = when (spec.look) {
            Look.WARM -> matrix(sat = 1.03f, r = 1.06f, g = 1.01f, b = 0.95f)
            Look.COOL -> matrix(sat = 1.02f, r = 0.96f, g = 1.0f, b = 1.06f)
            Look.CINEMATIC_WARM -> matrix(sat = 0.96f, contrast = 1.05f, r = 1.05f, g = 1.0f, b = 0.94f)
            Look.CINEMATIC_COOL -> matrix(sat = 0.92f, contrast = 1.06f, r = 0.95f, g = 1.01f, b = 1.06f)
            Look.MONOCHROME -> ColorMatrix().apply { setSaturation(0f) }
            Look.SOFT_PORTRAIT -> matrix(sat = 0.95f, contrast = 0.97f, r = 1.02f, g = 1.0f, b = 0.98f)
            Look.LANDSCAPE_VIVID -> matrix(sat = 1.09f, contrast = 1.04f)
            Look.NATURAL -> ColorMatrix()
        }
        val strength = spec.normalizedStrength
        val a = target.array
        val blended = FloatArray(20) { i ->
            val identity = when (i) { 0, 6, 12, 18 -> 1f; else -> 0f }
            identity + (a[i] - identity) * strength
        }
        val out = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
        Canvas(out).drawBitmap(source, 0f, 0f, Paint(Paint.ANTI_ALIAS_FLAG).apply { colorFilter = ColorMatrixColorFilter(ColorMatrix(blended)) })
        return out
    }

    private fun matrix(sat: Float, contrast: Float = 1f, r: Float = 1f, g: Float = 1f, b: Float = 1f): ColorMatrix {
        val m = ColorMatrix().apply { setSaturation(sat) }
        val t = (1f - contrast) * 128f
        m.postConcat(ColorMatrix(floatArrayOf(contrast*r,0f,0f,0f,t, 0f,contrast*g,0f,0f,t, 0f,0f,contrast*b,0f,t, 0f,0f,0f,1f,0f)))
        return m
    }
}
