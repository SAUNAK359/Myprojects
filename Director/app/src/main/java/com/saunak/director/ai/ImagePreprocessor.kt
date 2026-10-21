package com.saunak.director.ai

import android.graphics.Bitmap
import kotlin.math.roundToInt

object ImagePreprocessor {
    fun forVlm(source: Bitmap, maxDimension: Int): Bitmap {
        val max = maxOf(source.width, source.height)
        if (max <= maxDimension) return source
        val scale = maxDimension.toFloat() / max.toFloat()
        val w = (source.width * scale).roundToInt().coerceAtLeast(1)
        val h = (source.height * scale).roundToInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(source, w, h, true)
    }
}
