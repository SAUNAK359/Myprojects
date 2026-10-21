package com.saunak.director.processing

import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.sqrt

data class TechnicalQuality(
    val meanLuma: Double,
    val lumaStdDev: Double,
    val darkClipFraction: Double,
    val brightClipFraction: Double,
    val edgeEnergy: Double,
    val catastrophic: Boolean,
    val heuristicScore: Double
)

object PhotoQuality {
    fun analyze(source: Bitmap): TechnicalQuality {
        require(source.width > 0 && source.height > 0)
        val maxDim = 256
        val scale = minOf(1.0, maxDim.toDouble() / maxOf(source.width, source.height))
        val w = maxOf(1, (source.width * scale).toInt())
        val h = maxOf(1, (source.height * scale).toInt())
        val bitmap = if (w == source.width && h == source.height) source else Bitmap.createScaledBitmap(source, w, h, true)
        try {
            val pixels = IntArray(w * h); bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
            val luma = DoubleArray(pixels.size)
            var sum = 0.0; var dark = 0; var bright = 0
            for (i in pixels.indices) {
                val p = pixels[i]; val y = 0.2126 * ((p shr 16) and 255) + 0.7152 * ((p shr 8) and 255) + 0.0722 * (p and 255)
                luma[i] = y; sum += y; if (y < 3.0) dark++; if (y > 252.0) bright++
            }
            val mean = sum / luma.size
            var variance = 0.0; var edges = 0.0; var edgeCount = 0
            for (y in 0 until h) for (x in 0 until w) {
                val i = y * w + x; val d = luma[i] - mean; variance += d * d
                if (x + 1 < w) { edges += abs(luma[i] - luma[i + 1]); edgeCount++ }
                if (y + 1 < h) { edges += abs(luma[i] - luma[i + w]); edgeCount++ }
            }
            val std = sqrt(variance / luma.size)
            val darkF = dark.toDouble() / luma.size; val brightF = bright.toDouble() / luma.size
            val edge = if (edgeCount == 0) 0.0 else edges / edgeCount
            val catastrophic = (darkF > .995 || brightF > .995) && edge < 0.5
            val exposureScore = (1.0 - (darkF + brightF).coerceIn(0.0, 1.0)).coerceIn(0.0, 1.0)
            val detailScore = (edge / 12.0).coerceIn(0.0, 1.0)
            val dynamicScore = (std / 45.0).coerceIn(0.0, 1.0)
            val score = if (catastrophic) 0.0 else (0.45 * exposureScore + 0.3 * detailScore + 0.25 * dynamicScore).coerceIn(0.0, 1.0)
            return TechnicalQuality(mean, std, darkF, brightF, edge, catastrophic, score)
        } finally { if (bitmap !== source) bitmap.recycle() }
    }
}
