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
    val heuristicScore: Double,
    val shadowP05: Double = 0.0,
    val medianLuma: Double = 0.0,
    val highlightP95: Double = 255.0,
    val topMeanLuma: Double = 0.0,
    val bottomMeanLuma: Double = 0.0,
    val horizontalEdgeEnergy: Double = 0.0,
    val verticalEdgeEnergy: Double = 0.0,
    val strongEdgeFraction: Double = 0.0
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
            val pixels = IntArray(w * h)
            bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
            val luma = DoubleArray(pixels.size)
            var sum = 0.0
            var dark = 0
            var bright = 0
            var topSum = 0.0
            var topCount = 0
            var bottomSum = 0.0
            var bottomCount = 0
            val topEnd = maxOf(1, h / 3)
            val bottomStart = minOf(h - 1, (h * 2) / 3)

            for (i in pixels.indices) {
                val p = pixels[i]
                val y = 0.2126 * ((p shr 16) and 255) + 0.7152 * ((p shr 8) and 255) + 0.0722 * (p and 255)
                luma[i] = y
                sum += y
                if (y < 3.0) dark++
                if (y > 252.0) bright++
                val row = i / w
                if (row < topEnd) {
                    topSum += y
                    topCount++
                }
                if (row >= bottomStart) {
                    bottomSum += y
                    bottomCount++
                }
            }

            val mean = sum / luma.size
            var variance = 0.0
            var horizontalEdges = 0.0
            var verticalEdges = 0.0
            var horizontalCount = 0
            var verticalCount = 0
            var strongEdges = 0

            for (y in 0 until h) for (x in 0 until w) {
                val i = y * w + x
                val d = luma[i] - mean
                variance += d * d
                if (x + 1 < w) {
                    val delta = abs(luma[i] - luma[i + 1])
                    horizontalEdges += delta
                    horizontalCount++
                    if (delta >= 18.0) strongEdges++
                }
                if (y + 1 < h) {
                    val delta = abs(luma[i] - luma[i + w])
                    verticalEdges += delta
                    verticalCount++
                    if (delta >= 18.0) strongEdges++
                }
            }

            val std = sqrt(variance / luma.size)
            val darkF = dark.toDouble() / luma.size
            val brightF = bright.toDouble() / luma.size
            val edgeCount = horizontalCount + verticalCount
            val edge = if (edgeCount == 0) 0.0 else (horizontalEdges + verticalEdges) / edgeCount
            val hEdge = if (horizontalCount == 0) 0.0 else horizontalEdges / horizontalCount
            val vEdge = if (verticalCount == 0) 0.0 else verticalEdges / verticalCount
            val strongEdgeFraction = if (edgeCount == 0) 0.0 else strongEdges.toDouble() / edgeCount

            val sorted = luma.clone().apply { sort() }
            fun percentile(fraction: Double): Double {
                val index = ((sorted.size - 1) * fraction).toInt().coerceIn(0, sorted.lastIndex)
                return sorted[index]
            }
            val p05 = percentile(0.05)
            val p50 = percentile(0.50)
            val p95 = percentile(0.95)
            val topMean = if (topCount == 0) mean else topSum / topCount
            val bottomMean = if (bottomCount == 0) mean else bottomSum / bottomCount

            val catastrophic = (darkF > .995 || brightF > .995) && edge < 0.5
            val exposureScore = (1.0 - (darkF + brightF).coerceIn(0.0, 1.0)).coerceIn(0.0, 1.0)
            val detailScore = (edge / 12.0).coerceIn(0.0, 1.0)
            val dynamicScore = (std / 45.0).coerceIn(0.0, 1.0)
            val score = if (catastrophic) 0.0 else (
                0.45 * exposureScore + 0.30 * detailScore + 0.25 * dynamicScore
            ).coerceIn(0.0, 1.0)

            return TechnicalQuality(
                meanLuma = mean,
                lumaStdDev = std,
                darkClipFraction = darkF,
                brightClipFraction = brightF,
                edgeEnergy = edge,
                catastrophic = catastrophic,
                heuristicScore = score,
                shadowP05 = p05,
                medianLuma = p50,
                highlightP95 = p95,
                topMeanLuma = topMean,
                bottomMeanLuma = bottomMean,
                horizontalEdgeEnergy = hEdge,
                verticalEdgeEnergy = vEdge,
                strongEdgeFraction = strongEdgeFraction
            )
        } finally {
            if (bitmap !== source) bitmap.recycle()
        }
    }
}
