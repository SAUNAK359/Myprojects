package ai.auraforge.director.quality

import ai.auraforge.director.core.ImageQuality
import ai.auraforge.director.core.LuminanceGrid
import android.graphics.Bitmap
import kotlin.math.sqrt

class ImageQualityAnalyzer {
    fun analyze(bitmap: Bitmap, cameraMotion: Float, rollDeg: Float): ImageQuality {
        val width = bitmap.width
        val height = bitmap.height
        if (width < 3 || height < 3) return ImageQuality(cameraMotion = cameraMotion, deviceRollDeg = rollDeg)

        val step = when {
            width * height > 1_000_000 -> 8
            width * height > 400_000 -> 6
            else -> 4
        }

        var n = 0
        var sum = 0.0
        var sumSq = 0.0
        var highlights = 0
        var shadows = 0
        var gradSum = 0.0
        var gradSq = 0.0
        var gradN = 0

        var y = step
        while (y < height - step) {
            var x = step
            while (x < width - step) {
                val lum = luma(bitmap.getPixel(x, y))
                n++
                sum += lum
                sumSq += lum * lum
                if (lum > 0.97) highlights++
                if (lum < 0.03) shadows++

                val gx = luma(bitmap.getPixel(x + step, y)) - luma(bitmap.getPixel(x - step, y))
                val gy = luma(bitmap.getPixel(x, y + step)) - luma(bitmap.getPixel(x, y - step))
                val g = sqrt(gx * gx + gy * gy)
                gradSum += g
                gradSq += g * g
                gradN++
                x += step
            }
            y += step
        }

        if (n == 0) return ImageQuality(cameraMotion = cameraMotion, deviceRollDeg = rollDeg)
        val mean = (sum / n).toFloat()
        val variance = (sumSq / n - (sum / n) * (sum / n)).coerceAtLeast(0.0)
        val contrast = (sqrt(variance) * 3.0).toFloat().coerceIn(0f, 1f)

        val gradMean = if (gradN > 0) gradSum / gradN else 0.0
        val gradVariance = if (gradN > 0) (gradSq / gradN - gradMean * gradMean).coerceAtLeast(0.0) else 0.0
        val sharpness = ((gradMean * 3.3) + sqrt(gradVariance) * 2.0).toFloat().coerceIn(0f, 1f)

        return ImageQuality(
            meanLuma = mean,
            contrast = contrast,
            highlightClipping = highlights.toFloat() / n,
            shadowClipping = shadows.toFloat() / n,
            sharpness = sharpness,
            cameraMotion = cameraMotion.coerceIn(0f, 1f),
            deviceRollDeg = rollDeg,
            lumaGrid = buildLuminanceGrid(bitmap, GRID_COLUMNS, GRID_ROWS)
        )
    }

    private fun buildLuminanceGrid(bitmap: Bitmap, columns: Int, rows: Int): LuminanceGrid {
        val values = ArrayList<Float>(columns * rows)
        for (row in 0 until rows) {
            val y0 = row * bitmap.height / rows
            val y1 = ((row + 1) * bitmap.height / rows).coerceAtMost(bitmap.height)
            for (column in 0 until columns) {
                val x0 = column * bitmap.width / columns
                val x1 = ((column + 1) * bitmap.width / columns).coerceAtMost(bitmap.width)
                val sx = maxOf(1, (x1 - x0) / 5)
                val sy = maxOf(1, (y1 - y0) / 5)
                var sum = 0.0
                var count = 0
                var y = y0
                while (y < y1) {
                    var x = x0
                    while (x < x1) {
                        sum += luma(bitmap.getPixel(x.coerceAtMost(bitmap.width - 1), y.coerceAtMost(bitmap.height - 1)))
                        count++
                        x += sx
                    }
                    y += sy
                }
                values += if (count == 0) 0.5f else (sum / count).toFloat()
            }
        }
        return LuminanceGrid(columns, rows, values)
    }

    private fun luma(pixel: Int): Double {
        val r = (pixel shr 16) and 0xff
        val g = (pixel shr 8) and 0xff
        val b = pixel and 0xff
        return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    }

    fun scoreJpegThumbnail(bitmap: Bitmap): Float {
        val q = analyze(bitmap, 0f, 0f)
        val clippingPenalty = (q.highlightClipping + q.shadowClipping).coerceIn(0f, 0.5f)
        return (0.48f * q.sharpness + 0.30f * q.exposureScore + 0.22f * q.contrast - 0.25f * clippingPenalty)
            .coerceIn(0f, 1f)
    }

    companion object {
        private const val GRID_COLUMNS = 8
        private const val GRID_ROWS = 8
    }
}
