package com.saunak.director.engine.reconstruction

import kotlin.math.max
import kotlin.math.pow

data class WhiteBalanceGains(
    val red: Float,
    val green: Float,
    val blue: Float
) {
    init {
        require(red > 0f && green > 0f && blue > 0f)
    }
}

data class Matrix3x3(val m: FloatArray) {
    init {
        require(m.size == 9)
        require(m.all { it.isFinite() })
    }

    fun transform(r: Float, g: Float, b: Float): Triple<Float, Float, Float> = Triple(
        m[0] * r + m[1] * g + m[2] * b,
        m[3] * r + m[4] * g + m[5] * b,
        m[6] * r + m[7] * g + m[8] * b
    )

    companion object {
        val IDENTITY = Matrix3x3(floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f))
    }
}

data class SierraColorCalibration(
    val whiteBalance: WhiteBalanceGains,
    /** Camera-linear RGB -> linear sRGB/D65 working transform for the active sensor/illuminant. */
    val cameraToWorking: Matrix3x3,
    val exposureGain: Float = 1f,
    val highlightShoulder: Float = 0.82f,
    val contrast: Float = 1f
) {
    init {
        require(exposureGain > 0f)
        require(highlightShoulder in 0.5f..0.98f)
        require(contrast in 0.5f..1.8f)
    }
}

data class DisplayRgbImage(
    val width: Int,
    val height: Int,
    /** Interleaved nonlinear display RGB in [0,1]. */
    val rgb: FloatArray
) {
    init {
        require(width > 0 && height > 0)
        require(rgb.size == width * height * 3)
        require(rgb.all { it.isFinite() && it in 0f..1f })
    }
}

/**
 * Minimal physically ordered ISP baseline. Every operation before tone/output transfer remains in
 * linear light. Learned/semantic rendering may later operate after this calibrated foundation.
 */
object DeterministicIsp {
    fun render(linearCamera: LinearRgbImage, calibration: SierraColorCalibration): DisplayRgbImage {
        val src = linearCamera.rgb
        val out = FloatArray(src.size)
        val wb = calibration.whiteBalance

        var i = 0
        while (i < src.size) {
            var r = max(0f, src[i] * wb.red * calibration.exposureGain)
            var g = max(0f, src[i + 1] * wb.green * calibration.exposureGain)
            var b = max(0f, src[i + 2] * wb.blue * calibration.exposureGain)

            val working = calibration.cameraToWorking.transform(r, g, b)
            r = max(0f, working.first)
            g = max(0f, working.second)
            b = max(0f, working.third)

            // Highlight shoulder is applied channel-wise in linear light to avoid hard clipping.
            r = shoulder(r, calibration.highlightShoulder)
            g = shoulder(g, calibration.highlightShoulder)
            b = shoulder(b, calibration.highlightShoulder)

            // Luminance-preserving contrast around middle grey.
            val luma = 0.2126f * r + 0.7152f * g + 0.0722f * b
            val targetLuma = contrastCurve(luma, calibration.contrast)
            val scale = if (luma > 1e-6f) targetLuma / luma else 1f
            r *= scale; g *= scale; b *= scale

            out[i] = srgbOetf(r.coerceIn(0f, 1f))
            out[i + 1] = srgbOetf(g.coerceIn(0f, 1f))
            out[i + 2] = srgbOetf(b.coerceIn(0f, 1f))
            i += 3
        }
        return DisplayRgbImage(linearCamera.width, linearCamera.height, out)
    }

    private fun shoulder(value: Float, knee: Float): Float {
        if (value <= knee) return value
        val excess = value - knee
        val headroom = 1f - knee
        return (knee + headroom * (excess / (excess + headroom))).coerceAtMost(1f)
    }

    private fun contrastCurve(value: Float, contrast: Float): Float {
        val pivot = 0.18f
        if (value <= 0f) return 0f
        return (pivot * (value / pivot).toDouble().pow(contrast.toDouble()).toFloat()).coerceIn(0f, 1f)
    }

    private fun srgbOetf(linear: Float): Float =
        if (linear <= 0.0031308f) 12.92f * linear
        else (1.055 * linear.toDouble().pow(1.0 / 2.4) - 0.055).toFloat()
}
