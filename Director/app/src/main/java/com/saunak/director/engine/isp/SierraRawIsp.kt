package com.saunak.director.engine.isp

import android.graphics.Bitmap
import com.saunak.director.camera.RawJpegBurstCapture
import com.saunak.director.engine.device.CameraModuleTwin
import com.saunak.director.engine.device.CfaPattern
import com.saunak.director.engine.reconstruction.LinearRgbTile
import com.saunak.director.engine.reconstruction.LinearRgbTileSink
import com.saunak.director.engine.reconstruction.SierraRawContainer
import java.util.Arrays
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/** Immutable RAW rendering parameters derived from sensor evidence and Camera2 calibration. */
data class SierraIspPlan(
    val redGain: Double,
    val greenGain: Double,
    val blueGain: Double,
    /** Row-major camera-linear RGB -> linear sRGB matrix. */
    val cameraToSrgb: DoubleArray,
    val exposureScale: Double,
    val blackPoint: Double,
    val confidence: Double
) {
    init {
        require(redGain in 0.25..6.0 && greenGain in 0.25..6.0 && blueGain in 0.25..6.0)
        require(cameraToSrgb.size == 9 && cameraToSrgb.all { it.isFinite() })
        require(exposureScale in 0.125..16.0)
        require(blackPoint in 0.0..0.1)
        require(confidence in 0.0..1.0)
    }
}

/**
 * Sensor-statistics ISP planner. White balance is estimated in RAW space from unclipped samples,
 * then Camera2 forward colorimetry is used when available. No semantic/generative modification is
 * performed in this deterministic truth path.
 */
object SierraRawIspPlanner {
    fun plan(capture: RawJpegBurstCapture, camera: CameraModuleTwin): SierraIspPlan {
        val referenceIndex = capture.frames.indices.minByOrNull { kotlin.math.abs(capture.frames[it].requested.evOffset) } ?: 0
        SierraRawContainer.open(capture.frames[referenceIndex].rawFile).use { raw ->
            val white = camera.whiteLevel?.toDouble()?.takeIf { it > 0 } ?: error("Sensor white level unavailable")
            val targetSamples = 12_000
            val stride = max(2, kotlin.math.sqrt((raw.header.width.toDouble() * raw.header.height) / targetSamples).toInt())
            val channelLogs = Array(3) { DoubleArray(targetSamples) }
            val counts = IntArray(3)
            val luminance = DoubleArray(targetSamples)
            var lumCount = 0

            var y = 0
            while (y < raw.header.height) {
                var x = 0
                while (x < raw.header.width) {
                    val black = blackLevel(camera, x, y).toDouble()
                    val signal = ((raw.sampleUnsigned16(x, y) - black) / max(1.0, white - black)).coerceIn(0.0, 1.0)
                    if (signal in 0.015..0.92) {
                        val channel = channelIndex(camera.cfaPattern, x, y)
                        val idx = counts[channel]
                        if (idx < channelLogs[channel].size) {
                            channelLogs[channel][idx] = ln(max(1e-6, signal))
                            counts[channel]++
                        }
                        if (channel == 1 && lumCount < luminance.size) luminance[lumCount++] = signal
                    }
                    x += stride
                }
                y += stride
            }

            val rMean = geometricTrimmedMean(channelLogs[0], counts[0])
            val gMean = geometricTrimmedMean(channelLogs[1], counts[1])
            val bMean = geometricTrimmedMean(channelLogs[2], counts[2])
            val redGain = (gMean / max(1e-6, rMean)).coerceIn(0.45, 3.5)
            val blueGain = (gMean / max(1e-6, bMean)).coerceIn(0.45, 3.5)

            val p92 = if (lumCount > 0) {
                Arrays.sort(luminance, 0, lumCount)
                luminance[((lumCount - 1) * 0.92).toInt().coerceIn(0, lumCount - 1)]
            } else 0.18
            val exposureScale = (0.72 / max(0.03, p92)).coerceIn(0.35, 6.0)
            val matrix = cameraToSrgb(camera)
            val sampleConfidence = (min(1.0, counts.sum() / 4000.0) * if (camera.colorCalibration?.forwardMatrix1?.size == 9) 1.0 else 0.72)
                .coerceIn(0.0, 1.0)
            return SierraIspPlan(redGain, 1.0, blueGain, matrix, exposureScale, 0.0035, sampleConfidence)
        }
    }

    private fun geometricTrimmedMean(values: DoubleArray, count: Int): Double {
        if (count <= 0) return 0.18
        Arrays.sort(values, 0, count)
        val lo = (count * 0.12).toInt()
        val hi = max(lo + 1, (count * 0.88).toInt()).coerceAtMost(count)
        var sum = 0.0
        for (i in lo until hi) sum += values[i]
        return kotlin.math.exp(sum / (hi - lo))
    }

    private fun blackLevel(camera: CameraModuleTwin, x: Int, y: Int): Int {
        if (camera.blackLevelPattern.size != 4) return 0
        return camera.blackLevelPattern[(y and 1) * 2 + (x and 1)]
    }

    private fun channelIndex(pattern: CfaPattern, x: Int, y: Int): Int {
        val phase = (y and 1) * 2 + (x and 1)
        return when (pattern) {
            CfaPattern.RGGB -> when (phase) { 0 -> 0; 3 -> 2; else -> 1 }
            CfaPattern.BGGR -> when (phase) { 0 -> 2; 3 -> 0; else -> 1 }
            CfaPattern.GRBG -> when (phase) { 1 -> 0; 2 -> 2; else -> 1 }
            CfaPattern.GBRG -> when (phase) { 1 -> 2; 2 -> 0; else -> 1 }
            else -> error("Unsupported Bayer pattern $pattern")
        }
    }

    private fun cameraToSrgb(camera: CameraModuleTwin): DoubleArray {
        val forward = camera.colorCalibration?.forwardMatrix1
        if (forward == null || forward.size != 9) return IDENTITY.copyOf()
        return multiply3x3(XYZ_D65_TO_SRGB, multiply3x3(D50_TO_D65, forward.toDoubleArray()))
    }

    private fun multiply3x3(a: DoubleArray, b: DoubleArray): DoubleArray {
        val out = DoubleArray(9)
        for (r in 0..2) for (c in 0..2) {
            var v = 0.0
            for (k in 0..2) v += a[r * 3 + k] * b[k * 3 + c]
            out[r * 3 + c] = v
        }
        return out
    }

    private val IDENTITY = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    private val D50_TO_D65 = doubleArrayOf(
        0.9555766, -0.0230393, 0.0631636,
        -0.0282895, 1.0099416, 0.0210077,
        0.0122982, -0.0204830, 1.3299098
    )
    private val XYZ_D65_TO_SRGB = doubleArrayOf(
        3.2404542, -1.5371385, -0.4985314,
        -0.9692660, 1.8760108, 0.0415560,
        0.0556434, -0.2040259, 1.0572252
    )
}

/** Streaming tile sink that renders the deterministic linear reconstruction into display sRGB. */
class SierraIspBitmapSink(
    private val plan: SierraIspPlan,
    private val maxOutputPixels: Long = 30_000_000L
) : LinearRgbTileSink {
    private var output: Bitmap? = null
    private var width: Int = 0
    private var height: Int = 0

    override fun begin(width: Int, height: Int) {
        require(width > 0 && height > 0)
        require(width.toLong() * height.toLong() <= maxOutputPixels) {
            "Computational output ${width}x$height exceeds bounded bitmap budget"
        }
        this.width = width
        this.height = height
        output = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    }

    override fun accept(tile: LinearRgbTile) {
        val bitmap = checkNotNull(output) { "ISP sink not started" }
        require(tile.x + tile.width <= width && tile.y + tile.height <= height)
        val pixels = IntArray(tile.width * tile.height)
        var p = 0
        for (i in pixels.indices) {
            val confidence = tile.confidence[i].toDouble().coerceIn(0.0, 1.0)
            val sourceR = max(0.0, tile.rgb[p++].toDouble() - plan.blackPoint) * plan.redGain
            val sourceG = max(0.0, tile.rgb[p++].toDouble() - plan.blackPoint) * plan.greenGain
            val sourceB = max(0.0, tile.rgb[p++].toDouble() - plan.blackPoint) * plan.blueGain
            val m = plan.cameraToSrgb
            var r = m[0] * sourceR + m[1] * sourceG + m[2] * sourceB
            var g = m[3] * sourceR + m[4] * sourceG + m[5] * sourceB
            var b = m[6] * sourceR + m[7] * sourceG + m[8] * sourceB

            // Reduce amplification in low-evidence areas instead of synthesizing unsupported detail.
            val evidenceScale = 0.78 + 0.22 * confidence
            r = aces(max(0.0, r) * plan.exposureScale * evidenceScale)
            g = aces(max(0.0, g) * plan.exposureScale * evidenceScale)
            b = aces(max(0.0, b) * plan.exposureScale * evidenceScale)
            val ir = (srgbOetf(r) * 255.0 + 0.5).toInt().coerceIn(0, 255)
            val ig = (srgbOetf(g) * 255.0 + 0.5).toInt().coerceIn(0, 255)
            val ib = (srgbOetf(b) * 255.0 + 0.5).toInt().coerceIn(0, 255)
            pixels[i] = (0xff shl 24) or (ir shl 16) or (ig shl 8) or ib
        }
        bitmap.setPixels(pixels, 0, tile.width, tile.x, tile.y, tile.width, tile.height)
    }

    override fun end() = Unit

    fun bitmap(): Bitmap = checkNotNull(output) { "ISP has not produced an image" }

    private fun aces(x: Double): Double {
        val a = 2.51
        val b = 0.03
        val c = 2.43
        val d = 0.59
        val e = 0.14
        return ((x * (a * x + b)) / (x * (c * x + d) + e)).coerceIn(0.0, 1.0)
    }

    private fun srgbOetf(x: Double): Double = if (x <= 0.0031308) 12.92 * x else 1.055 * x.pow(1.0 / 2.4) - 0.055
}
