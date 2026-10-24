package com.saunak.director.engine.reconstruction

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

enum class BayerPattern { RGGB, BGGR, GRBG, GBRG }

/** 2x2 sensor black offsets in row-major CFA order: [y0x0, y0x1, y1x0, y1x1]. */
data class CfaBlackLevel(val offsets: FloatArray) {
    init {
        require(offsets.size == 4)
        require(offsets.all { it.isFinite() && it >= 0f })
    }

    operator fun get(x: Int, y: Int): Float = offsets[((y and 1) shl 1) or (x and 1)]

    companion object {
        fun uniform(value: Float): CfaBlackLevel = CfaBlackLevel(FloatArray(4) { value })
    }
}

data class RawPlane(
    val width: Int,
    val height: Int,
    val values: FloatArray,
    val pattern: BayerPattern,
    val blackLevel: Float,
    val whiteLevel: Float,
    val blackLevelPattern: CfaBlackLevel = CfaBlackLevel.uniform(blackLevel)
) {
    init {
        require(width > 1 && height > 1)
        require(values.size == width * height)
        require(blackLevel >= 0f)
        require(whiteLevel > blackLevelPattern.offsets.max())
    }

    operator fun get(x: Int, y: Int): Float = values[y * width + x]
}

data class NoiseModel(
    val shotNoiseSlope: Float,
    val readNoiseVariance: Float
) {
    init {
        require(shotNoiseSlope >= 0f)
        require(readNoiseVariance >= 0f)
    }

    fun variance(signal: Float): Float =
        (readNoiseVariance + shotNoiseSlope * max(signal, 0f)).coerceAtLeast(MIN_VARIANCE)

    companion object { private const val MIN_VARIANCE = 1e-8f }
}

data class RawObservation(
    val raw: RawPlane,
    val exposureTimeNs: Long,
    val sensitivityIso: Int,
    val alignmentConfidence: Float = 1f,
    val motionConfidence: Float = 1f,
    val saturationConfidence: Float = 1f,
    val noise: NoiseModel
) {
    init {
        require(exposureTimeNs > 0)
        require(sensitivityIso > 0)
        require(alignmentConfidence in 0f..1f)
        require(motionConfidence in 0f..1f)
        require(saturationConfidence in 0f..1f)
    }
}

data class LinearRgbImage(
    val width: Int,
    val height: Int,
    val rgb: FloatArray
) {
    init {
        require(width > 0 && height > 0)
        require(rgb.size == width * height * 3)
    }
}

/** Deterministic evidence-preserving RAW image formation baseline. */
object RawImageFormation {
    fun normalize(raw: RawPlane): FloatArray = FloatArray(raw.values.size) { i ->
        val x = i % raw.width
        val y = i / raw.width
        val black = raw.blackLevelPattern[x, y]
        val scale = 1f / (raw.whiteLevel - black)
        ((raw.values[i] - black) * scale).coerceIn(0f, 1f)
    }

    /**
     * Robust variance-weighted burst fusion for already geometrically aligned mosaic observations.
     * Every frame keeps its CFA layout; no RGB warp or demosaic has happened yet.
     */
    fun fuseAligned(observations: List<RawObservation>): RawPlane {
        require(observations.isNotEmpty())
        val reference = observations.first().raw
        require(observations.all {
            it.raw.width == reference.width && it.raw.height == reference.height && it.raw.pattern == reference.pattern
        })

        val normalized = observations.map { normalize(it.raw) }
        val referenceEnergy = observations.first().exposureTimeNs.toDouble() * observations.first().sensitivityIso.toDouble()
        val out = FloatArray(reference.values.size)

        for (i in out.indices) {
            var weighted = 0.0
            var weightSum = 0.0
            val referenceValue = normalized.first()[i]

            for (j in observations.indices) {
                val obs = observations[j]
                val energy = obs.exposureTimeNs.toDouble() * obs.sensitivityIso.toDouble()
                val exposureScale = (referenceEnergy / energy).coerceIn(0.125, 8.0)
                val value = (normalized[j][i] * exposureScale).coerceAtLeast(0.0)
                val noiseVariance = obs.noise.variance(normalized[j][i]).toDouble() * exposureScale * exposureScale

                val residual = abs(value - referenceValue.toDouble())
                val sigma = sqrt(noiseVariance + obs.noise.variance(referenceValue).toDouble())
                val robust = exp(-0.5 * (residual / max(3.0 * sigma, 1e-4)).let { it * it })
                val support = obs.alignmentConfidence.toDouble() *
                    obs.motionConfidence.toDouble() *
                    obs.saturationConfidence.toDouble()
                val weight = support * robust / max(noiseVariance, 1e-8)

                weighted += value * weight
                weightSum += weight
            }

            out[i] = if (weightSum > 1e-12) (weighted / weightSum).toFloat().coerceIn(0f, 1f) else referenceValue
        }

        return RawPlane(
            width = reference.width,
            height = reference.height,
            values = out,
            pattern = reference.pattern,
            blackLevel = 0f,
            whiteLevel = 1f,
            blackLevelPattern = CfaBlackLevel.uniform(0f)
        )
    }

    /** Edge-aware deterministic Bayer demosaic portable baseline. */
    fun demosaic(raw: RawPlane): LinearRgbImage {
        val normalizedAlready = raw.whiteLevel == 1f && raw.blackLevelPattern.offsets.all { it == 0f }
        val src = if (normalizedAlready) raw.values else normalize(raw)
        val out = FloatArray(raw.width * raw.height * 3)

        fun sample(x: Int, y: Int): Float {
            val xx = x.coerceIn(0, raw.width - 1)
            val yy = y.coerceIn(0, raw.height - 1)
            return src[yy * raw.width + xx]
        }

        for (y in 0 until raw.height) {
            for (x in 0 until raw.width) {
                val color = cfaColor(raw.pattern, x, y)
                val c = sample(x, y)
                val left = sample(x - 1, y)
                val right = sample(x + 1, y)
                val up = sample(x, y - 1)
                val down = sample(x, y + 1)
                val diag = (sample(x - 1, y - 1) + sample(x + 1, y - 1) + sample(x - 1, y + 1) + sample(x + 1, y + 1)) * 0.25f
                val horizontal = (left + right) * 0.5f
                val vertical = (up + down) * 0.5f
                val gradientH = abs(left - right)
                val gradientV = abs(up - down)
                val greenInterp = if (gradientH < gradientV) horizontal else vertical

                val (r, g, b) = when (color) {
                    0 -> Triple(c, greenInterp, diag)
                    2 -> Triple(diag, greenInterp, c)
                    else -> {
                        val redHorizontal = redIsHorizontal(raw.pattern, x, y)
                        if (redHorizontal) Triple(horizontal, c, vertical) else Triple(vertical, c, horizontal)
                    }
                }

                val base = (y * raw.width + x) * 3
                out[base] = r.coerceAtLeast(0f)
                out[base + 1] = g.coerceAtLeast(0f)
                out[base + 2] = b.coerceAtLeast(0f)
            }
        }
        return LinearRgbImage(raw.width, raw.height, out)
    }

    private fun cfaColor(pattern: BayerPattern, x: Int, y: Int): Int {
        val xe = x and 1
        val ye = y and 1
        return when (pattern) {
            BayerPattern.RGGB -> when { ye == 0 && xe == 0 -> 0; ye == 1 && xe == 1 -> 2; else -> 1 }
            BayerPattern.BGGR -> when { ye == 0 && xe == 0 -> 2; ye == 1 && xe == 1 -> 0; else -> 1 }
            BayerPattern.GRBG -> when { ye == 0 && xe == 1 -> 0; ye == 1 && xe == 0 -> 2; else -> 1 }
            BayerPattern.GBRG -> when { ye == 0 && xe == 1 -> 2; ye == 1 && xe == 0 -> 0; else -> 1 }
        }
    }

    private fun redIsHorizontal(pattern: BayerPattern, x: Int, y: Int): Boolean = when (pattern) {
        BayerPattern.RGGB -> (y and 1) == 0
        BayerPattern.BGGR -> (y and 1) == 1
        BayerPattern.GRBG -> (y and 1) == 0
        BayerPattern.GBRG -> (y and 1) == 1
    }
}
