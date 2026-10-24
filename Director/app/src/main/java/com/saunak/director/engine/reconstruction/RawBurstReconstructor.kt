package com.saunak.director.engine.reconstruction

import android.graphics.BitmapFactory
import com.saunak.director.camera.RawJpegBurstCapture
import com.saunak.director.engine.device.CameraModuleTwin
import com.saunak.director.engine.device.CfaPattern
import java.io.File
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/** Global phase-preserving translation from one burst frame into the reference RAW lattice. */
data class RawAlignment(
    val frameIndex: Int,
    val rawDx: Int,
    val rawDy: Int,
    val confidence: Double,
    val thumbnailSad: Double
) {
    init {
        require(frameIndex >= 0)
        require(rawDx % 2 == 0 && rawDy % 2 == 0) { "Bayer translation must preserve 2x2 CFA phase" }
        require(confidence in 0.0..1.0)
        require(thumbnailSad >= 0.0)
    }
}

data class LinearRgbTile(
    val x: Int,
    val y: Int,
    val width: Int,
    val height: Int,
    /** Interleaved linear RGB in reference-exposure radiance, three floats per pixel. */
    val rgb: FloatArray,
    /** Per-output-pixel confidence derived from valid physical observations. */
    val confidence: FloatArray
) {
    init {
        require(x >= 0 && y >= 0 && width > 0 && height > 0)
        require(rgb.size == width * height * 3)
        require(confidence.size == width * height)
    }
}

interface LinearRgbTileSink {
    fun begin(width: Int, height: Int)
    fun accept(tile: LinearRgbTile)
    fun end()
}

data class RawReconstructionSummary(
    val width: Int,
    val height: Int,
    val referenceFrameIndex: Int,
    val alignments: List<RawAlignment>,
    val binned2x2: Boolean,
    val meanConfidence: Double,
    val rejectedFrameCount: Int
)

/**
 * Deterministic evidence-preserving burst reconstructor.
 *
 * This path intentionally avoids generative detail. Frames are aligned using their simultaneous
 * safety JPEGs, translated back to the RAW lattice with even-pixel shifts, normalized into the
 * reference exposure, robustly fused per CFA sample, then demosaiced/2x2-binned into linear RGB.
 * The implementation is tile streamed so sensor resolution does not determine Java heap usage.
 */
class RawBurstReconstructor(
    private val tileSize: Int = 384,
    private val maxFullResolutionPixels: Long = 24_000_000L
) {
    init {
        require(tileSize in 128..1024)
        require(maxFullResolutionPixels >= 4_000_000)
    }

    fun reconstruct(
        capture: RawJpegBurstCapture,
        camera: CameraModuleTwin,
        sink: LinearRgbTileSink
    ): RawReconstructionSummary {
        require(camera.cfaPattern in SUPPORTED_CFA) { "Unsupported CFA ${camera.cfaPattern} for deterministic v4 fusion" }
        require(capture.frames.isNotEmpty())

        val containers = capture.frames.map { SierraRawContainer.open(it.rawFile) }
        try {
            val width = containers.first().header.width
            val height = containers.first().header.height
            require(containers.all { it.header.width == width && it.header.height == height }) { "RAW burst dimensions changed" }
            require(camera.whiteLevel != null && camera.whiteLevel > 0) { "RAW white level unavailable" }

            val referenceIndex = capture.frames.indices.minByOrNull { abs(capture.frames[it].requested.evOffset) } ?: 0
            val alignments = estimateAlignments(capture, width, height, referenceIndex)
            val usable = alignments.filter { it.frameIndex == referenceIndex || it.confidence >= MIN_ALIGNMENT_CONFIDENCE }
            require(usable.isNotEmpty())

            val binned = width.toLong() * height.toLong() > maxFullResolutionPixels
            val outputWidth = if (binned) width / 2 else width
            val outputHeight = if (binned) height / 2 else height
            sink.begin(outputWidth, outputHeight)

            var confidenceSum = 0.0
            var confidenceCount = 0L
            var outY = 0
            while (outY < outputHeight) {
                val tileH = min(tileSize, outputHeight - outY)
                var outX = 0
                while (outX < outputWidth) {
                    val tileW = min(tileSize, outputWidth - outX)
                    val tile = if (binned) {
                        reconstructBinnedTile(
                            outX, outY, tileW, tileH, containers, capture, camera, referenceIndex, usable
                        )
                    } else {
                        reconstructDemosaicTile(
                            outX, outY, tileW, tileH, containers, capture, camera, referenceIndex, usable
                        )
                    }
                    for (c in tile.confidence) {
                        confidenceSum += c
                        confidenceCount++
                    }
                    sink.accept(tile)
                    outX += tileW
                }
                outY += tileH
            }
            sink.end()
            return RawReconstructionSummary(
                width = outputWidth,
                height = outputHeight,
                referenceFrameIndex = referenceIndex,
                alignments = alignments,
                binned2x2 = binned,
                meanConfidence = if (confidenceCount == 0L) 0.0 else (confidenceSum / confidenceCount).coerceIn(0.0, 1.0),
                rejectedFrameCount = alignments.count { it.frameIndex != referenceIndex && it.confidence < MIN_ALIGNMENT_CONFIDENCE }
            )
        } finally {
            containers.forEach { runCatching { it.close() } }
        }
    }

    private fun reconstructBinnedTile(
        outX: Int,
        outY: Int,
        outW: Int,
        outH: Int,
        containers: List<SierraRawContainer>,
        capture: RawJpegBurstCapture,
        camera: CameraModuleTwin,
        referenceIndex: Int,
        usable: List<RawAlignment>
    ): LinearRgbTile {
        val rgb = FloatArray(outW * outH * 3)
        val confidence = FloatArray(outW * outH)
        val white = camera.whiteLevel!!.toDouble()
        val refEnergy = exposureEnergy(capture, referenceIndex)
        var p = 0
        for (oy in 0 until outH) {
            val rawY = (outY + oy) * 2
            for (ox in 0 until outW) {
                val rawX = (outX + ox) * 2
                var r = 0.0
                var g = 0.0
                var b = 0.0
                var rW = 0.0
                var gW = 0.0
                var bW = 0.0
                var evidence = 0.0
                var evidenceMax = 0.0

                for (phaseY in 0..1) for (phaseX in 0..1) {
                    val x = rawX + phaseX
                    val y = rawY + phaseY
                    val fused = fuseMosaicSample(
                        x, y, containers, capture, camera, refEnergy, white, usable
                    )
                    val color = cfaColor(camera.cfaPattern, x, y)
                    when (color) {
                        Color.R -> { r += fused.value * fused.weight; rW += fused.weight }
                        Color.G -> { g += fused.value * fused.weight; gW += fused.weight }
                        Color.B -> { b += fused.value * fused.weight; bW += fused.weight }
                    }
                    evidence += fused.confidence
                    evidenceMax += 1.0
                }

                val base = p * 3
                rgb[base] = if (rW > 0) (r / rW).toFloat() else 0f
                rgb[base + 1] = if (gW > 0) (g / gW).toFloat() else 0f
                rgb[base + 2] = if (bW > 0) (b / bW).toFloat() else 0f
                confidence[p] = if (evidenceMax > 0) (evidence / evidenceMax).toFloat().coerceIn(0f, 1f) else 0f
                p++
            }
        }
        return LinearRgbTile(outX, outY, outW, outH, rgb, confidence)
    }

    private fun reconstructDemosaicTile(
        outX: Int,
        outY: Int,
        outW: Int,
        outH: Int,
        containers: List<SierraRawContainer>,
        capture: RawJpegBurstCapture,
        camera: CameraModuleTwin,
        referenceIndex: Int,
        usable: List<RawAlignment>
    ): LinearRgbTile {
        val halo = 2
        val x0 = max(0, outX - halo)
        val y0 = max(0, outY - halo)
        val x1 = min(containers.first().header.width, outX + outW + halo)
        val y1 = min(containers.first().header.height, outY + outH + halo)
        val mosaicW = x1 - x0
        val mosaicH = y1 - y0
        val mosaic = FloatArray(mosaicW * mosaicH)
        val mosaicConfidence = FloatArray(mosaic.size)
        val white = camera.whiteLevel!!.toDouble()
        val refEnergy = exposureEnergy(capture, referenceIndex)

        var i = 0
        for (y in y0 until y1) for (x in x0 until x1) {
            val fused = fuseMosaicSample(x, y, containers, capture, camera, refEnergy, white, usable)
            mosaic[i] = fused.value.toFloat()
            mosaicConfidence[i] = fused.confidence.toFloat()
            i++
        }

        fun m(x: Int, y: Int): Float {
            val cx = x.coerceIn(x0, x1 - 1)
            val cy = y.coerceIn(y0, y1 - 1)
            return mosaic[(cy - y0) * mosaicW + (cx - x0)]
        }
        fun c(x: Int, y: Int): Float {
            val cx = x.coerceIn(x0, x1 - 1)
            val cy = y.coerceIn(y0, y1 - 1)
            return mosaicConfidence[(cy - y0) * mosaicW + (cx - x0)]
        }

        val rgb = FloatArray(outW * outH * 3)
        val confidence = FloatArray(outW * outH)
        var p = 0
        for (oy in 0 until outH) {
            val y = outY + oy
            for (ox in 0 until outW) {
                val x = outX + ox
                val own = m(x, y)
                val color = cfaColor(camera.cfaPattern, x, y)
                val left = m(x - 1, y)
                val right = m(x + 1, y)
                val up = m(x, y - 1)
                val down = m(x, y + 1)
                val ul = m(x - 1, y - 1)
                val ur = m(x + 1, y - 1)
                val dl = m(x - 1, y + 1)
                val dr = m(x + 1, y + 1)

                val horizontalGrad = abs(left - right)
                val verticalGrad = abs(up - down)
                val greenCross = if (horizontalGrad < verticalGrad) (left + right) * 0.5f else (up + down) * 0.5f
                val diagonal = (ul + ur + dl + dr) * 0.25f

                val (r, g, b) = when (color) {
                    Color.R -> Triple(own, greenCross, diagonal)
                    Color.B -> Triple(diagonal, greenCross, own)
                    Color.G -> {
                        val redHorizontal = cfaColor(camera.cfaPattern, x - 1, y) == Color.R || cfaColor(camera.cfaPattern, x + 1, y) == Color.R
                        if (redHorizontal) Triple((left + right) * 0.5f, own, (up + down) * 0.5f)
                        else Triple((up + down) * 0.5f, own, (left + right) * 0.5f)
                    }
                }
                val base = p * 3
                rgb[base] = r
                rgb[base + 1] = g
                rgb[base + 2] = b
                confidence[p] = ((c(x, y) + c(x - 1, y) + c(x + 1, y) + c(x, y - 1) + c(x, y + 1)) / 5f).coerceIn(0f, 1f)
                p++
            }
        }
        return LinearRgbTile(outX, outY, outW, outH, rgb, confidence)
    }

    private data class FusedSample(val value: Double, val weight: Double, val confidence: Double)

    private fun fuseMosaicSample(
        x: Int,
        y: Int,
        containers: List<SierraRawContainer>,
        capture: RawJpegBurstCapture,
        camera: CameraModuleTwin,
        referenceEnergy: Double,
        whiteLevel: Double,
        usable: List<RawAlignment>
    ): FusedSample {
        val observations = ArrayList<Pair<Double, Double>>(usable.size)
        val black = blackLevel(camera, x, y).toDouble()
        val denominator = max(1.0, whiteLevel - black)
        var totalPotential = 0.0

        for (alignment in usable) {
            val sourceX = x + alignment.rawDx
            val sourceY = y + alignment.rawDy
            val container = containers[alignment.frameIndex]
            if (sourceX !in 0 until container.header.width || sourceY !in 0 until container.header.height) continue
            val raw = container.sampleUnsigned16(sourceX, sourceY).toDouble()
            val sensorSignal = ((raw - black) / denominator).coerceIn(0.0, 1.0)
            val frameEnergy = exposureEnergy(capture, alignment.frameIndex)
            val value = sensorSignal * (referenceEnergy / frameEnergy)
            val saturationWeight = when {
                sensorSignal >= 0.995 -> 0.0
                sensorSignal >= 0.92 -> ((0.995 - sensorSignal) / 0.075).coerceIn(0.0, 1.0)
                else -> 1.0
            }
            val shadowWeight = (0.18 + 0.82 * sqrt(sensorSignal.coerceAtLeast(0.0))).coerceIn(0.18, 1.0)
            val alignmentWeight = if (alignment.frameIndex == usable.first().frameIndex) max(0.75, alignment.confidence) else alignment.confidence
            val weight = saturationWeight * shadowWeight * alignmentWeight
            totalPotential += alignmentWeight
            if (weight > 1e-6 && value.isFinite()) observations += value to weight
        }

        if (observations.isEmpty()) return FusedSample(0.0, 0.0, 0.0)
        val sorted = observations.map { it.first }.sorted()
        val median = sorted[sorted.size / 2]
        val tolerance = 0.035 + 0.12 * median.coerceIn(0.0, 1.5)
        var sum = 0.0
        var weightSum = 0.0
        for ((value, weight) in observations) {
            val robust = if (abs(value - median) <= tolerance) 1.0 else 0.08
            val w = weight * robust
            sum += value * w
            weightSum += w
        }
        val fused = if (weightSum > 0) sum / weightSum else median
        val confidence = if (totalPotential <= 0) 0.0 else (weightSum / totalPotential).coerceIn(0.0, 1.0)
        return FusedSample(fused.coerceIn(0.0, 4.0), weightSum, confidence)
    }

    private fun estimateAlignments(
        capture: RawJpegBurstCapture,
        rawWidth: Int,
        rawHeight: Int,
        referenceIndex: Int
    ): List<RawAlignment> {
        val thumbnails = capture.frames.map { decodeLumaThumbnail(it.jpegFile, ALIGNMENT_LONG_SIDE) }
        val ref = thumbnails[referenceIndex]
        return thumbnails.mapIndexed { index, candidate ->
            if (index == referenceIndex) {
                RawAlignment(index, 0, 0, 1.0, 0.0)
            } else {
                val shift = estimateThumbnailShift(ref, candidate)
                val scaleX = rawWidth.toDouble() / ref.width
                val scaleY = rawHeight.toDouble() / ref.height
                val rawDx = evenRound(shift.dx * scaleX)
                val rawDy = evenRound(shift.dy * scaleY)
                RawAlignment(index, rawDx, rawDy, shift.confidence, shift.sad)
            }
        }
    }

    private data class LumaThumb(val width: Int, val height: Int, val luma: ByteArray)
    private data class ThumbShift(val dx: Int, val dy: Int, val confidence: Double, val sad: Double)

    private fun decodeLumaThumbnail(file: File, maxLongSide: Int): LumaThumb {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        require(bounds.outWidth > 0 && bounds.outHeight > 0)
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / sample > maxLongSide * 2) sample *= 2
        val bitmap = BitmapFactory.decodeFile(file.absolutePath, BitmapFactory.Options().apply { inSampleSize = sample })
            ?: error("Could not decode burst alignment JPEG")
        try {
            val longest = max(bitmap.width, bitmap.height)
            val scaled = if (longest <= maxLongSide) bitmap else {
                val scale = maxLongSide.toFloat() / longest
                android.graphics.Bitmap.createScaledBitmap(
                    bitmap,
                    max(1, (bitmap.width * scale).roundToInt()),
                    max(1, (bitmap.height * scale).roundToInt()),
                    true
                )
            }
            try {
                val pixels = IntArray(scaled.width * scaled.height)
                scaled.getPixels(pixels, 0, scaled.width, 0, 0, scaled.width, scaled.height)
                val luma = ByteArray(pixels.size)
                pixels.forEachIndexed { i, pixel ->
                    val r = (pixel ushr 16) and 0xff
                    val g = (pixel ushr 8) and 0xff
                    val b = pixel and 0xff
                    luma[i] = ((77 * r + 150 * g + 29 * b) ushr 8).toByte()
                }
                return LumaThumb(scaled.width, scaled.height, luma)
            } finally {
                if (scaled !== bitmap) scaled.recycle()
            }
        } finally {
            if (!bitmap.isRecycled) bitmap.recycle()
        }
    }

    private fun estimateThumbnailShift(reference: LumaThumb, candidate: LumaThumb): ThumbShift {
        require(reference.width == candidate.width && reference.height == candidate.height) { "Burst JPEG geometry changed" }
        var bestSad = Double.POSITIVE_INFINITY
        var secondSad = Double.POSITIVE_INFINITY
        var bestDx = 0
        var bestDy = 0
        val margin = ALIGNMENT_SEARCH + 8
        val step = 3
        for (dy in -ALIGNMENT_SEARCH..ALIGNMENT_SEARCH) {
            for (dx in -ALIGNMENT_SEARCH..ALIGNMENT_SEARCH) {
                var sad = 0L
                var count = 0
                var y = margin
                while (y < reference.height - margin) {
                    var x = margin
                    while (x < reference.width - margin) {
                        val cX = x + dx
                        val cY = y + dy
                        val a = reference.luma[y * reference.width + x].toInt() and 0xff
                        val b = candidate.luma[cY * candidate.width + cX].toInt() and 0xff
                        sad += abs(a - b)
                        count++
                        x += step
                    }
                    y += step
                }
                if (count == 0) continue
                val normalized = sad.toDouble() / count
                if (normalized < bestSad) {
                    secondSad = bestSad
                    bestSad = normalized
                    bestDx = dx
                    bestDy = dy
                } else if (normalized < secondSad) {
                    secondSad = normalized
                }
            }
        }
        val absolute = (1.0 - bestSad / 70.0).coerceIn(0.0, 1.0)
        val separation = if (secondSad.isFinite() && secondSad > 0) ((secondSad - bestSad) / secondSad * 5.0).coerceIn(0.0, 1.0) else 0.0
        return ThumbShift(bestDx, bestDy, (0.8 * absolute + 0.2 * separation).coerceIn(0.0, 1.0), bestSad)
    }

    private fun exposureEnergy(capture: RawJpegBurstCapture, index: Int): Double {
        val spec = capture.frames[index].requested
        return max(1.0, spec.exposureTimeNs.toDouble() * spec.sensitivityIso.toDouble())
    }

    private fun blackLevel(camera: CameraModuleTwin, x: Int, y: Int): Int {
        if (camera.blackLevelPattern.size != 4) return 0
        return camera.blackLevelPattern[(y and 1) * 2 + (x and 1)]
    }

    private enum class Color { R, G, B }

    private fun cfaColor(pattern: CfaPattern, x: Int, y: Int): Color {
        val px = x and 1
        val py = y and 1
        return when (pattern) {
            CfaPattern.RGGB -> when (py * 2 + px) { 0 -> Color.R; 3 -> Color.B; else -> Color.G }
            CfaPattern.BGGR -> when (py * 2 + px) { 0 -> Color.B; 3 -> Color.R; else -> Color.G }
            CfaPattern.GRBG -> when (py * 2 + px) { 1 -> Color.R; 2 -> Color.B; else -> Color.G }
            CfaPattern.GBRG -> when (py * 2 + px) { 1 -> Color.B; 2 -> Color.R; else -> Color.G }
            else -> error("Unsupported CFA $pattern")
        }
    }

    private fun evenRound(value: Double): Int = (value / 2.0).roundToInt() * 2

    companion object {
        private val SUPPORTED_CFA = setOf(CfaPattern.RGGB, CfaPattern.BGGR, CfaPattern.GRBG, CfaPattern.GBRG)
        private const val ALIGNMENT_LONG_SIDE = 256
        private const val ALIGNMENT_SEARCH = 14
        private const val MIN_ALIGNMENT_CONFIDENCE = 0.32
    }
}
