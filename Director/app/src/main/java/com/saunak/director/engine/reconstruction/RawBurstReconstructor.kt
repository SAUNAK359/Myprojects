package com.saunak.director.engine.reconstruction

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.saunak.director.camera.RawJpegBurstCapture
import com.saunak.director.engine.device.CameraModuleTwin
import com.saunak.director.engine.device.CfaPattern
import java.io.File
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/** Global summary displacement from one burst frame into the reference RAW lattice. */
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
    val rgb: FloatArray,
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
    val binFactor: Int,
    val meanConfidence: Double,
    val rejectedFrameCount: Int
) {
    init { require(binFactor >= 1) }
    val binned2x2: Boolean get() = binFactor == 2
}

private data class LumaThumb(
    val width: Int,
    val height: Int,
    val edge: ByteArray,
    val sharpness: Double
)

private data class MotionVector(
    val dxRaw: Int,
    val dyRaw: Int,
    val confidence: Double
) {
    init {
        require(dxRaw % 2 == 0 && dyRaw % 2 == 0)
        require(confidence in 0.0..1.0)
    }
}

private class MotionField(
    val frameIndex: Int,
    val rawWidth: Int,
    val rawHeight: Int,
    val columns: Int,
    val rows: Int,
    private val vectors: Array<MotionVector>,
    val global: RawAlignment
) {
    init {
        require(vectors.size == columns * rows)
        require(columns > 0 && rows > 0)
    }

    val averageConfidence: Double = vectors.map { it.confidence }.average().coerceIn(0.0, 1.0)
    val minDx: Int = vectors.minOf { it.dxRaw }
    val maxDx: Int = vectors.maxOf { it.dxRaw }
    val minDy: Int = vectors.minOf { it.dyRaw }
    val maxDy: Int = vectors.maxOf { it.dyRaw }

    fun at(x: Int, y: Int): MotionVector {
        val col = ((x.toLong() * columns) / max(1, rawWidth)).toInt().coerceIn(0, columns - 1)
        val row = ((y.toLong() * rows) / max(1, rawHeight)).toInt().coerceIn(0, rows - 1)
        return vectors[row * columns + col]
    }
}

private data class PreparedFrameTile(
    val frameIndex: Int,
    val field: MotionField,
    val sourceX0: Int,
    val sourceY0: Int,
    val width: Int,
    val height: Int,
    val samples: IntArray,
    val metadataWeight: Double
) {
    fun sample(referenceX: Int, referenceY: Int): Pair<Int, Double>? {
        val motion = field.at(referenceX, referenceY)
        if (motion.confidence <= 0.0) return null
        val x = referenceX + motion.dxRaw
        val y = referenceY + motion.dyRaw
        if (x !in sourceX0 until sourceX0 + width || y !in sourceY0 until sourceY0 + height) return null
        val value = samples[(y - sourceY0) * width + (x - sourceX0)]
        return value to (motion.confidence * metadataWeight).coerceIn(0.0, 1.0)
    }
}

/**
 * Evidence-preserving RAW burst reconstructor inspired by production mobile burst pipelines.
 *
 * Alignment uses simultaneous JPEGs only to estimate motion; all fused pixel evidence comes from
 * RAW. Exposure-invariant edge thumbnails provide global + local tile translations. Translations
 * are quantized to even RAW pixels so Bayer phase never changes. Each reconstruction tile preloads
 * one aligned RAW region per usable frame, replacing millions of random file seeks with sequential
 * row reads. Fusion uses exact observed exposure/ISO when available and robustly rejects motion
 * disagreement instead of hallucinating replacement content.
 */
class RawBurstReconstructor(
    private val rawTileBudget: Int = 512,
    private val maxOutputPixels: Long = 30_000_000L
) {
    init {
        require(rawTileBudget in 256..1024)
        require(maxOutputPixels >= 8_000_000L)
    }

    fun reconstruct(
        capture: RawJpegBurstCapture,
        camera: CameraModuleTwin,
        sink: LinearRgbTileSink
    ): RawReconstructionSummary {
        require(camera.cfaPattern in SUPPORTED_CFA) { "Unsupported CFA ${camera.cfaPattern}" }
        require(capture.frames.isNotEmpty())
        require(camera.whiteLevel != null && camera.whiteLevel > 0) { "RAW white level unavailable" }

        val containers = capture.frames.map { SierraRawContainer.open(it.rawFile) }
        try {
            val width = containers.first().header.width
            val height = containers.first().header.height
            require(containers.all { it.header.width == width && it.header.height == height }) { "RAW burst dimensions changed" }

            val thumbnails = capture.frames.map { decodeEdgeThumbnail(it.jpegFile, ALIGNMENT_LONG_SIDE) }
            require(thumbnails.all { it.width == thumbnails.first().width && it.height == thumbnails.first().height }) {
                "Burst JPEG geometry changed"
            }
            val referenceIndex = selectReference(capture, thumbnails)
            val fields = estimateMotionFields(thumbnails, width, height, referenceIndex)
            val usableFields = fields.filter {
                it.frameIndex == referenceIndex || it.averageConfidence >= MIN_FRAME_ALIGNMENT_CONFIDENCE
            }
            require(usableFields.any { it.frameIndex == referenceIndex })

            val binFactor = chooseBinFactor(width, height)
            val outputWidth = width / binFactor
            val outputHeight = height / binFactor
            require(outputWidth > 0 && outputHeight > 0)
            sink.begin(outputWidth, outputHeight)

            val outputTileSize = max(64, rawTileBudget / binFactor)
            var confidenceSum = 0.0
            var confidenceCount = 0L
            var outY = 0
            while (outY < outputHeight) {
                val tileH = min(outputTileSize, outputHeight - outY)
                var outX = 0
                while (outX < outputWidth) {
                    val tileW = min(outputTileSize, outputWidth - outX)
                    val tile = if (binFactor == 1) {
                        reconstructDemosaicTile(
                            outX, outY, tileW, tileH,
                            containers, capture, camera, referenceIndex, usableFields
                        )
                    } else {
                        reconstructBinnedTile(
                            outX, outY, tileW, tileH, binFactor,
                            containers, capture, camera, referenceIndex, usableFields
                        )
                    }
                    for (confidence in tile.confidence) {
                        confidenceSum += confidence
                        confidenceCount++
                    }
                    sink.accept(tile)
                    outX += tileW
                }
                outY += tileH
            }
            sink.end()

            val summaries = fields.map { it.global.copy(confidence = it.averageConfidence) }
            return RawReconstructionSummary(
                width = outputWidth,
                height = outputHeight,
                referenceFrameIndex = referenceIndex,
                alignments = summaries,
                binFactor = binFactor,
                meanConfidence = if (confidenceCount == 0L) 0.0 else (confidenceSum / confidenceCount).coerceIn(0.0, 1.0),
                rejectedFrameCount = fields.count {
                    it.frameIndex != referenceIndex && it.averageConfidence < MIN_FRAME_ALIGNMENT_CONFIDENCE
                }
            )
        } finally {
            containers.forEach { runCatching { it.close() } }
        }
    }

    internal fun chooseBinFactor(width: Int, height: Int): Int {
        require(width > 0 && height > 0)
        if (width.toLong() * height <= maxOutputPixels) return 1
        var factor = 2
        while ((width / factor).toLong() * (height / factor) > maxOutputPixels) factor += 2
        return factor
    }

    private fun reconstructBinnedTile(
        outX: Int,
        outY: Int,
        outW: Int,
        outH: Int,
        factor: Int,
        containers: List<SierraRawContainer>,
        capture: RawJpegBurstCapture,
        camera: CameraModuleTwin,
        referenceIndex: Int,
        fields: List<MotionField>
    ): LinearRgbTile {
        require(factor >= 2 && factor % 2 == 0)
        val rawX0 = outX * factor
        val rawY0 = outY * factor
        val rawX1 = rawX0 + outW * factor
        val rawY1 = rawY0 + outH * factor
        val prepared = prepareFrameTiles(rawX0, rawY0, rawX1, rawY1, containers, capture, fields)
        val rgb = FloatArray(outW * outH * 3)
        val confidence = FloatArray(outW * outH)
        val white = camera.whiteLevel!!.toDouble()
        val referenceEnergy = exposureEnergy(capture, referenceIndex)

        var pixel = 0
        for (oy in 0 until outH) {
            val baseY = rawY0 + oy * factor
            for (ox in 0 until outW) {
                val baseX = rawX0 + ox * factor
                var r = 0.0; var g = 0.0; var b = 0.0
                var rw = 0.0; var gw = 0.0; var bw = 0.0
                var evidence = 0.0; var evidenceCount = 0
                for (py in 0 until factor) for (px in 0 until factor) {
                    val x = baseX + px
                    val y = baseY + py
                    val fused = fuseMosaicSample(
                        x, y, prepared, capture, camera, referenceIndex, referenceEnergy, white
                    )
                    when (cfaColor(camera.cfaPattern, x, y)) {
                        Color.R -> { r += fused.value * fused.weight; rw += fused.weight }
                        Color.G -> { g += fused.value * fused.weight; gw += fused.weight }
                        Color.B -> { b += fused.value * fused.weight; bw += fused.weight }
                    }
                    evidence += fused.confidence
                    evidenceCount++
                }
                val base = pixel * 3
                rgb[base] = if (rw > 0) (r / rw).toFloat() else 0f
                rgb[base + 1] = if (gw > 0) (g / gw).toFloat() else 0f
                rgb[base + 2] = if (bw > 0) (b / bw).toFloat() else 0f
                confidence[pixel] = if (evidenceCount > 0) {
                    (evidence / evidenceCount).toFloat().coerceIn(0f, 1f)
                } else 0f
                pixel++
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
        fields: List<MotionField>
    ): LinearRgbTile {
        val halo = 2
        val rawX0 = max(0, outX - halo)
        val rawY0 = max(0, outY - halo)
        val rawX1 = min(containers.first().header.width, outX + outW + halo)
        val rawY1 = min(containers.first().header.height, outY + outH + halo)
        val prepared = prepareFrameTiles(rawX0, rawY0, rawX1, rawY1, containers, capture, fields)
        val mosaicW = rawX1 - rawX0
        val mosaicH = rawY1 - rawY0
        val mosaic = FloatArray(mosaicW * mosaicH)
        val mosaicConfidence = FloatArray(mosaic.size)
        val white = camera.whiteLevel!!.toDouble()
        val referenceEnergy = exposureEnergy(capture, referenceIndex)

        var index = 0
        for (y in rawY0 until rawY1) for (x in rawX0 until rawX1) {
            val fused = fuseMosaicSample(
                x, y, prepared, capture, camera, referenceIndex, referenceEnergy, white
            )
            mosaic[index] = fused.value.toFloat()
            mosaicConfidence[index] = fused.confidence.toFloat()
            index++
        }

        fun m(x: Int, y: Int): Float {
            val cx = x.coerceIn(rawX0, rawX1 - 1)
            val cy = y.coerceIn(rawY0, rawY1 - 1)
            return mosaic[(cy - rawY0) * mosaicW + (cx - rawX0)]
        }
        fun c(x: Int, y: Int): Float {
            val cx = x.coerceIn(rawX0, rawX1 - 1)
            val cy = y.coerceIn(rawY0, rawY1 - 1)
            return mosaicConfidence[(cy - rawY0) * mosaicW + (cx - rawX0)]
        }

        val rgb = FloatArray(outW * outH * 3)
        val confidence = FloatArray(outW * outH)
        var pixel = 0
        for (oy in 0 until outH) {
            val y = outY + oy
            for (ox in 0 until outW) {
                val x = outX + ox
                val own = m(x, y)
                val left = m(x - 1, y); val right = m(x + 1, y)
                val up = m(x, y - 1); val down = m(x, y + 1)
                val ul = m(x - 1, y - 1); val ur = m(x + 1, y - 1)
                val dl = m(x - 1, y + 1); val dr = m(x + 1, y + 1)
                val horizontalGradient = abs(left - right)
                val verticalGradient = abs(up - down)
                val greenCross = if (horizontalGradient < verticalGradient) {
                    (left + right) * 0.5f
                } else {
                    (up + down) * 0.5f
                }
                val diagonal = (ul + ur + dl + dr) * 0.25f

                val (r, g, b) = when (cfaColor(camera.cfaPattern, x, y)) {
                    Color.R -> Triple(own, greenCross, diagonal)
                    Color.B -> Triple(diagonal, greenCross, own)
                    Color.G -> {
                        val redHorizontal = cfaColor(camera.cfaPattern, x - 1, y) == Color.R ||
                            cfaColor(camera.cfaPattern, x + 1, y) == Color.R
                        if (redHorizontal) Triple((left + right) * 0.5f, own, (up + down) * 0.5f)
                        else Triple((up + down) * 0.5f, own, (left + right) * 0.5f)
                    }
                }
                val base = pixel * 3
                rgb[base] = r
                rgb[base + 1] = g
                rgb[base + 2] = b
                confidence[pixel] = (
                    (c(x, y) + c(x - 1, y) + c(x + 1, y) + c(x, y - 1) + c(x, y + 1)) / 5f
                    ).coerceIn(0f, 1f)
                pixel++
            }
        }
        return LinearRgbTile(outX, outY, outW, outH, rgb, confidence)
    }

    private fun prepareFrameTiles(
        referenceX0: Int,
        referenceY0: Int,
        referenceX1: Int,
        referenceY1: Int,
        containers: List<SierraRawContainer>,
        capture: RawJpegBurstCapture,
        fields: List<MotionField>
    ): List<PreparedFrameTile> {
        return fields.mapNotNull { field ->
            val container = containers[field.frameIndex]
            val sourceX0 = max(0, referenceX0 + field.minDx)
            val sourceY0 = max(0, referenceY0 + field.minDy)
            val sourceX1 = min(container.header.width, referenceX1 + field.maxDx)
            val sourceY1 = min(container.header.height, referenceY1 + field.maxDy)
            if (sourceX1 <= sourceX0 || sourceY1 <= sourceY0) return@mapNotNull null
            val width = sourceX1 - sourceX0
            val height = sourceY1 - sourceY0
            val frame = capture.frames[field.frameIndex]
            PreparedFrameTile(
                frameIndex = field.frameIndex,
                field = field,
                sourceX0 = sourceX0,
                sourceY0 = sourceY0,
                width = width,
                height = height,
                samples = container.readTile(sourceX0, sourceY0, width, height),
                metadataWeight = when {
                    frame.requestMatched -> 1.0
                    frame.observed != null -> 0.78
                    else -> 0.52
                }
            )
        }
    }

    private data class FusedSample(val value: Double, val weight: Double, val confidence: Double)

    private fun fuseMosaicSample(
        x: Int,
        y: Int,
        prepared: List<PreparedFrameTile>,
        capture: RawJpegBurstCapture,
        camera: CameraModuleTwin,
        referenceIndex: Int,
        referenceEnergy: Double,
        whiteLevel: Double
    ): FusedSample {
        val observations = ArrayList<Pair<Double, Double>>(prepared.size)
        val black = blackLevel(camera, x, y).toDouble()
        val denominator = max(1.0, whiteLevel - black)
        var potentialWeight = 0.0

        for (tile in prepared) {
            val sampled = tile.sample(x, y) ?: continue
            val raw = sampled.first.toDouble()
            val alignmentWeight = sampled.second
            val signal = ((raw - black) / denominator).coerceIn(0.0, 1.0)
            val frameEnergy = exposureEnergy(capture, tile.frameIndex)
            val radiance = signal * (referenceEnergy / frameEnergy)
            val saturationWeight = when {
                signal >= 0.995 -> 0.0
                signal >= 0.92 -> ((0.995 - signal) / 0.075).coerceIn(0.0, 1.0)
                else -> 1.0
            }
            val shadowWeight = (0.16 + 0.84 * sqrt(signal.coerceAtLeast(0.0))).coerceIn(0.16, 1.0)
            val referenceBoost = if (tile.frameIndex == referenceIndex) 1.12 else 1.0
            val weight = alignmentWeight * saturationWeight * shadowWeight * referenceBoost
            potentialWeight += alignmentWeight * referenceBoost
            if (weight > 1e-6 && radiance.isFinite()) observations += radiance to weight
        }

        if (observations.isEmpty()) return FusedSample(0.0, 0.0, 0.0)
        val values = observations.map { it.first }.sorted()
        val median = values[values.size / 2]
        val tolerance = 0.025 + 0.10 * median.coerceIn(0.0, 2.0)
        var weightedSum = 0.0
        var weightSum = 0.0
        for ((value, baseWeight) in observations) {
            val residual = abs(value - median)
            val robust = when {
                residual <= tolerance -> 1.0
                residual <= tolerance * 2.5 -> 0.32
                else -> 0.04
            }
            val weight = baseWeight * robust
            weightedSum += value * weight
            weightSum += weight
        }
        val value = if (weightSum > 0) weightedSum / weightSum else median
        val confidence = if (potentialWeight <= 0) 0.0 else (weightSum / potentialWeight).coerceIn(0.0, 1.0)
        return FusedSample(value.coerceIn(0.0, 4.0), weightSum, confidence)
    }

    private fun selectReference(capture: RawJpegBurstCapture, thumbnails: List<LumaThumb>): Int {
        val firstCandidates = capture.frames.indices.take(min(3, capture.frames.size))
        val nearBase = firstCandidates.filter { abs(capture.frames[it].requested.evOffset) <= 0.8 }
        val candidates = if (nearBase.isNotEmpty()) nearBase else firstCandidates
        return candidates.maxByOrNull { thumbnails[it].sharpness } ?: 0
    }

    private fun estimateMotionFields(
        thumbnails: List<LumaThumb>,
        rawWidth: Int,
        rawHeight: Int,
        referenceIndex: Int
    ): List<MotionField> {
        val reference = thumbnails[referenceIndex]
        return thumbnails.mapIndexed { index, candidate ->
            if (index == referenceIndex) {
                val cols = max(1, ceil(reference.width / LOCAL_TILE_SIZE.toDouble()).toInt())
                val rows = max(1, ceil(reference.height / LOCAL_TILE_SIZE.toDouble()).toInt())
                MotionField(
                    frameIndex = index,
                    rawWidth = rawWidth,
                    rawHeight = rawHeight,
                    columns = cols,
                    rows = rows,
                    vectors = Array(cols * rows) { MotionVector(0, 0, 1.0) },
                    global = RawAlignment(index, 0, 0, 1.0, 0.0)
                )
            } else {
                estimateMotionField(reference, candidate, rawWidth, rawHeight, index)
            }
        }
    }

    private fun estimateMotionField(
        reference: LumaThumb,
        candidate: LumaThumb,
        rawWidth: Int,
        rawHeight: Int,
        frameIndex: Int
    ): MotionField {
        val global = estimateShift(reference, candidate, 0, 0, GLOBAL_SEARCH)
        val scaleX = rawWidth.toDouble() / reference.width
        val scaleY = rawHeight.toDouble() / reference.height
        val globalRawDx = evenRound(global.dx * scaleX)
        val globalRawDy = evenRound(global.dy * scaleY)
        val globalSummary = RawAlignment(frameIndex, globalRawDx, globalRawDy, global.confidence, global.sad)

        val cols = max(1, ceil(reference.width / LOCAL_TILE_SIZE.toDouble()).toInt())
        val rows = max(1, ceil(reference.height / LOCAL_TILE_SIZE.toDouble()).toInt())
        val vectors = Array(cols * rows) { MotionVector(globalRawDx, globalRawDy, global.confidence) }
        for (row in 0 until rows) {
            for (col in 0 until cols) {
                val centerX = min(reference.width - 1, col * LOCAL_TILE_SIZE + LOCAL_TILE_SIZE / 2)
                val centerY = min(reference.height - 1, row * LOCAL_TILE_SIZE + LOCAL_TILE_SIZE / 2)
                val local = estimateLocalShift(
                    reference = reference,
                    candidate = candidate,
                    centerX = centerX,
                    centerY = centerY,
                    seedDx = global.dx,
                    seedDy = global.dy
                )
                vectors[row * cols + col] = MotionVector(
                    dxRaw = evenRound(local.dx * scaleX),
                    dyRaw = evenRound(local.dy * scaleY),
                    confidence = (0.35 * global.confidence + 0.65 * local.confidence).coerceIn(0.0, 1.0)
                )
            }
        }
        return MotionField(frameIndex, rawWidth, rawHeight, cols, rows, vectors, globalSummary)
    }

    private data class ThumbShift(val dx: Int, val dy: Int, val confidence: Double, val sad: Double)

    private fun estimateLocalShift(
        reference: LumaThumb,
        candidate: LumaThumb,
        centerX: Int,
        centerY: Int,
        seedDx: Int,
        seedDy: Int
    ): ThumbShift {
        var bestSad = Double.POSITIVE_INFINITY
        var secondSad = Double.POSITIVE_INFINITY
        var bestDx = seedDx
        var bestDy = seedDy
        for (dy in seedDy - LOCAL_SEARCH..seedDy + LOCAL_SEARCH) {
            for (dx in seedDx - LOCAL_SEARCH..seedDx + LOCAL_SEARCH) {
                var total = 0L
                var count = 0
                val half = LOCAL_TILE_SIZE / 2
                var y = centerY - half
                while (y < centerY + half) {
                    var x = centerX - half
                    while (x < centerX + half) {
                        val cx = x + dx
                        val cy = y + dy
                        if (x in 1 until reference.width - 1 && y in 1 until reference.height - 1 &&
                            cx in 1 until candidate.width - 1 && cy in 1 until candidate.height - 1) {
                            val a = reference.edge[y * reference.width + x].toInt() and 0xff
                            val b = candidate.edge[cy * candidate.width + cx].toInt() and 0xff
                            total += abs(a - b)
                            count++
                        }
                        x += 2
                    }
                    y += 2
                }
                if (count < 16) continue
                val sad = total.toDouble() / count
                if (sad < bestSad) {
                    secondSad = bestSad
                    bestSad = sad
                    bestDx = dx
                    bestDy = dy
                } else if (sad < secondSad) {
                    secondSad = sad
                }
            }
        }
        return shiftWithConfidence(bestDx, bestDy, bestSad, secondSad)
    }

    private fun estimateShift(
        reference: LumaThumb,
        candidate: LumaThumb,
        seedDx: Int,
        seedDy: Int,
        search: Int
    ): ThumbShift {
        var bestSad = Double.POSITIVE_INFINITY
        var secondSad = Double.POSITIVE_INFINITY
        var bestDx = seedDx
        var bestDy = seedDy
        val margin = search + 5
        val step = 3
        for (dy in seedDy - search..seedDy + search) {
            for (dx in seedDx - search..seedDx + search) {
                var total = 0L
                var count = 0
                var y = margin
                while (y < reference.height - margin) {
                    var x = margin
                    while (x < reference.width - margin) {
                        val cx = x + dx
                        val cy = y + dy
                        if (cx in 0 until candidate.width && cy in 0 until candidate.height) {
                            val a = reference.edge[y * reference.width + x].toInt() and 0xff
                            val b = candidate.edge[cy * candidate.width + cx].toInt() and 0xff
                            total += abs(a - b)
                            count++
                        }
                        x += step
                    }
                    y += step
                }
                if (count == 0) continue
                val sad = total.toDouble() / count
                if (sad < bestSad) {
                    secondSad = bestSad
                    bestSad = sad
                    bestDx = dx
                    bestDy = dy
                } else if (sad < secondSad) {
                    secondSad = sad
                }
            }
        }
        return shiftWithConfidence(bestDx, bestDy, bestSad, secondSad)
    }

    private fun shiftWithConfidence(dx: Int, dy: Int, bestSad: Double, secondSad: Double): ThumbShift {
        if (!bestSad.isFinite()) return ThumbShift(dx, dy, 0.0, 255.0)
        val absolute = (1.0 - bestSad / 64.0).coerceIn(0.0, 1.0)
        val separation = if (secondSad.isFinite() && secondSad > 1e-6) {
            ((secondSad - bestSad) / secondSad * 6.0).coerceIn(0.0, 1.0)
        } else 0.0
        return ThumbShift(dx, dy, (0.82 * absolute + 0.18 * separation).coerceIn(0.0, 1.0), bestSad)
    }

    private fun decodeEdgeThumbnail(file: File, maxLongSide: Int): LumaThumb {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        require(bounds.outWidth > 0 && bounds.outHeight > 0)
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / sample > maxLongSide * 2) sample *= 2
        val decoded = BitmapFactory.decodeFile(file.absolutePath, BitmapFactory.Options().apply { inSampleSize = sample })
            ?: error("Could not decode burst alignment JPEG")
        try {
            val longest = max(decoded.width, decoded.height)
            val scaled = if (longest <= maxLongSide) decoded else {
                val scale = maxLongSide.toFloat() / longest
                Bitmap.createScaledBitmap(
                    decoded,
                    max(1, (decoded.width * scale).roundToInt()),
                    max(1, (decoded.height * scale).roundToInt()),
                    true
                )
            }
            try {
                val pixels = IntArray(scaled.width * scaled.height)
                scaled.getPixels(pixels, 0, scaled.width, 0, 0, scaled.width, scaled.height)
                val luma = IntArray(pixels.size)
                pixels.forEachIndexed { i, pixel ->
                    val r = (pixel ushr 16) and 0xff
                    val g = (pixel ushr 8) and 0xff
                    val b = pixel and 0xff
                    luma[i] = (77 * r + 150 * g + 29 * b) ushr 8
                }
                val edge = ByteArray(pixels.size)
                var edgeSum = 0L
                var edgeCount = 0
                for (y in 1 until scaled.height - 1) {
                    for (x in 1 until scaled.width - 1) {
                        val gx = abs(luma[y * scaled.width + x + 1] - luma[y * scaled.width + x - 1])
                        val gy = abs(luma[(y + 1) * scaled.width + x] - luma[(y - 1) * scaled.width + x])
                        val magnitude = min(255, (gx + gy) / 2)
                        edge[y * scaled.width + x] = magnitude.toByte()
                        edgeSum += magnitude
                        edgeCount++
                    }
                }
                return LumaThumb(
                    width = scaled.width,
                    height = scaled.height,
                    edge = edge,
                    sharpness = if (edgeCount == 0) 0.0 else edgeSum.toDouble() / edgeCount
                )
            } finally {
                if (scaled !== decoded) scaled.recycle()
            }
        } finally {
            if (!decoded.isRecycled) decoded.recycle()
        }
    }

    private fun exposureEnergy(capture: RawJpegBurstCapture, index: Int): Double {
        val frame = capture.frames[index]
        val observed = frame.observed
        return if (observed != null) {
            max(1.0, observed.exposureTimeNs.toDouble() * observed.sensitivityIso.toDouble())
        } else {
            max(1.0, frame.requested.exposureTimeNs.toDouble() * frame.requested.sensitivityIso.toDouble())
        }
    }

    private fun blackLevel(camera: CameraModuleTwin, x: Int, y: Int): Int {
        if (camera.blackLevelPattern.size != 4) return 0
        return camera.blackLevelPattern[(y and 1) * 2 + (x and 1)]
    }

    private enum class Color { R, G, B }

    private fun cfaColor(pattern: CfaPattern, x: Int, y: Int): Color {
        val phase = (y and 1) * 2 + (x and 1)
        return when (pattern) {
            CfaPattern.RGGB -> when (phase) { 0 -> Color.R; 3 -> Color.B; else -> Color.G }
            CfaPattern.BGGR -> when (phase) { 0 -> Color.B; 3 -> Color.R; else -> Color.G }
            CfaPattern.GRBG -> when (phase) { 1 -> Color.R; 2 -> Color.B; else -> Color.G }
            CfaPattern.GBRG -> when (phase) { 1 -> Color.B; 2 -> Color.R; else -> Color.G }
            else -> error("Unsupported CFA $pattern")
        }
    }

    private fun evenRound(value: Double): Int = (value / 2.0).roundToInt() * 2

    companion object {
        private val SUPPORTED_CFA = setOf(CfaPattern.RGGB, CfaPattern.BGGR, CfaPattern.GRBG, CfaPattern.GBRG)
        private const val ALIGNMENT_LONG_SIDE = 288
        private const val GLOBAL_SEARCH = 14
        private const val LOCAL_TILE_SIZE = 32
        private const val LOCAL_SEARCH = 3
        private const val MIN_FRAME_ALIGNMENT_CONFIDENCE = 0.26
    }
}
