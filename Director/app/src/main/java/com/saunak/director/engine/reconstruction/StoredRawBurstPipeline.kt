package com.saunak.director.engine.reconstruction

import android.graphics.Bitmap
import com.saunak.director.camera.RawCaptureMetadata
import kotlin.math.max
import kotlin.math.min

data class StoredRawBurstFrame(
    val frameId: Long,
    val stored: StoredRawFrame
) {
    init { require(frameId > 0) }
}

data class StoredBurstAlignment(
    val translations: List<CfaTranslation>,
    val cropX: Int,
    val cropY: Int,
    val width: Int,
    val height: Int
)

data class StoredRawReconstruction(
    val bitmap: Bitmap,
    val rotationDegrees: Int,
    val alignment: StoredBurstAlignment,
    val evidenceRegionIds: List<RegionId>
)

class RawPortableBudgetExceeded(message: String) : IllegalStateException(message)

/**
 * Memory-bounded RAW reconstruction for the Android portable baseline. RAW frames stay disk-backed;
 * only small CFA-preserving tiles are materialized as FloatArrays. A device/native backend may later
 * replace this implementation without changing evidence or capture contracts.
 */
class StoredRawBurstPipeline(
    private val ledger: EvidenceLedger,
    private val tileSize: Int = 512,
    private val halo: Int = 2
) {
    init {
        require(tileSize in 128..1024 && tileSize % 2 == 0)
        require(halo in 2..16 && halo % 2 == 0)
    }

    fun reconstruct(
        frames: List<StoredRawBurstFrame>,
        calibration: SierraColorCalibration,
        maxOutputPixels: Long,
        timestampNs: Long
    ): StoredRawReconstruction {
        require(frames.isNotEmpty())
        require(maxOutputPixels > 0)
        require(timestampNs > 0)
        val reference = frames.first().stored
        require(frames.all {
            it.stored.width == reference.width &&
                it.stored.height == reference.height &&
                it.stored.pattern == reference.pattern
        })

        val alignment = estimateAlignment(frames)
        val pixels = alignment.width.toLong() * alignment.height.toLong()
        if (pixels > maxOutputPixels) {
            throw RawPortableBudgetExceeded(
                "${alignment.width}x${alignment.height} RAW output exceeds portable ${maxOutputPixels} pixel budget"
            )
        }

        val bitmap = Bitmap.createBitmap(alignment.width, alignment.height, Bitmap.Config.ARGB_8888)
        val evidenceIds = ArrayList<RegionId>()
        val readers = frames.map { it.stored.openReader() }
        try {
            var tileY = 0
            while (tileY < alignment.height) {
                val centerHeight = min(tileSize, alignment.height - tileY)
                var tileX = 0
                while (tileX < alignment.width) {
                    val centerWidth = min(tileSize, alignment.width - tileX)
                    reconstructTile(
                        frames, readers, alignment, calibration,
                        tileX, tileY, centerWidth, centerHeight,
                        timestampNs, bitmap, evidenceIds
                    )
                    tileX += centerWidth
                }
                tileY += centerHeight
            }
        } catch (t: Throwable) {
            bitmap.recycle()
            throw t
        } finally {
            readers.forEach { runCatching { it.close() } }
        }
        return StoredRawReconstruction(
            bitmap = bitmap,
            rotationDegrees = reference.rotationDegrees,
            alignment = alignment,
            evidenceRegionIds = evidenceIds
        )
    }

    fun estimateAlignment(frames: List<StoredRawBurstFrame>): StoredBurstAlignment {
        require(frames.isNotEmpty())
        val reference = frames.first().stored
        val sampleWidth = min(1024, reference.width).let { it - (it and 1) }.coerceAtLeast(4)
        val sampleHeight = min(1024, reference.height).let { it - (it and 1) }.coerceAtLeast(4)
        val sampleX = ((reference.width - sampleWidth) / 2).and(-2)
        val sampleY = ((reference.height - sampleHeight) / 2).and(-2)

        val referenceTile = reference.openReader().use { it.readTile(sampleX, sampleY, sampleWidth, sampleHeight) }
        val translations = frames.mapIndexed { index, frame ->
            if (index == 0) CfaTranslation(0, 0, 1f, 0f)
            else frame.stored.openReader().use { reader ->
                val candidate = reader.readTile(sampleX, sampleY, sampleWidth, sampleHeight)
                CfaBurstAlignment.estimate(referenceTile, candidate, 16)
            }
        }

        var left = 0
        var top = 0
        var right = reference.width
        var bottom = reference.height
        for (t in translations) {
            left = max(left, -t.dx)
            top = max(top, -t.dy)
            right = min(right, reference.width - t.dx)
            bottom = min(bottom, reference.height - t.dy)
        }
        left = left.and(-2) + if ((left and 1) != 0) 2 else 0
        top = top.and(-2) + if ((top and 1) != 0) 2 else 0
        right = right.and(-2)
        bottom = bottom.and(-2)
        require(right - left >= 16 && bottom - top >= 16)
        return StoredBurstAlignment(translations, left, top, right - left, bottom - top)
    }

    private fun reconstructTile(
        frames: List<StoredRawBurstFrame>,
        readers: List<StoredRawReader>,
        alignment: StoredBurstAlignment,
        calibration: SierraColorCalibration,
        tileX: Int,
        tileY: Int,
        centerWidth: Int,
        centerHeight: Int,
        timestampNs: Long,
        bitmap: Bitmap,
        evidenceIds: MutableList<RegionId>
    ) {
        val leftHalo = min(halo, tileX).and(-2)
        val topHalo = min(halo, tileY).and(-2)
        val rightHalo = min(halo, alignment.width - tileX - centerWidth).and(-2)
        val bottomHalo = min(halo, alignment.height - tileY - centerHeight).and(-2)
        val readWidth = centerWidth + leftHalo + rightHalo
        val readHeight = centerHeight + topHalo + bottomHalo
        val refX = alignment.cropX + tileX - leftHalo
        val refY = alignment.cropY + tileY - topHalo

        val observations = frames.indices.map { index ->
            val t = alignment.translations[index]
            val raw = readers[index].readTile(refX + t.dx, refY + t.dy, readWidth, readHeight)
            val metadata = frames[index].stored.metadata
            RawObservation(
                raw = raw,
                exposureTimeNs = metadata?.telemetry?.exposureTimeNs ?: DEFAULT_EXPOSURE_NS,
                sensitivityIso = metadata?.telemetry?.sensitivityIso ?: DEFAULT_ISO,
                alignmentConfidence = t.confidence,
                motionConfidence = motionConfidence(metadata),
                saturationConfidence = saturationConfidence(raw),
                noise = metadata?.representativeNoise() ?: NoiseModel(0.002f, 0.0001f)
            )
        }
        val fused = RawImageFormation.fuseAligned(observations)
        val linear = RawImageFormation.demosaic(fused)
        val display = DeterministicIsp.render(linear, calibration)
        val pixels = IntArray(centerWidth * centerHeight)

        for (y in 0 until centerHeight) {
            for (x in 0 until centerWidth) {
                val sourcePixel = ((y + topHalo) * readWidth + (x + leftHalo)) * 3
                val r = (display.rgb[sourcePixel] * 255f + 0.5f).toInt().coerceIn(0, 255)
                val g = (display.rgb[sourcePixel + 1] * 255f + 0.5f).toInt().coerceIn(0, 255)
                val b = (display.rgb[sourcePixel + 2] * 255f + 0.5f).toInt().coerceIn(0, 255)
                pixels[y * centerWidth + x] = (0xff shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        bitmap.setPixels(pixels, 0, centerWidth, tileX, tileY, centerWidth, centerHeight)

        val regionId = RegionId("raw/tile/${tileX}_${tileY}_$timestampNs")
        val frameWeights = frames.indices.associate { index ->
            frames[index].frameId to max(1e-6, observations[index].alignmentConfidence.toDouble() /
                max(observations[index].noise.readNoiseVariance.toDouble(), 1e-6))
        }
        val evidence = EvidenceRecord(
            regionId = regionId,
            bounds = NormalizedRect(
                tileX.toDouble() / alignment.width,
                tileY.toDouble() / alignment.height,
                (tileX + centerWidth).toDouble() / alignment.width,
                (tileY + centerHeight).toDouble() / alignment.height
            ),
            semantic = RegionSemantic.GENERAL,
            sourceFrameIds = frames.map { it.frameId }.toSet(),
            fusionWeights = frameWeights,
            alignmentConfidence = observations.map { it.alignmentConfidence.toDouble() }.average(),
            motionConfidence = observations.map { it.motionConfidence.toDouble() }.average(),
            exposureValidity = 1.0,
            signalToNoiseConfidence = observations.map {
                (1.0 / (1.0 + 100.0 * it.noise.readNoiseVariance)).coerceIn(0.0, 1.0)
            }.average(),
            saturationConfidence = observations.map { it.saturationConfidence.toDouble() }.average(),
            geometryConfidence = observations.map { it.alignmentConfidence.toDouble() }.average(),
            neuralResidual = 0.0,
            timestampNs = timestampNs
        )
        ledger.put(evidence)
        evidenceIds += regionId
    }

    private fun motionConfidence(metadata: RawCaptureMetadata?): Float {
        // Until gyro residuals are wired per tile, short exposures receive conservative confidence.
        val exposure = metadata?.telemetry?.exposureTimeNs ?: return 0.65f
        return when {
            exposure <= 8_000_000L -> 0.93f
            exposure <= 16_000_000L -> 0.82f
            exposure <= 33_000_000L -> 0.70f
            else -> 0.52f
        }
    }

    private fun saturationConfidence(raw: RawPlane): Float {
        val normalized = RawImageFormation.normalize(raw)
        var saturated = 0
        val stride = 16
        var i = 0
        while (i < normalized.size) {
            if (normalized[i] >= 0.985f) saturated++
            i += stride
        }
        val samples = ((normalized.size + stride - 1) / stride).coerceAtLeast(1)
        return (1f - saturated.toFloat() / samples.toFloat()).coerceIn(0.05f, 1f)
    }

    companion object {
        private const val DEFAULT_EXPOSURE_NS = 10_000_000L
        private const val DEFAULT_ISO = 100
    }
}
