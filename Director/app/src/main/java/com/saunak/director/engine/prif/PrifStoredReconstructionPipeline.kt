package com.saunak.director.engine.prif

import android.graphics.Bitmap
import android.graphics.Matrix
import com.saunak.director.engine.reconstruction.CfaTranslation
import com.saunak.director.engine.reconstruction.DeterministicIsp
import com.saunak.director.engine.reconstruction.DisplayRgbImage
import com.saunak.director.engine.reconstruction.EvidenceLedger
import com.saunak.director.engine.reconstruction.EvidenceRecord
import com.saunak.director.engine.reconstruction.Matrix3x3
import com.saunak.director.engine.reconstruction.NoiseModel
import com.saunak.director.engine.reconstruction.NormalizedRect as EvidenceRect
import com.saunak.director.engine.reconstruction.RawImageFormation
import com.saunak.director.engine.reconstruction.RawObservation
import com.saunak.director.engine.reconstruction.RegionId
import com.saunak.director.engine.reconstruction.RegionSemantic
import com.saunak.director.engine.reconstruction.SierraColorCalibration
import com.saunak.director.engine.reconstruction.StoredRawBurstFrame
import com.saunak.director.engine.reconstruction.StoredRawBurstPipeline
import com.saunak.director.engine.reconstruction.WhiteBalanceGains
import com.saunak.director.engine.subject.NormalizedRect
import com.saunak.director.engine.subject.SubjectTrack
import kotlin.math.max
import kotlin.math.min

data class PrifLiveReconstruction(
    val bitmap: Bitmap,
    val sensorTwin: SensorDigitalTwin,
    val consistency: SensorConsistencyReport,
    val integrity: PhotoIntegrityReport,
    val evidenceRegionIds: List<RegionId>,
    val rotationDegrees: Int,
    val meanOcclusionProbability: Double
) {
    init {
        require(rotationDegrees in setOf(0, 90, 180, 270))
        require(meanOcclusionProbability in 0.0..1.0)
    }
}

/**
 * Disk-backed P-RIF reconstruction. Full RAW frames never need to be re-materialized after spill;
 * only one haloed tile per evidence frame plus one final output bitmap are resident at a time.
 */
class PrifStoredReconstructionPipeline(
    private val ledger: EvidenceLedger,
    private val tileSize: Int = 384,
    private val halo: Int = 2
) {
    init {
        require(tileSize in 128..1024 && tileSize % 2 == 0)
        require(halo in 2..16 && halo % 2 == 0)
    }

    fun reconstruct(
        frames: List<StoredRawBurstFrame>,
        program: ReconstructionProgram,
        subject: SubjectTrack?,
        motionRisk: Double,
        maxOutputPixels: Long,
        timestampNs: Long = System.nanoTime()
    ): PrifLiveReconstruction {
        require(frames.isNotEmpty())
        require(motionRisk in 0.0..1.0)
        require(maxOutputPixels > 0)
        val reference = frames.first().stored
        require(frames.all {
            it.stored.width == reference.width &&
                it.stored.height == reference.height &&
                it.stored.pattern == reference.pattern
        })
        val aligner = StoredRawBurstPipeline(ledger, tileSize, halo)
        val alignment = aligner.estimateAlignment(frames)
        val outputPixels = alignment.width.toLong() * alignment.height.toLong()
        require(outputPixels <= maxOutputPixels) {
            "P-RIF ${alignment.width}x${alignment.height} output exceeds live bitmap budget"
        }

        val bitmap = Bitmap.createBitmap(alignment.width, alignment.height, Bitmap.Config.ARGB_8888)
        val readers = frames.map { it.stored.openReader() }
        val evidenceIds = ArrayList<RegionId>()
        val frameScoreSums = DoubleArray(frames.size)
        val frameRmseSums = DoubleArray(frames.size)
        val frameInlierSums = DoubleArray(frames.size)
        val frameSaturationSums = DoubleArray(frames.size)
        var metricWeight = 0.0
        var supportSum = 0.0
        var multiFrameSum = 0.0
        var singleFrameSum = 0.0
        var uncertainSum = 0.0
        var uncertaintyPenaltySum = 0.0
        var occlusionSum = 0.0
        var renderedPixels = 0L
        val subjectSensor = subject?.takeIf { program.alignmentMode == PrifAlignmentMode.SEMANTIC_LAYERED }
            ?.let { orientedToSensorBounds(it.bounds, reference.rotationDegrees) }
        val calibration = colorCalibration(reference.metadata)

        try {
            var tileY = 0
            while (tileY < alignment.height) {
                val centerHeight = min(tileSize, alignment.height - tileY)
                var tileX = 0
                while (tileX < alignment.width) {
                    val centerWidth = min(tileSize, alignment.width - tileX)
                    val leftHalo = min(halo, tileX).and(-2)
                    val topHalo = min(halo, tileY).and(-2)
                    val rightHalo = min(halo, alignment.width - tileX - centerWidth).and(-2)
                    val bottomHalo = min(halo, alignment.height - tileY - centerHeight).and(-2)
                    val readWidth = centerWidth + leftHalo + rightHalo
                    val readHeight = centerHeight + topHalo + bottomHalo
                    val refX = alignment.cropX + tileX - leftHalo
                    val refY = alignment.cropY + tileY - topHalo

                    val observations = frames.indices.map { index ->
                        val translation: CfaTranslation = alignment.translations[index]
                        val raw = readers[index].readTile(
                            refX + translation.dx,
                            refY + translation.dy,
                            readWidth,
                            readHeight
                        )
                        val metadata = frames[index].stored.metadata
                        val rolling = RollingShutterEvidenceModel.globalAlignmentConfidence(
                            metadata?.telemetry?.rollingShutterSkewNs,
                            motionRisk
                        )
                        RawObservation(
                            raw = raw,
                            exposureTimeNs = metadata?.telemetry?.exposureTimeNs ?: DEFAULT_EXPOSURE_NS,
                            sensitivityIso = metadata?.telemetry?.sensitivityIso ?: DEFAULT_ISO,
                            alignmentConfidence = if (index == 0) 1f else (translation.confidence * rolling).toFloat(),
                            motionConfidence = if (index == 0) 1f else (1.0 - 0.70 * motionRisk).coerceIn(0.18, 0.98).toFloat(),
                            saturationConfidence = saturationConfidence(raw),
                            noise = metadata?.representativeNoise() ?: NoiseModel(0.002f, 0.0001f)
                        )
                    }

                    val occlusion = PrifOcclusionEstimator.estimate(observations)
                    val semanticSupport = subjectSensor?.let { bounds ->
                        semanticSupport(
                            bounds = bounds,
                            tileOriginX = refX,
                            tileOriginY = refY,
                            originalWidth = reference.width,
                            originalHeight = reference.height,
                            protection = program.subjectProtection
                        )
                    }
                    val posterior = PrifBayesianFusion.fuse(
                        observations = observations,
                        degreesOfFreedom = program.robustDegreesOfFreedom,
                        maxNormalizedResidual = program.maxNormalizedResidual,
                        conservativeAnchor = program.fusionMode == PrifFusionMode.CONSERVATIVE_ANCHOR,
                        spatialSupport = { frameIndex, x, y ->
                            val i = y * readWidth + x
                            val visibility = occlusion.visibility[frameIndex][i]
                            val semantic = semanticSupport?.invoke(frameIndex, x, y) ?: 1.0
                            (visibility * semantic).coerceIn(0.0, 1.0)
                        }
                    )
                    val consistency = SensorConsistencyVerifier.verify(
                        posterior,
                        observations,
                        acceptanceThreshold = TILE_CONSISTENCY_THRESHOLD
                    )
                    val integrity = PhotoIntegrityEvaluator.evaluate(
                        posterior,
                        consistency,
                        acceptanceThreshold = TILE_INTEGRITY_THRESHOLD
                    )
                    val display = DeterministicIsp.render(
                        RawImageFormation.demosaic(posterior.asNormalizedRaw()),
                        calibration
                    )
                    writeCenterTile(
                        display, bitmap, tileX, tileY, centerWidth, centerHeight, leftHalo, topHalo
                    )

                    val pixels = centerWidth.toLong() * centerHeight.toLong()
                    renderedPixels += pixels
                    val weight = pixels.toDouble()
                    metricWeight += weight
                    supportSum += integrity.physicalSupport * weight
                    multiFrameSum += integrity.provenance.multiFrameFraction * weight
                    singleFrameSum += integrity.provenance.singleFrameFraction * weight
                    uncertainSum += integrity.provenance.uncertainFraction * weight
                    uncertaintyPenaltySum += integrity.uncertaintyPenalty * weight
                    occlusionSum += occlusion.meanOcclusionProbability * weight
                    for (i in consistency.frames.indices) {
                        frameScoreSums[i] += consistency.frames[i].score * weight
                        frameRmseSums[i] += consistency.frames[i].normalizedRmse * weight
                        frameInlierSums[i] += consistency.frames[i].inlierFraction * weight
                        frameSaturationSums[i] += consistency.frames[i].saturationFraction * weight
                    }
                    evidenceIds += publishTileEvidence(
                        frames = frames,
                        posterior = posterior,
                        observations = observations,
                        consistency = consistency,
                        occlusion = occlusion,
                        tileX = tileX,
                        tileY = tileY,
                        centerWidth = centerWidth,
                        centerHeight = centerHeight,
                        leftHalo = leftHalo,
                        topHalo = topHalo,
                        alignmentWidth = alignment.width,
                        alignmentHeight = alignment.height,
                        timestampNs = timestampNs
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

        val denom = metricWeight.coerceAtLeast(1.0)
        val frameReports = frames.indices.map { index ->
            FrameConsistency(
                frameIndex = index,
                normalizedRmse = frameRmseSums[index] / denom,
                inlierFraction = (frameInlierSums[index] / denom).coerceIn(0.0, 1.0),
                saturationFraction = (frameSaturationSums[index] / denom).coerceIn(0.0, 1.0),
                score = (frameScoreSums[index] / denom).coerceIn(0.0, 1.0)
            )
        }
        val aggregateConsistency = frameReports.map { it.score }.average().coerceIn(0.0, 1.0)
        val consistency = SensorConsistencyReport(
            frames = frameReports,
            aggregateScore = aggregateConsistency,
            accepted = aggregateConsistency >= GLOBAL_CONSISTENCY_THRESHOLD
        )
        val provenance = ProvenanceSummary(
            averagePhysicalSupport = (supportSum / denom).coerceIn(0.0, 1.0),
            multiFrameFraction = (multiFrameSum / denom).coerceIn(0.0, 1.0),
            uncertainFraction = (uncertainSum / denom).coerceIn(0.0, 1.0),
            singleFrameFraction = (singleFrameSum / denom).coerceIn(0.0, 1.0),
            neuralFraction = 0.0
        )
        val uncertaintyPenalty = (uncertaintyPenaltySum / denom).coerceIn(0.0, 1.0)
        val integrityScore = (
            0.46 * consistency.aggregateScore +
                0.40 * provenance.averagePhysicalSupport +
                0.14 * provenance.multiFrameFraction -
                0.22 * uncertaintyPenalty
            ).coerceIn(0.0, 1.0)
        val integrity = PhotoIntegrityReport(
            score = integrityScore,
            sensorConsistency = consistency.aggregateScore,
            physicalSupport = provenance.averagePhysicalSupport,
            uncertaintyPenalty = uncertaintyPenalty,
            provenance = provenance,
            accepted = consistency.accepted && integrityScore >= GLOBAL_INTEGRITY_THRESHOLD
        )
        val oriented = orient(bitmap, reference.rotationDegrees)
        return PrifLiveReconstruction(
            bitmap = oriented,
            sensorTwin = sensorTwin(reference),
            consistency = consistency,
            integrity = integrity,
            evidenceRegionIds = evidenceIds,
            rotationDegrees = reference.rotationDegrees,
            meanOcclusionProbability = (occlusionSum / denom).coerceIn(0.0, 1.0)
        )
    }

    private fun semanticSupport(
        bounds: NormalizedRect,
        tileOriginX: Int,
        tileOriginY: Int,
        originalWidth: Int,
        originalHeight: Int,
        protection: Double
    ): (Int, Int, Int) -> Double {
        val expanded = bounds.expanded(1.18, 1.18)
        val core = (1.0 - 0.90 * protection).coerceIn(0.08, 1.0)
        val edge = (1.0 - 0.55 * protection).coerceIn(0.28, 1.0)
        return support@{ frameIndex, x, y ->
            if (frameIndex == 0) return@support 1.0
            val nx = (tileOriginX + x + 0.5) / originalWidth.toDouble()
            val ny = (tileOriginY + y + 0.5) / originalHeight.toDouble()
            when {
                bounds.contains(nx, ny) -> core
                expanded.contains(nx, ny) -> edge
                else -> 1.0
            }
        }
    }

    private fun publishTileEvidence(
        frames: List<StoredRawBurstFrame>,
        posterior: RadiancePosterior,
        observations: List<RawObservation>,
        consistency: SensorConsistencyReport,
        occlusion: PrifOcclusionEstimator.Result,
        tileX: Int,
        tileY: Int,
        centerWidth: Int,
        centerHeight: Int,
        leftHalo: Int,
        topHalo: Int,
        alignmentWidth: Int,
        alignmentHeight: Int,
        timestampNs: Long
    ): RegionId {
        val weights = HashMap<Long, Double>()
        var support = 0.0
        var sensorVar = 0.0
        for (y in 0 until centerHeight) {
            for (x in 0 until centerWidth) {
                val i = (y + topHalo) * posterior.width + (x + leftHalo)
                support += posterior.support[i]
                sensorVar += posterior.sensorVariance[i]
                val local = (posterior.dominantFrame[i].toInt() and 0xff).coerceIn(frames.indices)
                val id = frames[local].frameId
                weights[id] = (weights[id] ?: 0.0) + max(1e-5, posterior.support[i].toDouble())
            }
        }
        val pixelCount = max(1, centerWidth * centerHeight).toDouble()
        val meanSupport = (support / pixelCount).coerceIn(0.0, 1.0)
        val snrConfidence = (1.0 / (1.0 + 12.0 * sensorVar / pixelCount)).coerceIn(0.0, 1.0)
        val regionId = RegionId("prif/tile/${tileX}_${tileY}_$timestampNs")
        ledger.put(
            EvidenceRecord(
                regionId = regionId,
                bounds = EvidenceRect(
                    tileX.toDouble() / alignmentWidth,
                    tileY.toDouble() / alignmentHeight,
                    (tileX + centerWidth).toDouble() / alignmentWidth,
                    (tileY + centerHeight).toDouble() / alignmentHeight
                ),
                semantic = RegionSemantic.GENERAL,
                sourceFrameIds = frames.map { it.frameId }.toSet(),
                fusionWeights = if (weights.isNotEmpty()) weights else mapOf(frames.first().frameId to 1.0),
                alignmentConfidence = observations.map { it.alignmentConfidence.toDouble() }.average().coerceIn(0.0, 1.0),
                motionConfidence = observations.map { it.motionConfidence.toDouble() }.average().coerceIn(0.0, 1.0),
                exposureValidity = consistency.frames.map { 1.0 - it.saturationFraction }.average().coerceIn(0.0, 1.0),
                signalToNoiseConfidence = (0.65 * snrConfidence + 0.35 * meanSupport).coerceIn(0.0, 1.0),
                saturationConfidence = observations.map { it.saturationConfidence.toDouble() }.average().coerceIn(0.0, 1.0),
                geometryConfidence = (1.0 - occlusion.meanOcclusionProbability).coerceIn(0.0, 1.0),
                neuralResidual = 0.0,
                timestampNs = timestampNs
            )
        )
        return regionId
    }

    private fun writeCenterTile(
        display: DisplayRgbImage,
        bitmap: Bitmap,
        tileX: Int,
        tileY: Int,
        centerWidth: Int,
        centerHeight: Int,
        leftHalo: Int,
        topHalo: Int
    ) {
        val pixels = IntArray(centerWidth * centerHeight)
        for (y in 0 until centerHeight) {
            for (x in 0 until centerWidth) {
                val source = ((y + topHalo) * display.width + (x + leftHalo)) * 3
                val r = (display.rgb[source] * 255f + 0.5f).toInt().coerceIn(0, 255)
                val g = (display.rgb[source + 1] * 255f + 0.5f).toInt().coerceIn(0, 255)
                val b = (display.rgb[source + 2] * 255f + 0.5f).toInt().coerceIn(0, 255)
                pixels[y * centerWidth + x] = (0xff shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        bitmap.setPixels(pixels, 0, centerWidth, tileX, tileY, centerWidth, centerHeight)
    }

    private fun colorCalibration(metadata: com.saunak.director.camera.RawCaptureMetadata?) = SierraColorCalibration(
        whiteBalance = metadata?.whiteBalance ?: WhiteBalanceGains(1f, 1f, 1f),
        cameraToWorking = metadata?.cameraToLinearSrgb ?: Matrix3x3.IDENTITY,
        exposureGain = 1f,
        highlightShoulder = 0.84f,
        contrast = 1f
    )

    private fun sensorTwin(frame: com.saunak.director.engine.reconstruction.StoredRawFrame): SensorDigitalTwin {
        val metadata = frame.metadata
        var confidence = 0.45
        if (metadata?.noiseProfile?.size == 4) confidence += 0.18
        if (metadata?.dynamicBlackLevel?.size == 4) confidence += 0.12
        if (metadata?.cameraToLinearSrgb != null) confidence += 0.10
        if (metadata?.whiteBalance != null) confidence += 0.08
        if (metadata?.telemetry?.rollingShutterSkewNs != null) confidence += 0.07
        return SensorDigitalTwin(
            cameraId = metadata?.telemetry?.cameraId,
            bayerPattern = frame.pattern,
            blackLevel = frame.blackLevelPattern,
            whiteLevel = frame.whiteLevel,
            representativeNoise = metadata?.representativeNoise() ?: NoiseModel(0.002f, 0.0001f),
            rollingShutterSkewNs = metadata?.telemetry?.rollingShutterSkewNs,
            focalLengthMm = metadata?.telemetry?.focalLengthMm,
            aperture = metadata?.telemetry?.aperture,
            confidence = confidence.coerceIn(0.0, 1.0)
        )
    }

    private fun saturationConfidence(raw: com.saunak.director.engine.reconstruction.RawPlane): Float {
        val normalized = RawImageFormation.normalize(raw)
        var saturated = 0
        var count = 0
        var i = 0
        while (i < normalized.size) {
            if (normalized[i] >= 0.985f) saturated++
            count++
            i += 16
        }
        return (1f - saturated.toFloat() / max(1, count).toFloat()).coerceIn(0.05f, 1f)
    }

    private fun orient(source: Bitmap, rotation: Int): Bitmap {
        if (rotation == 0) return source
        val result = Bitmap.createBitmap(
            source, 0, 0, source.width, source.height,
            Matrix().apply { postRotate(rotation.toFloat()) }, true
        )
        if (result !== source && !source.isRecycled) source.recycle()
        return result
    }

    private fun NormalizedRect.contains(x: Double, y: Double): Boolean =
        x >= left && x <= right && y >= top && y <= bottom

    companion object {
        private const val DEFAULT_EXPOSURE_NS = 10_000_000L
        private const val DEFAULT_ISO = 100
        private const val TILE_CONSISTENCY_THRESHOLD = 0.42
        private const val TILE_INTEGRITY_THRESHOLD = 0.42
        private const val GLOBAL_CONSISTENCY_THRESHOLD = 0.58
        private const val GLOBAL_INTEGRITY_THRESHOLD = 0.56
    }
}
