package com.saunak.director.engine.prif

import android.graphics.Bitmap
import android.graphics.Matrix
import com.saunak.director.camera.RawJpegCaptureFrame
import com.saunak.director.engine.capture.CaptureProgram
import com.saunak.director.engine.motion.MotionSnapshot
import com.saunak.director.engine.reconstruction.CfaAlignedBurst
import com.saunak.director.engine.reconstruction.CfaBurstAlignment
import com.saunak.director.engine.reconstruction.DeterministicIsp
import com.saunak.director.engine.reconstruction.DisplayRgbImage
import com.saunak.director.engine.reconstruction.Matrix3x3
import com.saunak.director.engine.reconstruction.RawImageFormation
import com.saunak.director.engine.reconstruction.RawObservation
import com.saunak.director.engine.reconstruction.RawPlane
import com.saunak.director.engine.reconstruction.SierraColorCalibration
import com.saunak.director.engine.reconstruction.WhiteBalanceGains
import com.saunak.director.engine.subject.NormalizedRect
import com.saunak.director.engine.subject.SubjectTrack
import kotlin.math.max

data class PrifReconstructionResult(
    val bitmap: Bitmap,
    val posterior: RadiancePosterior,
    val sensorTwin: SensorDigitalTwin,
    val consistency: SensorConsistencyReport,
    val integrity: PhotoIntegrityReport,
    val program: ReconstructionProgram,
    val usedCaptureIndices: List<Int>,
    val rotationDegrees: Int
) {
    init {
        require(usedCaptureIndices.isNotEmpty())
        require(rotationDegrees in setOf(0, 90, 180, 270))
    }
}

/** Trusted P-RIF physical core. Neural restoration is deliberately downstream of this stage. */
class PrifReconstructionPipeline {
    fun reconstruct(
        captured: List<RawJpegCaptureFrame>,
        captureProgram: CaptureProgram,
        motion: MotionSnapshot?,
        subject: SubjectTrack?,
        thermalPressure: Double,
        memoryPressure: Double
    ): PrifReconstructionResult {
        require(captured.isNotEmpty())
        require(captured.size == captureProgram.frames.size)
        val program = PrifReconstructionCompiler.compile(
            captureProgram, motion, subject, thermalPressure, memoryPressure
        )
        val selectedIndices = PrifReconstructionCompiler.selectCaptureIndices(program)
        val orderedIndices = listOf(program.referenceFrameIndex) + selectedIndices.filter { it != program.referenceFrameIndex }
        val selected = orderedIndices.map(captured::get)
        val motionConfidence = (1.0 - (motion?.conservativeRisk ?: 0.75) * 0.55).coerceIn(0.18, 0.98)

        val observations = selected.mapIndexed { localIndex, frame ->
            RawObservation(
                raw = frame.raw.raw,
                exposureTimeNs = frame.metadata.telemetry.exposureTimeNs,
                sensitivityIso = frame.metadata.telemetry.sensitivityIso,
                alignmentConfidence = if (localIndex == 0) 1f else 0.92f,
                motionConfidence = if (localIndex == 0) 1f else motionConfidence.toFloat(),
                saturationConfidence = saturationConfidence(frame.raw.raw),
                noise = frame.metadata.representativeNoise()
            )
        }

        val aligned = CfaBurstAlignment.align(observations, program.alignmentSearchRadiusPx)
        val referenceCapture = selected.first()
        val subjectSensorBounds = subject?.takeIf { program.alignmentMode == PrifAlignmentMode.SEMANTIC_LAYERED }
            ?.let { orientedToSensorBounds(it.bounds, referenceCapture.raw.rotationDegrees) }
        val spatialSupport = subjectSensorBounds?.let { bounds ->
            buildSubjectSupport(
                bounds, aligned, referenceCapture.raw.raw.width, referenceCapture.raw.raw.height,
                program.subjectProtection
            )
        }

        val posterior = PrifBayesianFusion.fuse(
            observations = aligned.observations,
            degreesOfFreedom = program.robustDegreesOfFreedom,
            maxNormalizedResidual = program.maxNormalizedResidual,
            conservativeAnchor = program.fusionMode == PrifFusionMode.CONSERVATIVE_ANCHOR,
            spatialSupport = spatialSupport
        )
        val consistency = if (program.verifySensorConsistency) {
            SensorConsistencyVerifier.verify(posterior, aligned.observations)
        } else {
            SensorConsistencyReport(listOf(FrameConsistency(0, 0.0, 1.0, 0.0, 1.0)), 1.0, true)
        }
        val integrity = PhotoIntegrityEvaluator.evaluate(posterior, consistency)
        val linear = RawImageFormation.demosaic(posterior.asNormalizedRaw())
        val display = DeterministicIsp.render(linear, colorCalibration(referenceCapture))
        val sensorBitmap = display.toBitmap()
        val orientedBitmap = orient(sensorBitmap, referenceCapture.raw.rotationDegrees)

        return PrifReconstructionResult(
            bitmap = orientedBitmap,
            posterior = posterior,
            sensorTwin = sensorTwin(referenceCapture),
            consistency = consistency,
            integrity = integrity,
            program = program,
            usedCaptureIndices = orderedIndices,
            rotationDegrees = referenceCapture.raw.rotationDegrees
        )
    }

    private fun buildSubjectSupport(
        bounds: NormalizedRect,
        aligned: CfaAlignedBurst,
        originalWidth: Int,
        originalHeight: Int,
        protection: Double
    ): (Int, Int, Int) -> Double {
        val expanded = bounds.expanded(1.18, 1.18)
        val strongScale = (1.0 - 0.90 * protection).coerceIn(0.08, 1.0)
        val boundaryScale = (1.0 - 0.55 * protection).coerceIn(0.28, 1.0)
        return support@{ frameIndex, x, y ->
            if (frameIndex == 0) return@support 1.0
            val nx = (aligned.referenceCropX + x + 0.5) / originalWidth.toDouble()
            val ny = (aligned.referenceCropY + y + 0.5) / originalHeight.toDouble()
            when {
                bounds.contains(nx, ny) -> strongScale
                expanded.contains(nx, ny) -> boundaryScale
                else -> 1.0
            }
        }
    }

    private fun orientedToSensorBounds(bounds: NormalizedRect, rotation: Int): NormalizedRect = when (rotation) {
        0 -> bounds
        90 -> NormalizedRect(bounds.top, 1.0 - bounds.right, bounds.bottom, 1.0 - bounds.left).clamped()
        180 -> NormalizedRect(1.0 - bounds.right, 1.0 - bounds.bottom, 1.0 - bounds.left, 1.0 - bounds.top).clamped()
        270 -> NormalizedRect(1.0 - bounds.bottom, bounds.left, 1.0 - bounds.top, bounds.right).clamped()
        else -> bounds
    }

    private fun colorCalibration(frame: RawJpegCaptureFrame) = SierraColorCalibration(
        whiteBalance = frame.metadata.whiteBalance ?: WhiteBalanceGains(1f, 1f, 1f),
        cameraToWorking = frame.metadata.cameraToLinearSrgb ?: Matrix3x3.IDENTITY,
        exposureGain = 1f,
        highlightShoulder = 0.84f,
        contrast = 1f
    )

    private fun sensorTwin(frame: RawJpegCaptureFrame) = SensorDigitalTwin(
        cameraId = frame.metadata.telemetry.cameraId,
        bayerPattern = frame.raw.raw.pattern,
        blackLevel = frame.raw.raw.blackLevelPattern,
        whiteLevel = frame.raw.raw.whiteLevel,
        representativeNoise = frame.metadata.representativeNoise(),
        rollingShutterSkewNs = frame.metadata.telemetry.rollingShutterSkewNs,
        focalLengthMm = frame.metadata.telemetry.focalLengthMm,
        aperture = frame.metadata.telemetry.aperture,
        confidence = digitalTwinConfidence(frame)
    )

    private fun digitalTwinConfidence(frame: RawJpegCaptureFrame): Double {
        var score = 0.45
        if (frame.metadata.noiseProfile.size == 4) score += 0.18
        if (frame.metadata.dynamicBlackLevel?.size == 4) score += 0.12
        if (frame.metadata.cameraToLinearSrgb != null) score += 0.10
        if (frame.metadata.whiteBalance != null) score += 0.08
        if (frame.metadata.telemetry.rollingShutterSkewNs != null) score += 0.07
        return score.coerceIn(0.0, 1.0)
    }

    private fun saturationConfidence(raw: RawPlane): Float {
        val normalized = RawImageFormation.normalize(raw)
        var saturated = 0
        var samples = 0
        var i = 0
        while (i < normalized.size) {
            if (normalized[i] >= 0.985f) saturated++
            samples++
            i += 32
        }
        return (1f - saturated.toFloat() / max(1, samples).toFloat()).coerceIn(0.05f, 1f)
    }

    private fun DisplayRgbImage.toBitmap(): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)
        for (i in pixels.indices) {
            val base = i * 3
            val r = (rgb[base] * 255f + 0.5f).toInt().coerceIn(0, 255)
            val g = (rgb[base + 1] * 255f + 0.5f).toInt().coerceIn(0, 255)
            val b = (rgb[base + 2] * 255f + 0.5f).toInt().coerceIn(0, 255)
            pixels[i] = (0xff shl 24) or (r shl 16) or (g shl 8) or b
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    private fun orient(source: Bitmap, rotation: Int): Bitmap {
        if (rotation == 0) return source
        val rotated = Bitmap.createBitmap(source, 0, 0, source.width, source.height, Matrix().apply { postRotate(rotation.toFloat()) }, true)
        if (rotated !== source && !source.isRecycled) source.recycle()
        return rotated
    }

    private fun NormalizedRect.contains(x: Double, y: Double): Boolean =
        x >= left && x <= right && y >= top && y <= bottom
}
