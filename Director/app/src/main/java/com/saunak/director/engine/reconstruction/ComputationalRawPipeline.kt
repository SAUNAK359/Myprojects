package com.saunak.director.engine.reconstruction

import android.graphics.Bitmap
import android.graphics.Matrix
import com.saunak.director.camera.CameraEngine
import com.saunak.director.camera.ManualRawFrameSpec
import com.saunak.director.camera.RawJpegBurstCapture
import com.saunak.director.engine.SierraEngine
import com.saunak.director.engine.capture.CapturePath
import com.saunak.director.engine.capture.CaptureProgram
import com.saunak.director.engine.device.CfaPattern
import com.saunak.director.engine.isp.SierraIspBitmapSink
import com.saunak.director.engine.isp.SierraIspPlan
import com.saunak.director.engine.isp.SierraRawIspPlanner
import com.saunak.director.processing.PhotoQuality
import com.saunak.director.processing.TechnicalQuality
import com.saunak.director.storage.PhotoRepository
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** Fully owned computational result; caller must close it after saving/fallback. */
class ComputationalRawResult internal constructor(
    val bitmap: Bitmap,
    /** Protected no-backup JPEG that survives process death until the caller marks finalization. */
    val safetyJpeg: File,
    val capture: RawJpegBurstCapture,
    val reconstruction: RawReconstructionSummary,
    val ispPlan: SierraIspPlan,
    val computationalQuality: TechnicalQuality,
    val safetyQuality: TechnicalQuality,
    val acceptedOverSafetyJpeg: Boolean,
    val rejectionReason: String?
) : AutoCloseable {
    override fun close() {
        if (!bitmap.isRecycled) bitmap.recycle()
        capture.cleanup()
        // safetyJpeg intentionally remains until MediaStore commit succeeds or startup recovery runs.
    }
}

/**
 * High-level Sierra computational-photography transaction.
 *
 * A RAW result is never trusted merely because reconstruction completed. It must have sufficient
 * physical-evidence confidence and must not materially regress technical quality against the JPEG
 * produced from the same sensor capture. The JPEG remains an independently decodable safety path.
 */
class ComputationalRawPipeline(
    private val camera: CameraEngine,
    private val storage: PhotoRepository,
    private val engine: SierraEngine = SierraEngine.get(),
    private val reconstructor: RawBurstReconstructor = RawBurstReconstructor()
) {
    suspend fun execute(sessionId: Long, program: CaptureProgram): Result<ComputationalRawResult> = runCatching {
        require(program.path == CapturePath.RAW_CUSTOM) { "Computational RAW requires RAW_CUSTOM program" }
        require(program.frames.isNotEmpty())
        check(camera.supportsComputationalRawBurst()) { "Camera session does not support certified RAW+JPEG manual burst" }

        val specs = program.frames.map { frame ->
            ManualRawFrameSpec(
                exposureTimeNs = frame.exposureTimeNs,
                sensitivityIso = frame.sensitivityIso,
                role = frame.role.name,
                evOffset = frame.evOffset
            )
        }
        val capture = camera.captureRawJpegBurst(specs)
        val protectedSafety = storage.newPendingReference(sessionId)
        try {
            // Protect real captured evidence before any expensive reconstruction work begins.
            capture.frames.first().jpegFile.copyTo(protectedSafety, overwrite = true)
            check(protectedSafety.isFile && protectedSafety.length() > 0) { "Could not protect RAW safety JPEG" }

            val cameraId = capture.frames.firstNotNullOfOrNull { it.observed?.cameraId } ?: camera.activeCameraId()
                ?: error("Active Camera2 id unavailable after RAW capture")
            val twin = (engine.currentDeviceTwin() ?: engine.deviceTwin().getOrThrow()).camera(cameraId)
                ?: error("DeviceTwin does not contain active camera $cameraId")
            require(twin.cfaPattern in SUPPORTED_CFA) { "Unsupported RAW CFA ${twin.cfaPattern}" }
            require(twin.whiteLevel != null && twin.whiteLevel > 0) { "Sensor white level unavailable" }

            val ispPlan = withContext(Dispatchers.Default) { SierraRawIspPlanner.plan(capture, twin) }
            val sink = SierraIspBitmapSink(ispPlan)
            val summary = withContext(Dispatchers.Default) { reconstructor.reconstruct(capture, twin, sink) }
            var rendered = sink.bitmap()
            val rotation = SierraRawContainer.open(capture.frames[summary.referenceFrameIndex].rawFile).use { it.header.rotationDegrees }
            if (rotation != 0) {
                val rotated = Bitmap.createBitmap(
                    rendered,
                    0,
                    0,
                    rendered.width,
                    rendered.height,
                    Matrix().apply { postRotate(rotation.toFloat()) },
                    true
                )
                if (rotated !== rendered && !rendered.isRecycled) rendered.recycle()
                rendered = rotated
            }

            val comparisonJpeg = capture.frames[summary.referenceFrameIndex].jpegFile
            val computationalQuality = analyzeBounded(rendered)
            val safetyBitmap = storage.decodeOriented(comparisonJpeg, QUALITY_LONG_SIDE)
            val safetyQuality = try { PhotoQuality.analyze(safetyBitmap) } finally { if (!safetyBitmap.isRecycled) safetyBitmap.recycle() }

            val rejection = qualityRejection(summary, computationalQuality, safetyQuality)
            ComputationalRawResult(
                bitmap = rendered,
                safetyJpeg = protectedSafety,
                capture = capture,
                reconstruction = summary,
                ispPlan = ispPlan,
                computationalQuality = computationalQuality,
                safetyQuality = safetyQuality,
                acceptedOverSafetyJpeg = rejection == null,
                rejectionReason = rejection
            )
        } catch (t: Throwable) {
            capture.cleanup()
            // protectedSafety deliberately survives if the camera already produced a JPEG.
            throw t
        }
    }

    private fun analyzeBounded(bitmap: Bitmap): TechnicalQuality {
        val longest = maxOf(bitmap.width, bitmap.height)
        if (longest <= QUALITY_LONG_SIDE) return PhotoQuality.analyze(bitmap)
        val scale = QUALITY_LONG_SIDE.toFloat() / longest
        val sampled = Bitmap.createScaledBitmap(
            bitmap,
            maxOf(1, (bitmap.width * scale).toInt()),
            maxOf(1, (bitmap.height * scale).toInt()),
            true
        )
        return try { PhotoQuality.analyze(sampled) } finally { if (!sampled.isRecycled) sampled.recycle() }
    }

    private fun qualityRejection(
        summary: RawReconstructionSummary,
        computational: TechnicalQuality,
        safety: TechnicalQuality
    ): String? {
        if (summary.meanConfidence < MIN_RECONSTRUCTION_CONFIDENCE) return "insufficient_physical_evidence"
        if (computational.catastrophic) return "computational_render_catastrophic"
        if (!safety.catastrophic && computational.heuristicScore + MAX_SCORE_REGRESSION < safety.heuristicScore) {
            return "technical_quality_regressed_vs_same_capture_jpeg"
        }
        if (!safety.catastrophic && computational.brightClipFraction > safety.brightClipFraction + MAX_CLIP_REGRESSION) {
            return "highlight_clipping_regressed"
        }
        return null
    }

    companion object {
        private val SUPPORTED_CFA = setOf(CfaPattern.RGGB, CfaPattern.BGGR, CfaPattern.GRBG, CfaPattern.GBRG)
        private const val QUALITY_LONG_SIDE = 1024
        private const val MIN_RECONSTRUCTION_CONFIDENCE = 0.28
        private const val MAX_SCORE_REGRESSION = 0.10
        private const val MAX_CLIP_REGRESSION = 0.035
    }
}
