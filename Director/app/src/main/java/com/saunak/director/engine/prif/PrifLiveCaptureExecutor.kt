package com.saunak.director.engine.prif

import com.saunak.director.camera.CameraEngine
import com.saunak.director.camera.ManualRawFrameSpec
import com.saunak.director.engine.SierraEngine
import com.saunak.director.engine.capture.CaptureProgram
import com.saunak.director.platform.MemoryPressureBus
import com.saunak.director.platform.ThermalPolicy

/** Result always retains a camera-generated JPEG safety observation from the RAW+JPEG capture. */
data class PrifLiveCaptureResult(
    val reconstruction: PrifReconstructionResult?,
    val safeJpegBytes: ByteArray,
    val capturedFrameCount: Int,
    val accepted: Boolean,
    val reason: String
) {
    init {
        require(safeJpegBytes.isNotEmpty())
        require(capturedFrameCount > 0)
        require(accepted == (reconstruction != null))
    }
}

/**
 * Live P-RIF shutter executor. Evidence selection happens BEFORE RAW allocation. Reconstruction is
 * promoted only when forward sensor consistency + photo integrity pass; otherwise the camera's own
 * timestamp-matched JPEG observation is returned as the safety output.
 */
class PrifLiveCaptureExecutor(
    private val camera: CameraEngine,
    private val engine: SierraEngine = SierraEngine.get(),
    private val pipeline: PrifReconstructionPipeline = PrifReconstructionPipeline()
) {
    suspend fun execute(
        captureProgram: CaptureProgram,
        thermal: ThermalPolicy
    ): PrifLiveCaptureResult {
        require(camera.rawJpegCaptureSupported()) { "Active camera does not satisfy Sierra P-RIF RAW contract" }
        val motion = engine.motion.latest()
        val subject = engine.subjects.latest()
        val memoryPressure = MemoryPressureBus.normalizedPressure()
        val preflight = PrifReconstructionCompiler.compile(
            captureProgram = captureProgram,
            motion = motion,
            subject = subject,
            thermalPressure = thermal.thermalPressure,
            memoryPressure = memoryPressure
        )
        val indices = PrifReconstructionCompiler.selectCaptureIndices(preflight)
        val reducedFrames = indices.map(captureProgram.frames::get)
        val reducedProgram = captureProgram.copy(
            frames = reducedFrames,
            hdrFusion = captureProgram.hdrFusion && reducedFrames.size >= 2 &&
                reducedFrames.map { it.evOffset }.distinct().size > 1,
            burstSuperResolution = captureProgram.burstSuperResolution && reducedFrames.size >= 3,
            expectedCaptureDurationMs = reducedFrames.sumOf {
                maxOf(it.exposureTimeNs / 1_000_000.0, 8.0)
            } + 18.0 * reducedFrames.size,
            rationale = captureProgram.rationale + "; P-RIF acquired ${reducedFrames.size}/${captureProgram.frames.size} evidence frames"
        )
        val specs = reducedFrames.map { frame ->
            ManualRawFrameSpec(
                exposureTimeNs = frame.exposureTimeNs,
                sensitivityIso = frame.sensitivityIso,
                frameDurationNs = frame.exposureTimeNs
            )
        }
        val captured = camera.captureRawJpegBurst(specs)
        check(captured.size == reducedProgram.frames.size)
        val safe = selectSafetyJpeg(captured, reducedProgram)

        val reconstruction = runCatching {
            pipeline.reconstruct(
                captured = captured,
                captureProgram = reducedProgram,
                motion = motion,
                subject = subject,
                thermalPressure = thermal.thermalPressure,
                memoryPressure = memoryPressure
            )
        }.getOrNull()

        if (reconstruction == null) {
            return PrifLiveCaptureResult(null, safe, captured.size, false, "P-RIF reconstruction failed; preserved RAW+JPEG safety observation")
        }
        if (!reconstruction.integrity.accepted) {
            reconstruction.bitmap.recycle()
            return PrifLiveCaptureResult(
                null,
                safe,
                captured.size,
                false,
                "P-RIF rejected by sensor consistency/integrity gate (${String.format("%.2f", reconstruction.integrity.score)})"
            )
        }
        return PrifLiveCaptureResult(
            reconstruction = reconstruction,
            safeJpegBytes = safe,
            capturedFrameCount = captured.size,
            accepted = true,
            reason = "P-RIF physical reconstruction accepted (${String.format("%.2f", reconstruction.integrity.score)})"
        )
    }

    private fun selectSafetyJpeg(
        captured: List<com.saunak.director.camera.RawJpegCaptureFrame>,
        program: CaptureProgram
    ): ByteArray {
        val index = program.frames.indices.minByOrNull { kotlin.math.abs(program.frames[it].evOffset) } ?: 0
        return captured[index.coerceIn(captured.indices)].jpegBytes
    }
}
