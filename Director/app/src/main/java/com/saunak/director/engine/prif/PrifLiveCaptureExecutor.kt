package com.saunak.director.engine.prif

import com.saunak.director.camera.CameraEngine
import com.saunak.director.camera.ManualRawFrameSpec
import com.saunak.director.camera.RawJpegCaptureFrame
import com.saunak.director.engine.SierraEngine
import com.saunak.director.engine.capture.CaptureProgram
import com.saunak.director.engine.reconstruction.StoredRawBurstFrame
import com.saunak.director.engine.reconstruction.StoredRawWriter
import com.saunak.director.platform.MemoryPressureBus
import com.saunak.director.platform.ThermalPolicy
import java.io.File
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/** Result always retains a camera-generated JPEG safety observation from the RAW+JPEG capture. */
data class PrifLiveCaptureResult(
    val reconstruction: PrifLiveReconstruction?,
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
 * Live P-RIF shutter executor. Evidence selection happens before RAW allocation, captured RAW floats
 * are immediately spilled to canonical disk storage, and probabilistic reconstruction is tile-bounded.
 */
class PrifLiveCaptureExecutor(
    private val camera: CameraEngine,
    private val engine: SierraEngine = SierraEngine.get()
) {
    suspend fun execute(
        captureProgram: CaptureProgram,
        thermal: ThermalPolicy
    ): PrifLiveCaptureResult {
        require(camera.rawJpegCaptureSupported()) { "Active camera does not satisfy Sierra P-RIF RAW contract" }
        val motion = engine.motion.latest()
        val subject = engine.subjects.latest()
        val memoryPressure = MemoryPressureBus.normalizedPressure()
        val initial = PrifReconstructionCompiler.compile(
            captureProgram = captureProgram,
            motion = motion,
            subject = subject,
            thermalPressure = thermal.thermalPressure,
            memoryPressure = memoryPressure
        )
        val heapCap = liveRawFrameHeapCap(initial.maxFrames)
        require(heapCap >= 1) { "P-RIF RAW allocation exceeds conservative live heap budget" }
        val preflight = initial.copy(maxFrames = minOf(initial.maxFrames, heapCap, thermal.maxSierraBurstFrames))
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

        val scratch = engine.newPrifScratchSession()
        var captured: List<RawJpegCaptureFrame> = emptyList()
        try {
            captured = camera.captureRawJpegBurst(specs)
            check(captured.size == reducedProgram.frames.size)
            val capturedCount = captured.size
            val safe = selectSafetyJpeg(captured, reducedProgram).copyOf()
            val storedFrames = captured.mapIndexed { index, frame ->
                val base = StoredRawWriter.write(frame.raw, File(scratch, "frame_${index}_${frame.raw.timestampNs}.raw16"))
                val stored = if (base.metadata === frame.metadata) base else base.copy(metadata = frame.metadata)
                StoredRawBurstFrame(frameId = frame.raw.timestampNs, stored = stored)
            }

            // Remove the only collection-level strong reference to the large RAW FloatArrays before
            // tile reconstruction. The runtime may reclaim them while canonical RAW16 stays on disk.
            captured = emptyList()

            val liveProgram = PrifReconstructionCompiler.compile(
                captureProgram = reducedProgram,
                motion = motion,
                subject = subject,
                thermalPressure = thermal.thermalPressure,
                memoryPressure = MemoryPressureBus.normalizedPressure()
            )
            val outputBudget = liveOutputPixelBudget(storedFrames.first().stored.width, storedFrames.first().stored.height)
            val pipeline = PrifStoredReconstructionPipeline(
                ledger = engine.evidence,
                tileSize = liveProgram.tileSize
            )
            val reconstruction = runCatching {
                pipeline.reconstruct(
                    frames = storedFrames,
                    program = liveProgram,
                    subject = subject,
                    motionRisk = motion.conservativeRisk,
                    maxOutputPixels = outputBudget
                )
            }.getOrNull()

            if (reconstruction == null) {
                return PrifLiveCaptureResult(
                    reconstruction = null,
                    safeJpegBytes = safe,
                    capturedFrameCount = capturedCount,
                    accepted = false,
                    reason = "P-RIF reconstruction failed; preserved RAW+JPEG safety observation"
                )
            }
            if (!reconstruction.integrity.accepted) {
                reconstruction.evidenceRegionIds.forEach(engine.evidence::remove)
                reconstruction.bitmap.recycle()
                return PrifLiveCaptureResult(
                    reconstruction = null,
                    safeJpegBytes = safe,
                    capturedFrameCount = capturedCount,
                    accepted = false,
                    reason = "P-RIF sensor-consistency/integrity gate rejected reconstruction"
                )
            }
            return PrifLiveCaptureResult(
                reconstruction = reconstruction,
                safeJpegBytes = safe,
                capturedFrameCount = capturedCount,
                accepted = true,
                reason = "P-RIF physical reconstruction accepted"
            )
        } finally {
            captured = emptyList()
            runCatching { scratch.deleteRecursively() }
        }
    }

    /**
     * CameraX currently materializes the full RAW FloatArrays before P-RIF can spill them. Refuse an
     * unsafe burst before shutter instead of relying on an OOM recovery that Android cannot guarantee.
     */
    private fun liveRawFrameHeapCap(requested: Int): Int {
        val cameraId = camera.activeCameraId()
        val module = engine.deviceTwin().getOrNull()?.let { twin -> cameraId?.let(twin::camera) }
        val pixels = module?.let { it.activeArrayWidth.toLong() * it.activeArrayHeight.toLong() }
            ?: DEFAULT_SENSOR_PIXELS
        val runtime = Runtime.getRuntime()
        val used = runtime.totalMemory() - runtime.freeMemory()
        val headroom = (runtime.maxMemory() - used).coerceAtLeast(0L)
        // RAW FloatArray (4 B/px) + concurrent JPEG/native/callback overhead. This is deliberately
        // conservative because one unsupported giant allocation can terminate the process.
        val estimatedPerFrame = max(MIN_FRAME_BYTES, (pixels * ESTIMATED_CAPTURE_BYTES_PER_PIXEL).toLong())
        val allocationBudget = (headroom * LIVE_CAPTURE_HEAP_FRACTION).toLong()
        if (allocationBudget < estimatedPerFrame) return 0
        return floor(allocationBudget.toDouble() / estimatedPerFrame.toDouble()).toInt()
            .coerceIn(1, requested.coerceAtLeast(1))
    }

    private fun liveOutputPixelBudget(width: Int, height: Int): Long {
        val runtime = Runtime.getRuntime()
        val used = runtime.totalMemory() - runtime.freeMemory()
        val headroom = (runtime.maxMemory() - used).coerceAtLeast(0L)
        // Final ARGB bitmap plus haloed per-frame tiles and temporary RGB tile. Keep a strong reserve
        // for CameraX, UI, ORT and GC fragmentation.
        val byHeap = (headroom * LIVE_OUTPUT_HEAP_FRACTION / OUTPUT_BYTES_PER_PIXEL_ESTIMATE).toLong()
        return minOf(width.toLong() * height.toLong(), byHeap.coerceAtLeast(MIN_OUTPUT_PIXELS))
    }

    private fun selectSafetyJpeg(
        captured: List<RawJpegCaptureFrame>,
        program: CaptureProgram
    ): ByteArray {
        val index = program.frames.indices.minByOrNull { kotlin.math.abs(program.frames[it].evOffset) } ?: 0
        return captured[index.coerceIn(captured.indices)].jpegBytes
    }

    companion object {
        private const val DEFAULT_SENSOR_PIXELS = 12_000_000L
        private const val ESTIMATED_CAPTURE_BYTES_PER_PIXEL = 6.5
        private const val MIN_FRAME_BYTES = 32L * 1024L * 1024L
        private const val LIVE_CAPTURE_HEAP_FRACTION = 0.38
        private const val LIVE_OUTPUT_HEAP_FRACTION = 0.48
        private const val OUTPUT_BYTES_PER_PIXEL_ESTIMATE = 5.5
        private const val MIN_OUTPUT_PIXELS = 2_000_000L
    }
}
