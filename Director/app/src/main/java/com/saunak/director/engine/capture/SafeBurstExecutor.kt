package com.saunak.director.engine.capture

import com.saunak.director.camera.CameraEngine
import com.saunak.director.processing.PhotoQuality
import com.saunak.director.processing.TechnicalQuality
import com.saunak.director.storage.PhotoRepository
import java.io.File
import kotlin.math.max

/**
 * Production-safe multi-frame executor built entirely on the already-certified CameraX JPEG path.
 * The first successfully captured still is written to the recoverable no-backup pending directory;
 * later burst frames are disposable cache. This path is smart frame selection, not RAW/HDR fusion.
 */
class SafeBurstExecutor(
    private val camera: CameraEngine,
    private val storage: PhotoRepository
) {
    data class CapturedFrame(
        val file: File,
        val quality: TechnicalQuality,
        val score: Double,
        val index: Int,
        val protectedReference: Boolean = false
    )

    data class Result(
        val selected: CapturedFrame,
        val frames: List<CapturedFrame>,
        val rejectedCount: Int,
        val requestedCount: Int,
        val executedCount: Int,
        val truncatedByFailure: Boolean,
        val allCatastrophic: Boolean,
        val rationale: String
    ) {
        init {
            require(frames.isNotEmpty())
            require(frames.first().protectedReference) { "First burst frame must be the recoverable reference" }
            require(selected in frames)
            require(requestedCount >= 1)
            require(executedCount == frames.size)
            require(executedCount in 1..requestedCount)
            require(rejectedCount in 0..executedCount)
            require(allCatastrophic == frames.all { it.quality.catastrophic })
        }

        val protectedReference: CapturedFrame get() = frames.first()
    }

    suspend fun execute(
        sessionId: Long,
        program: CaptureProgram?,
        maxFrames: Int = DEFAULT_MAX_FRAMES
    ): Result {
        require(maxFrames in 1..MAX_ALLOWED_FRAMES)
        val requested = program?.frames?.size?.coerceAtLeast(1) ?: 1
        val count = requested.coerceAtMost(maxFrames)
        require(storage.hasWorkingSpace(requiredWorkingBytes(count))) {
            "Insufficient working storage for Sierra burst"
        }

        val captured = ArrayList<CapturedFrame>(count)
        var truncated = false

        for (index in 0 until count) {
            val protectedReference = index == 0
            val file = if (protectedReference) {
                storage.newPendingReference(sessionId)
            } else {
                storage.newAttempt(sessionId, 100 + index)
            }

            val capture = runCatching { camera.capturePrivate(file) }
            if (capture.isFailure) {
                runCatching { file.delete() }
                if (captured.isEmpty()) throw capture.exceptionOrNull() ?: IllegalStateException("First burst frame failed")
                truncated = true
                break
            }

            // A successful camera file is evidence even if our thumbnail decoder/quality analyser fails.
            // Mark that frame as technically rejected rather than deleting a recoverable photograph.
            val quality = runCatching {
                val preview = storage.decodeOriented(file, QUALITY_DECODE_MAX_DIMENSION)
                try {
                    PhotoQuality.analyze(preview)
                } finally {
                    if (!preview.isRecycled) preview.recycle()
                }
            }.getOrElse { rejectedAnalysisQuality() }

            captured += CapturedFrame(
                file = file,
                quality = quality,
                score = frameScore(quality, index, count),
                index = index,
                protectedReference = protectedReference
            )
        }

        val valid = captured.filterNot { it.quality.catastrophic }
        val allCatastrophic = valid.isEmpty()
        val pool = if (valid.isNotEmpty()) valid else captured
        val selected = pool.maxWithOrNull(
            compareBy<CapturedFrame> { it.score }
                .thenByDescending { -kotlin.math.abs(it.index - (captured.size - 1) / 2.0) }
        ) ?: error("Sierra burst produced no frames")

        return Result(
            selected = selected,
            frames = captured,
            rejectedCount = captured.count { it.quality.catastrophic },
            requestedCount = requested,
            executedCount = captured.size,
            truncatedByFailure = truncated,
            allCatastrophic = allCatastrophic,
            rationale = when {
                allCatastrophic -> "all captured frames failed the technical quality gate; preserve an untouched fallback"
                truncated -> "burst stopped after a later capture failure; best valid earlier evidence retained"
                captured.size == 1 -> "single certified CameraX evidence frame"
                else -> "best-evidence selection across ${captured.size} serialized CameraX frames"
            }
        )
    }

    private fun rejectedAnalysisQuality(): TechnicalQuality = TechnicalQuality(
        meanLuma = 0.0,
        lumaStdDev = 0.0,
        darkClipFraction = 0.0,
        brightClipFraction = 0.0,
        edgeEnergy = 0.0,
        catastrophic = true,
        heuristicScore = 0.0
    )

    private fun frameScore(quality: TechnicalQuality, index: Int, total: Int): Double {
        if (quality.catastrophic) return 0.0
        val clippingPenalty = (quality.darkClipFraction + quality.brightClipFraction).coerceIn(0.0, 1.0)
        val detail = (quality.edgeEnergy / 14.0).coerceIn(0.0, 1.0)
        val dynamic = (quality.lumaStdDev / 50.0).coerceIn(0.0, 1.0)
        val center = (total - 1) / 2.0
        val temporalCenterBias = if (total <= 1) 1.0 else {
            (1.0 - kotlin.math.abs(index - center) / max(1.0, center)).coerceIn(0.0, 1.0)
        }
        return (
            0.46 * quality.heuristicScore +
                0.24 * detail +
                0.14 * dynamic +
                0.06 * temporalCenterBias +
                0.10 * (1.0 - clippingPenalty)
            ).coerceIn(0.0, 1.0)
    }

    private fun requiredWorkingBytes(frameCount: Int): Long =
        STORAGE_RESERVE_BYTES + frameCount.toLong() * ESTIMATED_MAX_JPEG_BYTES

    companion object {
        private const val QUALITY_DECODE_MAX_DIMENSION = 768
        private const val DEFAULT_MAX_FRAMES = 5
        private const val MAX_ALLOWED_FRAMES = 8
        private const val ESTIMATED_MAX_JPEG_BYTES = 32L * 1024L * 1024L
        private const val STORAGE_RESERVE_BYTES = 64L * 1024L * 1024L
    }
}
