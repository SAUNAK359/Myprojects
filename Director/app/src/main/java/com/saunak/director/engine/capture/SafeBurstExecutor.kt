package com.saunak.director.engine.capture

import com.saunak.director.camera.CameraEngine
import com.saunak.director.processing.PhotoQuality
import com.saunak.director.processing.TechnicalQuality
import com.saunak.director.storage.PhotoRepository
import java.io.File
import kotlin.math.max

/**
 * Production-safe multi-frame executor built entirely on the already-certified CameraX JPEG path.
 * It is the universal fallback underneath optional RAW execution: no Camera2 session mutation,
 * no vendor-specific assumptions, and every frame remains independently decodable evidence.
 */
class SafeBurstExecutor(
    private val camera: CameraEngine,
    private val storage: PhotoRepository
) {
    data class CapturedFrame(
        val file: File,
        val quality: TechnicalQuality,
        val score: Double,
        val index: Int
    )

    data class Result(
        val selected: CapturedFrame,
        val frames: List<CapturedFrame>,
        val rejectedCount: Int,
        val requestedCount: Int,
        val executedCount: Int,
        val rationale: String
    ) {
        init {
            require(frames.isNotEmpty())
            require(selected in frames)
            require(requestedCount >= 1)
            require(executedCount == frames.size)
            require(rejectedCount in 0..executedCount)
        }
    }

    suspend fun execute(
        sessionId: Long,
        program: CaptureProgram?,
        maxFrames: Int = DEFAULT_MAX_FRAMES
    ): Result {
        require(maxFrames in 1..MAX_ALLOWED_FRAMES)
        val requested = program?.frames?.size?.coerceAtLeast(1) ?: 1
        val count = requested.coerceAtMost(maxFrames)
        val captured = ArrayList<CapturedFrame>(count)

        repeat(count) { index ->
            // Each still is captured through the existing CameraX path. The first frame has already
            // benefited from settled 3A; subsequent frames are naturally serialized by CameraX.
            val file = storage.newAttempt(sessionId, 100 + index)
            camera.capturePrivate(file)
            val preview = storage.decodeOriented(file, QUALITY_DECODE_MAX_DIMENSION)
            val quality = try {
                PhotoQuality.analyze(preview)
            } finally {
                if (!preview.isRecycled) preview.recycle()
            }
            captured += CapturedFrame(
                file = file,
                quality = quality,
                score = frameScore(quality, index, count),
                index = index
            )
        }

        val valid = captured.filterNot { it.quality.catastrophic }
        val pool = if (valid.isNotEmpty()) valid else captured
        val selected = pool.maxWithOrNull(
            compareBy<CapturedFrame> { it.score }
                .thenByDescending { -kotlin.math.abs(it.index - (count - 1) / 2.0) }
        ) ?: error("Sierra burst produced no frames")

        return Result(
            selected = selected,
            frames = captured,
            rejectedCount = captured.count { it.quality.catastrophic },
            requestedCount = requested,
            executedCount = captured.size,
            rationale = if (count == 1) {
                "single certified CameraX evidence frame"
            } else {
                "best-evidence selection across $count serialized CameraX frames"
            }
        )
    }

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

    companion object {
        private const val QUALITY_DECODE_MAX_DIMENSION = 768
        private const val DEFAULT_MAX_FRAMES = 5
        private const val MAX_ALLOWED_FRAMES = 8
    }
}
