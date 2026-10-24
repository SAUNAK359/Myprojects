package com.saunak.director.engine.experts

import android.graphics.Bitmap
import com.saunak.director.engine.SierraEngine
import com.saunak.director.engine.device.ComputeBackend
import com.saunak.director.engine.models.InstalledModelArtifact
import com.saunak.director.engine.models.SierraShippingManifests
import com.saunak.director.engine.runtime.InferenceRequest
import com.saunak.director.engine.runtime.ProfileKey
import com.saunak.director.engine.runtime.RuntimePressure
import com.saunak.director.engine.runtime.WorkPriority
import java.util.Arrays
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/**
 * Portable Depth Anything V2 Small expert. Output is RELATIVE INVERSE depth: larger values are
 * closer. It must never be interpreted as metres; metric depth requires a separately calibrated
 * expert/hardware source.
 */
class DepthAnythingV2Expert(
    private val engine: SierraEngine = SierraEngine.get(),
    private val longSide: Int = 518
) {
    init { require(longSide >= PATCH && longSide % PATCH == 0) }

    data class RelativeDepthMap(
        val width: Int,
        val height: Int,
        val inverseDepth: FloatArray,
        val confidence: Double,
        val sourceWidth: Int,
        val sourceHeight: Int,
        val metric: Boolean = false
    ) {
        init {
            require(width > 0 && height > 0)
            require(inverseDepth.size == width * height)
            require(inverseDepth.all { it.isFinite() && it in 0f..1f })
            require(confidence in 0.0..1.0)
            require(sourceWidth > 0 && sourceHeight > 0)
            require(!metric) { "Depth Anything V2 Small is relative depth, not metric depth" }
        }
    }

    fun isProvisioned(): Boolean =
        engine.modelArtifacts.findVerified(SierraShippingManifests.depthAnythingV2Small) != null

    suspend fun profileIfProvisioned(): Result<Unit> = withContext(Dispatchers.Default) {
        runCatching {
            val artifact = engine.modelArtifacts.findVerified(SierraShippingManifests.depthAnythingV2Small)
                ?: error("Depth Anything V2 Small pack is not provisioned")
            ensureMeasuredProfiles(artifact)
        }
    }

    suspend fun estimate(source: Bitmap): Result<RelativeDepthMap> = withContext(Dispatchers.Default) {
        runCatching {
            require(!source.isRecycled && source.width > 0 && source.height > 0)
            val artifact = engine.modelArtifacts.findVerified(SierraShippingManifests.depthAnythingV2Small)
                ?: error("Depth Anything V2 Small pack is not provisioned")
            val (width, height) = chooseInputSize(source.width, source.height, longSide)
            val tensor = preprocess(source, width, height)
            val raw = runWithMeasuredRouting(artifact, tensor, width, height)
            require(raw.size == width * height) {
                "Depth output has ${raw.size} values; expected ${width * height} for ${width}x$height"
            }
            normalize(raw, width, height, source.width, source.height)
        }
    }

    internal fun chooseInputSize(width: Int, height: Int, targetLongSide: Int = longSide): Pair<Int, Int> =
        contractInputSize(width, height, targetLongSide)

    private fun preprocess(source: Bitmap, width: Int, height: Int): FloatArray {
        val scaled = if (source.width == width && source.height == height) source
        else Bitmap.createScaledBitmap(source, width, height, true)
        try {
            val pixels = IntArray(width * height)
            scaled.getPixels(pixels, 0, width, 0, 0, width, height)
            val plane = pixels.size
            val out = FloatArray(plane * 3)
            for (i in pixels.indices) {
                val p = pixels[i]
                val r = ((p ushr 16) and 0xff) / 255f
                val g = ((p ushr 8) and 0xff) / 255f
                val b = (p and 0xff) / 255f
                out[i] = (r - MEAN_R) / STD_R
                out[plane + i] = (g - MEAN_G) / STD_G
                out[plane * 2 + i] = (b - MEAN_B) / STD_B
            }
            return out
        } finally {
            if (scaled !== source) scaled.recycle()
        }
    }

    private fun runWithMeasuredRouting(
        artifact: InstalledModelArtifact,
        input: FloatArray,
        width: Int,
        height: Int
    ): FloatArray {
        val shape = longArrayOf(1, 3, height.toLong(), width.toLong())
        val signature = inputSignature(width, height)
        val candidates = engine.measuredProfiles(artifact.manifest.id, signature)
        val decision = engine.hypervisor.choose(
            request = InferenceRequest(
                requestId = "depth-${System.nanoTime()}",
                modelId = artifact.manifest.id,
                inputSignature = signature,
                valueOfInformation = 0.78,
                priority = WorkPriority.PERCEPTION,
                deadlineMs = DEPTH_DEADLINE_MS,
                minimumGraphCoverage = artifact.manifest.minimumGraphCoverage,
                maximumTransferBoundaries = artifact.manifest.maxTransferBoundaries
            ),
            candidates = candidates,
            pressure = RuntimePressure()
        )

        decision?.let {
            val attempt = runCatching { run(artifact, it.profile.backend, input, shape) }
            if (attempt.isSuccess) return attempt.getOrThrow()
            engine.runtimeProfiles.invalidate(
                ProfileKey(engine.profileFingerprint(), artifact.manifest.id, signature, it.profile.backend)
            )
            engine.onnx.unload(artifact.manifest.id)
        }

        for (backend in BACKEND_PROBE_ORDER) {
            val attempt = runCatching { run(artifact, backend, input, shape) }
            if (attempt.isSuccess) return attempt.getOrThrow()
        }
        error("Depth Anything V2 Small has no usable Sierra ONNX backend on this device")
    }

    private suspend fun ensureMeasuredProfiles(artifact: InstalledModelArtifact) {
        val width = longSide
        val height = longSide
        val signature = inputSignature(width, height)
        val existing = engine.measuredProfiles(artifact.manifest.id, signature).map { it.backend }.toSet()
        val input = FloatArray(width * height * 3)
        val shape = longArrayOf(1, 3, height.toLong(), width.toLong())
        for (backend in BACKEND_PROBE_ORDER) {
            if (backend in existing || backend !in artifact.manifest.supportedBackends) continue
            runCatching {
                engine.backendProfiler.profile(
                    deviceFingerprint = engine.profileFingerprint(),
                    modelId = artifact.manifest.id,
                    inputSignature = signature,
                    backend = backend,
                    graphCoverage = 1.0,
                    transferBoundaries = 0,
                    warmups = 1,
                    repeats = 3
                ) { run(artifact, backend, input, shape) }
            }
        }
    }

    private fun run(
        artifact: InstalledModelArtifact,
        backend: ComputeBackend,
        input: FloatArray,
        shape: LongArray
    ): FloatArray = engine.onnx.runFloatTensor(
        artifact = artifact,
        backend = backend,
        inputName = INPUT,
        values = input,
        shape = shape,
        requestedOutputs = setOf(OUTPUT)
    ).getValue(OUTPUT)

    private fun normalize(
        raw: FloatArray,
        width: Int,
        height: Int,
        sourceWidth: Int,
        sourceHeight: Int
    ): RelativeDepthMap {
        val finite = FloatArray(raw.size)
        var finiteCount = 0
        for (value in raw) if (value.isFinite()) finite[finiteCount++] = value
        require(finiteCount > 0) { "Depth model produced no finite values" }
        Arrays.sort(finite, 0, finiteCount)

        val finiteFraction = finiteCount.toDouble() / raw.size.toDouble()
        val lo = percentile(finite, finiteCount, 0.02)
        val hi = percentile(finite, finiteCount, 0.98)
        val range = (hi - lo).coerceAtLeast(1e-6f)
        val normalized = FloatArray(raw.size) { index ->
            val value = raw[index]
            if (!value.isFinite()) 0.5f else ((value - lo) / range).coerceIn(0f, 1f)
        }
        val dynamicConfidence = ((hi - lo).toDouble() /
            (max(kotlin.math.abs(hi.toDouble()), kotlin.math.abs(lo.toDouble())) + 1e-6)).coerceIn(0.0, 1.0)
        return RelativeDepthMap(
            width = width,
            height = height,
            inverseDepth = normalized,
            confidence = (finiteFraction * (0.65 + 0.35 * dynamicConfidence)).coerceIn(0.0, 1.0),
            sourceWidth = sourceWidth,
            sourceHeight = sourceHeight
        )
    }

    private fun percentile(sorted: FloatArray, count: Int, fraction: Double): Float {
        require(count in 1..sorted.size)
        val index = ((count - 1) * fraction).toInt().coerceIn(0, count - 1)
        return sorted[index]
    }

    private fun inputSignature(width: Int, height: Int): String = "rgb-f32-nchw:1x3x${height}x$width"

    companion object {
        private const val PATCH = 14
        private const val INPUT = "pixel_values"
        private const val OUTPUT = "predicted_depth"
        private const val DEPTH_DEADLINE_MS = 260.0
        private const val MEAN_R = 0.485f
        private const val MEAN_G = 0.456f
        private const val MEAN_B = 0.406f
        private const val STD_R = 0.229f
        private const val STD_G = 0.224f
        private const val STD_B = 0.225f
        private val BACKEND_PROBE_ORDER = listOf(ComputeBackend.NNAPI, ComputeBackend.CPU_XNNPACK)

        internal fun contractInputSize(width: Int, height: Int, targetLongSide: Int = 518): Pair<Int, Int> {
            require(width > 0 && height > 0)
            require(targetLongSide >= PATCH && targetLongSide % PATCH == 0)
            fun align14(value: Double): Int = (ceil(value / PATCH).toInt().coerceAtLeast(1) * PATCH)
            return if (width >= height) {
                targetLongSide to min(targetLongSide, align14(targetLongSide.toDouble() * height / width))
            } else {
                min(targetLongSide, align14(targetLongSide.toDouble() * width / height)) to targetLongSide
            }
        }
    }
}
