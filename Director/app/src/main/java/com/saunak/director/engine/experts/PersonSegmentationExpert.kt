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
import com.saunak.director.engine.subject.NormalizedPoint
import com.saunak.director.engine.subject.NormalizedRect
import com.saunak.director.engine.subject.SubjectMaskState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Mobile person-mask expert backed by the pinned MediaPipe Selfie Segmentation web ONNX export.
 * The model accepts raw RGB UINT8 NHWC pixels. Connected-component selection is anchored to the
 * current subject track so a second person does not silently replace the photographic subject.
 */
class PersonSegmentationExpert(
    private val engine: SierraEngine = SierraEngine.get(),
    private val inputSize: Int = 256
) {
    init { require(inputSize in 128..512) }

    fun isProvisioned(): Boolean =
        engine.modelArtifacts.findVerified(SierraShippingManifests.personSegmentation) != null

    suspend fun segment(source: Bitmap, subjectHint: NormalizedRect?): Result<SubjectMaskState> =
        withContext(Dispatchers.Default) {
            runCatching {
                require(!source.isRecycled && source.width > 0 && source.height > 0)
                val artifact = engine.modelArtifacts.findVerified(SierraShippingManifests.personSegmentation)
                    ?: error("Person segmentation pack is not provisioned")
                val pixels = preprocess(source)
                val output = runWithMeasuredRouting(artifact, pixels)
                require(output.size == inputSize * inputSize) {
                    "Segmentation output has ${output.size} values; expected ${inputSize * inputSize}"
                }
                decode(output, subjectHint)
            }
        }

    suspend fun profileIfProvisioned(): Result<Unit> = withContext(Dispatchers.Default) {
        runCatching {
            val artifact = engine.modelArtifacts.findVerified(SierraShippingManifests.personSegmentation)
                ?: error("Person segmentation pack is not provisioned")
            ensureMeasuredProfiles(artifact)
        }
    }

    private fun preprocess(source: Bitmap): ByteArray {
        val scaled = if (source.width == inputSize && source.height == inputSize) source
        else Bitmap.createScaledBitmap(source, inputSize, inputSize, true)
        try {
            val argb = IntArray(inputSize * inputSize)
            scaled.getPixels(argb, 0, inputSize, 0, 0, inputSize, inputSize)
            val out = ByteArray(argb.size * 3)
            var o = 0
            for (p in argb) {
                out[o++] = ((p ushr 16) and 0xff).toByte()
                out[o++] = ((p ushr 8) and 0xff).toByte()
                out[o++] = (p and 0xff).toByte()
            }
            return out
        } finally {
            if (scaled !== source) scaled.recycle()
        }
    }

    private fun runWithMeasuredRouting(artifact: InstalledModelArtifact, input: ByteArray): FloatArray {
        val signature = inputSignature()
        val candidates = engine.measuredProfiles(artifact.manifest.id, signature)
        val decision = engine.hypervisor.choose(
            request = InferenceRequest(
                requestId = "person-mask-${System.nanoTime()}",
                modelId = artifact.manifest.id,
                inputSignature = signature,
                valueOfInformation = 0.82,
                priority = WorkPriority.HUMAN,
                deadlineMs = SEGMENTATION_DEADLINE_MS,
                minimumGraphCoverage = artifact.manifest.minimumGraphCoverage,
                maximumTransferBoundaries = artifact.manifest.maxTransferBoundaries
            ),
            candidates = candidates,
            pressure = RuntimePressure()
        )
        decision?.let {
            val attempt = runCatching { run(artifact, it.profile.backend, input) }
            if (attempt.isSuccess) return attempt.getOrThrow()
            engine.runtimeProfiles.invalidate(ProfileKey(engine.profileFingerprint(), artifact.manifest.id, signature, it.profile.backend))
            engine.onnx.unload(artifact.manifest.id)
        }
        for (backend in BACKEND_PROBE_ORDER) {
            val attempt = runCatching { run(artifact, backend, input) }
            if (attempt.isSuccess) return attempt.getOrThrow()
        }
        error("Person segmentation has no usable Sierra ONNX backend")
    }

    private fun run(artifact: InstalledModelArtifact, backend: ComputeBackend, input: ByteArray): FloatArray =
        engine.onnx.runUInt8Tensor(
            artifact = artifact,
            backend = backend,
            inputName = INPUT,
            values = input,
            shape = longArrayOf(1, inputSize.toLong(), inputSize.toLong(), 3),
            requestedOutputs = setOf(OUTPUT)
        ).getValue(OUTPUT)

    private suspend fun ensureMeasuredProfiles(artifact: InstalledModelArtifact) {
        val signature = inputSignature()
        val existing = engine.measuredProfiles(artifact.manifest.id, signature).map { it.backend }.toSet()
        val input = ByteArray(inputSize * inputSize * 3)
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
                ) { run(artifact, backend, input) }
            }
        }
    }

    internal fun decode(raw: FloatArray, subjectHint: NormalizedRect?): SubjectMaskState {
        require(raw.size == inputSize * inputSize)
        val probabilities = FloatArray(raw.size) { raw[it].coerceIn(0f, 1f) }
        val selected = selectComponent(probabilities, subjectHint)
        require(selected.indices.isNotEmpty()) { "No credible person component in segmentation output" }

        val selectedMask = FloatArray(probabilities.size)
        var minX = inputSize
        var minY = inputSize
        var maxX = -1
        var maxY = -1
        var weightedX = 0.0
        var weightedY = 0.0
        var weight = 0.0
        var confidenceSum = 0.0
        for (index in selected.indices) {
            val p = probabilities[index]
            selectedMask[index] = p
            val x = index % inputSize
            val y = index / inputSize
            minX = minOf(minX, x); maxX = maxOf(maxX, x)
            minY = minOf(minY, y); maxY = maxOf(maxY, y)
            weightedX += x * p
            weightedY += y * p
            weight += p
            confidenceSum += p
        }
        require(maxX >= minX && maxY >= minY && weight > 0.0)
        val bounds = NormalizedRect(
            minX.toDouble() / inputSize,
            minY.toDouble() / inputSize,
            (maxX + 1).toDouble() / inputSize,
            (maxY + 1).toDouble() / inputSize
        ).clamped()
        val centroid = NormalizedPoint(
            (weightedX / weight) / inputSize,
            (weightedY / weight) / inputSize
        ).clamped()
        val coverage = selected.indices.size.toDouble() / probabilities.size.toDouble()
        val meanForeground = confidenceSum / selected.indices.size.toDouble()
        val sizeSupport = (coverage / 0.06).coerceIn(0.0, 1.0)
        val confidence = (meanForeground * (0.72 + 0.28 * sizeSupport) * selected.hintAffinity).coerceIn(0.0, 1.0)
        return SubjectMaskState(
            width = inputSize,
            height = inputSize,
            probabilities = selectedMask,
            bounds = bounds,
            centroid = centroid,
            coverage = coverage,
            confidence = confidence,
            source = "mediapipe_selfie_segmentation_web"
        )
    }

    private data class Component(val indices: IntArray, val hintAffinity: Double)

    private fun selectComponent(probabilities: FloatArray, hint: NormalizedRect?): Component {
        val visited = BooleanArray(probabilities.size)
        val queue = IntArray(probabilities.size)
        var best: IntArray = IntArray(0)
        var bestScore = Double.NEGATIVE_INFINITY
        var bestAffinity = 1.0

        for (seed in probabilities.indices) {
            if (visited[seed] || probabilities[seed] < MASK_THRESHOLD) continue
            var head = 0
            var tail = 0
            queue[tail++] = seed
            visited[seed] = true
            val members = IntArray(probabilities.size)
            var count = 0
            var probabilitySum = 0.0
            var hintHits = 0
            while (head < tail) {
                val index = queue[head++]
                members[count++] = index
                probabilitySum += probabilities[index]
                val x = index % inputSize
                val y = index / inputSize
                if (hint != null) {
                    val nx = (x + 0.5) / inputSize.toDouble()
                    val ny = (y + 0.5) / inputSize.toDouble()
                    if (nx in hint.left..hint.right && ny in hint.top..hint.bottom) hintHits++
                }
                fun enqueue(nx: Int, ny: Int) {
                    if (nx !in 0 until inputSize || ny !in 0 until inputSize) return
                    val ni = ny * inputSize + nx
                    if (!visited[ni] && probabilities[ni] >= MASK_THRESHOLD) {
                        visited[ni] = true
                        queue[tail++] = ni
                    }
                }
                enqueue(x - 1, y); enqueue(x + 1, y); enqueue(x, y - 1); enqueue(x, y + 1)
            }
            if (count < MIN_COMPONENT_PIXELS) continue
            val mean = probabilitySum / count
            val affinity = if (hint == null) 1.0 else (hintHits.toDouble() / count.toDouble()).coerceIn(0.0, 1.0)
            val sizeScore = (count.toDouble() / probabilities.size.toDouble()).coerceAtMost(0.5)
            val score = mean + 1.35 * affinity + 0.25 * sizeScore
            if (score > bestScore) {
                bestScore = score
                best = members.copyOf(count)
                bestAffinity = if (hint == null) 1.0 else (0.55 + 0.45 * affinity).coerceIn(0.55, 1.0)
            }
        }
        return Component(best, bestAffinity)
    }

    private fun inputSignature(): String = "rgb-u8-nhwc:1x${inputSize}x${inputSize}x3"

    companion object {
        private const val INPUT = "pixel_values"
        private const val OUTPUT = "alphas"
        private const val MASK_THRESHOLD = 0.18f
        private const val MIN_COMPONENT_PIXELS = 48
        private const val SEGMENTATION_DEADLINE_MS = 110.0
        private val BACKEND_PROBE_ORDER = listOf(ComputeBackend.NNAPI, ComputeBackend.CPU_XNNPACK)
    }
}
