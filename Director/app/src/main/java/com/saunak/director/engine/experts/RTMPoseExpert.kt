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
import com.saunak.director.engine.subject.PoseKeypoint
import com.saunak.director.engine.subject.SubjectPoseState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.max

/**
 * Top-down COCO-17 RTMPose-S expert. The subject hint is expanded to the model aspect ratio before
 * resize, then SimCC coordinates are decoded and mapped back into the oriented source frame.
 */
class RTMPoseExpert(
    private val engine: SierraEngine = SierraEngine.get()
) {
    fun isProvisioned(): Boolean =
        engine.modelArtifacts.findVerified(SierraShippingManifests.rtmposeSmall) != null

    suspend fun estimate(source: Bitmap, subjectHint: NormalizedRect): Result<SubjectPoseState> =
        withContext(Dispatchers.Default) {
            runCatching {
                require(!source.isRecycled && source.width > 0 && source.height > 0)
                val artifact = engine.modelArtifacts.findVerified(SierraShippingManifests.rtmposeSmall)
                    ?: error("RTMPose-S pack is not provisioned")
                val crop = buildCrop(subjectHint, source.width, source.height)
                val tensor = preprocess(source, crop)
                val outputs = runWithMeasuredRouting(artifact, tensor)
                decode(
                    simccX = outputs.getValue(OUTPUT_X),
                    simccY = outputs.getValue(OUTPUT_Y),
                    crop = crop,
                    sourceWidth = source.width,
                    sourceHeight = source.height
                )
            }
        }

    suspend fun profileIfProvisioned(): Result<Unit> = withContext(Dispatchers.Default) {
        runCatching {
            val artifact = engine.modelArtifacts.findVerified(SierraShippingManifests.rtmposeSmall)
                ?: error("RTMPose-S pack is not provisioned")
            ensureMeasuredProfiles(artifact)
        }
    }

    private data class PixelCrop(val left: Int, val top: Int, val right: Int, val bottom: Int) {
        val width: Int get() = right - left
        val height: Int get() = bottom - top
    }

    private fun buildCrop(subject: NormalizedRect, sourceWidth: Int, sourceHeight: Int): PixelCrop {
        val bounded = subject.clamped()
        var cx = bounded.center.x * sourceWidth
        var cy = bounded.center.y * sourceHeight
        var width = max(2.0, bounded.width * sourceWidth * BBOX_SCALE)
        var height = max(2.0, bounded.height * sourceHeight * BBOX_SCALE)
        val targetAspect = INPUT_WIDTH.toDouble() / INPUT_HEIGHT.toDouble()
        if (width / height > targetAspect) height = width / targetAspect else width = height * targetAspect

        val left = (cx - width * 0.5).toInt().coerceAtLeast(0)
        val top = (cy - height * 0.5).toInt().coerceAtLeast(0)
        val right = (cx + width * 0.5).toInt().coerceAtMost(sourceWidth)
        val bottom = (cy + height * 0.5).toInt().coerceAtMost(sourceHeight)
        require(right - left >= 2 && bottom - top >= 2)
        return PixelCrop(left, top, right, bottom)
    }

    private fun preprocess(source: Bitmap, crop: PixelCrop): FloatArray {
        val cropped = Bitmap.createBitmap(source, crop.left, crop.top, crop.width, crop.height)
        val scaled = Bitmap.createScaledBitmap(cropped, INPUT_WIDTH, INPUT_HEIGHT, true)
        try {
            val pixels = IntArray(INPUT_WIDTH * INPUT_HEIGHT)
            scaled.getPixels(pixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT)
            val plane = pixels.size
            val output = FloatArray(plane * 3)
            for (i in pixels.indices) {
                val p = pixels[i]
                val r = ((p ushr 16) and 0xff).toFloat()
                val g = ((p ushr 8) and 0xff).toFloat()
                val b = (p and 0xff).toFloat()
                output[i] = (r - MEAN_R) / STD_R
                output[plane + i] = (g - MEAN_G) / STD_G
                output[plane * 2 + i] = (b - MEAN_B) / STD_B
            }
            return output
        } finally {
            if (scaled !== cropped && !scaled.isRecycled) scaled.recycle()
            if (!cropped.isRecycled) cropped.recycle()
        }
    }

    private fun runWithMeasuredRouting(
        artifact: InstalledModelArtifact,
        input: FloatArray
    ): Map<String, FloatArray> {
        val signature = inputSignature()
        val candidates = engine.measuredProfiles(artifact.manifest.id, signature)
        val decision = engine.hypervisor.choose(
            request = InferenceRequest(
                requestId = "rtmpose-${System.nanoTime()}",
                modelId = artifact.manifest.id,
                inputSignature = signature,
                valueOfInformation = 0.88,
                priority = WorkPriority.HUMAN,
                deadlineMs = POSE_DEADLINE_MS,
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
        error("RTMPose-S has no usable Sierra ONNX backend")
    }

    private fun run(
        artifact: InstalledModelArtifact,
        backend: ComputeBackend,
        input: FloatArray
    ): Map<String, FloatArray> = engine.onnx.runFloatTensor(
        artifact = artifact,
        backend = backend,
        inputName = INPUT,
        values = input,
        shape = longArrayOf(1, 3, INPUT_HEIGHT.toLong(), INPUT_WIDTH.toLong()),
        requestedOutputs = setOf(OUTPUT_X, OUTPUT_Y)
    )

    private suspend fun ensureMeasuredProfiles(artifact: InstalledModelArtifact) {
        val signature = inputSignature()
        val existing = engine.measuredProfiles(artifact.manifest.id, signature).map { it.backend }.toSet()
        val tensor = FloatArray(INPUT_WIDTH * INPUT_HEIGHT * 3)
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
                ) { run(artifact, backend, tensor) }
            }
        }
    }

    internal fun decode(
        simccX: FloatArray,
        simccY: FloatArray,
        crop: PixelCrop,
        sourceWidth: Int,
        sourceHeight: Int
    ): SubjectPoseState {
        require(simccX.size == KEYPOINT_COUNT * SIMCC_X_SIZE) {
            "simcc_x contains ${simccX.size} values; expected ${KEYPOINT_COUNT * SIMCC_X_SIZE}"
        }
        require(simccY.size == KEYPOINT_COUNT * SIMCC_Y_SIZE) {
            "simcc_y contains ${simccY.size} values; expected ${KEYPOINT_COUNT * SIMCC_Y_SIZE}"
        }
        val points = ArrayList<PoseKeypoint>(KEYPOINT_COUNT)
        for (k in 0 until KEYPOINT_COUNT) {
            val xBase = k * SIMCC_X_SIZE
            val yBase = k * SIMCC_Y_SIZE
            val xIndex = argmax(simccX, xBase, SIMCC_X_SIZE)
            val yIndex = argmax(simccY, yBase, SIMCC_Y_SIZE)
            val xScore = simccX[xBase + xIndex].toDouble().coerceIn(0.0, 1.0)
            val yScore = simccY[yBase + yIndex].toDouble().coerceIn(0.0, 1.0)
            val score = minOf(xScore, yScore)
            val modelX = xIndex / SIMCC_SPLIT_RATIO
            val modelY = yIndex / SIMCC_SPLIT_RATIO
            val sourceX = crop.left + (modelX / INPUT_WIDTH.toDouble()) * crop.width
            val sourceY = crop.top + (modelY / INPUT_HEIGHT.toDouble()) * crop.height
            points += PoseKeypoint(
                index = k,
                position = NormalizedPoint(sourceX / sourceWidth, sourceY / sourceHeight).clamped(),
                confidence = score
            )
        }
        val reliable = points.filter { it.confidence >= KEYPOINT_THRESHOLD }
        require(reliable.size >= MIN_RELIABLE_KEYPOINTS) { "Pose has too few reliable keypoints" }
        val minX = reliable.minOf { it.position.x }
        val minY = reliable.minOf { it.position.y }
        val maxX = reliable.maxOf { it.position.x }
        val maxY = reliable.maxOf { it.position.y }
        val bounds = NormalizedRect(minX, minY, maxX, maxY).expanded(1.18, 1.22)
        val confidence = reliable.map { it.confidence }.average().coerceIn(0.0, 1.0)
        return SubjectPoseState(points, bounds, confidence, "rtmpose_s_coco17")
    }

    private fun argmax(values: FloatArray, start: Int, length: Int): Int {
        var bestIndex = 0
        var best = Float.NEGATIVE_INFINITY
        for (i in 0 until length) {
            val value = values[start + i]
            if (value > best) { best = value; bestIndex = i }
        }
        return bestIndex
    }

    private fun inputSignature() = "rgb-f32-nchw:1x3x${INPUT_HEIGHT}x$INPUT_WIDTH"

    companion object {
        private const val INPUT_WIDTH = 192
        private const val INPUT_HEIGHT = 256
        private const val INPUT = "input"
        private const val OUTPUT_X = "simcc_x"
        private const val OUTPUT_Y = "simcc_y"
        private const val KEYPOINT_COUNT = 17
        private const val SIMCC_X_SIZE = 384
        private const val SIMCC_Y_SIZE = 512
        private const val SIMCC_SPLIT_RATIO = 2.0
        private const val BBOX_SCALE = 1.25
        private const val KEYPOINT_THRESHOLD = 0.18
        private const val MIN_RELIABLE_KEYPOINTS = 4
        private const val POSE_DEADLINE_MS = 150.0
        private const val MEAN_R = 123.675f
        private const val MEAN_G = 116.28f
        private const val MEAN_B = 103.53f
        private const val STD_R = 58.395f
        private const val STD_G = 57.12f
        private const val STD_B = 57.375f
        private val BACKEND_PROBE_ORDER = listOf(ComputeBackend.NNAPI, ComputeBackend.CPU_XNNPACK)
    }
}
