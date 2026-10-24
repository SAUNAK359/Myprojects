package com.saunak.director.engine.experts

import android.graphics.Bitmap
import com.saunak.director.engine.SierraEngine
import com.saunak.director.engine.device.ComputeBackend
import com.saunak.director.engine.models.InstalledModelArtifact
import com.saunak.director.engine.models.SierraShippingManifests
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt

/** Production face monitor using the pinned MIT YuNet ONNX artifact. */
class YuNetFaceExpert(
    private val engine: SierraEngine = SierraEngine.get(),
    private val inputLongSide: Int = 320,
    private val scoreThreshold: Float = 0.72f,
    private val nmsThreshold: Float = 0.30f
) {
    init {
        require(inputLongSide >= 160 && inputLongSide % 32 == 0)
        require(scoreThreshold in 0f..1f)
        require(nmsThreshold in 0f..1f)
    }

    @Volatile private var artifact: InstalledModelArtifact? = null
    @Volatile private var selectedBackend: ComputeBackend? = null

    suspend fun prepare(): Result<Unit> = withContext(Dispatchers.IO) {
        val installed = engine.modelArtifacts.ensure(SierraShippingManifests.yunet2026May)
        installed.map { artifact = it }
    }

    suspend fun detect(source: Bitmap): Result<List<FaceDetection>> = withContext(Dispatchers.Default) {
        runCatching {
            require(!source.isRecycled) { "Source bitmap has been recycled" }
            require(source.width > 0 && source.height > 0)
            val model = artifact ?: engine.modelArtifacts.findVerified(SierraShippingManifests.yunet2026May)
                ?: error("YuNet is not prepared")
            artifact = model

            val size = chooseInputSize(source.width, source.height, inputLongSide)
            val tensor = preprocess(source, size.first, size.second)
            val outputs = runWithCertifiedFallback(model, tensor, size.first, size.second)
            YuNetDecoder.decode(
                outputs = outputs,
                inputWidth = size.first,
                inputHeight = size.second,
                scoreThreshold = scoreThreshold,
                nmsThreshold = nmsThreshold
            ).map {
                YuNetDecoder.mapToSource(it, source.width, source.height, size.first, size.second)
            }.filter { it.width >= 1f && it.height >= 1f }
        }
    }

    private fun runWithCertifiedFallback(
        model: InstalledModelArtifact,
        tensor: FloatArray,
        width: Int,
        height: Int
    ): Map<String, FloatArray> {
        val shape = longArrayOf(1, 3, height.toLong(), width.toLong())
        selectedBackend?.let { backend ->
            val warm = runCatching { run(model, backend, tensor, shape) }
            if (warm.isSuccess) return warm.getOrThrow()
            // A previously valid accelerator can disappear/reset under driver or thermal faults.
            selectedBackend = null
            engine.onnx.unload(model.manifest.id)
        }

        for (backend in BACKEND_PROBE_ORDER) {
            val attempt = runCatching { run(model, backend, tensor, shape) }
            if (attempt.isSuccess) {
                selectedBackend = backend
                return attempt.getOrThrow()
            }
        }
        error("YuNet has no usable Sierra ONNX backend on this device")
    }

    private fun run(
        model: InstalledModelArtifact,
        backend: ComputeBackend,
        tensor: FloatArray,
        shape: LongArray
    ): Map<String, FloatArray> = engine.onnx.runFloatTensor(
        artifact = model,
        backend = backend,
        inputName = INPUT,
        values = tensor,
        shape = shape,
        requestedOutputs = OUTPUTS
    )

    internal fun chooseInputSize(width: Int, height: Int, longSide: Int = inputLongSide): Pair<Int, Int> {
        require(width > 0 && height > 0 && longSide >= 32)
        fun aligned(value: Double): Int = ((value / 32.0).roundToInt().coerceAtLeast(1) * 32)
        return if (width >= height) {
            longSide to aligned(longSide.toDouble() * height.toDouble() / width.toDouble())
        } else {
            aligned(longSide.toDouble() * width.toDouble() / height.toDouble()) to longSide
        }
    }

    private fun preprocess(source: Bitmap, width: Int, height: Int): FloatArray {
        val scaled = if (source.width == width && source.height == height) source else Bitmap.createScaledBitmap(source, width, height, true)
        try {
            val pixels = IntArray(width * height)
            scaled.getPixels(pixels, 0, width, 0, 0, width, height)
            val plane = width * height
            val output = FloatArray(plane * 3)
            for (i in pixels.indices) {
                val p = pixels[i]
                output[i] = (p and 0xff).toFloat() // B
                output[plane + i] = ((p ushr 8) and 0xff).toFloat() // G
                output[plane * 2 + i] = ((p ushr 16) and 0xff).toFloat() // R
            }
            return output
        } finally {
            if (scaled !== source) scaled.recycle()
        }
    }

    companion object {
        private const val INPUT = "input"
        private val BACKEND_PROBE_ORDER = listOf(ComputeBackend.NNAPI, ComputeBackend.CPU_XNNPACK)
        private val OUTPUTS = buildSet {
            for (stride in intArrayOf(8, 16, 32)) {
                add("cls_$stride")
                add("obj_$stride")
                add("bbox_$stride")
                add("kps_$stride")
            }
        }
    }
}
