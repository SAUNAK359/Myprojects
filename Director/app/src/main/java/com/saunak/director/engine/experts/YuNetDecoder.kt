package com.saunak.director.engine.experts

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class FacePoint(val x: Float, val y: Float)

data class FaceDetection(
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float,
    val landmarks: List<FacePoint>,
    val score: Float
) {
    init {
        require(x.isFinite() && y.isFinite() && width.isFinite() && height.isFinite())
        require(width >= 0f && height >= 0f)
        require(landmarks.size == 5)
        require(landmarks.all { it.x.isFinite() && it.y.isFinite() })
        require(score in 0f..1f)
    }

    val area: Float get() = width * height
}

/** Exact YuNet 12-head post-processing used by FaceDetectorYN-style exports. */
object YuNetDecoder {
    private val strides = intArrayOf(8, 16, 32)

    fun decode(
        outputs: Map<String, FloatArray>,
        inputWidth: Int,
        inputHeight: Int,
        scoreThreshold: Float = 0.72f,
        nmsThreshold: Float = 0.30f,
        topK: Int = 256,
        maxFaces: Int = 16
    ): List<FaceDetection> {
        require(inputWidth > 0 && inputHeight > 0)
        require(inputWidth % 32 == 0 && inputHeight % 32 == 0) { "YuNet input dimensions must be multiples of 32" }
        require(scoreThreshold in 0f..1f)
        require(nmsThreshold in 0f..1f)
        require(topK > 0 && maxFaces > 0)

        val candidates = ArrayList<FaceDetection>()
        for (stride in strides) {
            val cols = inputWidth / stride
            val rows = inputHeight / stride
            val anchors = cols * rows
            val cls = requireHead(outputs, "cls_$stride", anchors)
            val obj = requireHead(outputs, "obj_$stride", anchors)
            val bbox = requireHead(outputs, "bbox_$stride", anchors * 4)
            val kps = requireHead(outputs, "kps_$stride", anchors * 10)

            for (anchor in 0 until anchors) {
                val clsScore = cls[anchor].coerceIn(0f, 1f)
                val objScore = obj[anchor].coerceIn(0f, 1f)
                val score = sqrt((clsScore * objScore).toDouble()).toFloat()
                if (!score.isFinite() || score < scoreThreshold) continue

                val row = anchor / cols
                val col = anchor % cols
                val base = anchor * 4
                val dx = bbox[base]
                val dy = bbox[base + 1]
                val dw = bbox[base + 2].coerceIn(-20f, 20f)
                val dh = bbox[base + 3].coerceIn(-20f, 20f)
                if (!dx.isFinite() || !dy.isFinite() || !dw.isFinite() || !dh.isFinite()) continue

                val centerX = (col + dx) * stride
                val centerY = (row + dy) * stride
                val width = exp(dw.toDouble()).toFloat() * stride
                val height = exp(dh.toDouble()).toFloat() * stride
                if (!width.isFinite() || !height.isFinite() || width <= 0f || height <= 0f) continue

                val landmarks = ArrayList<FacePoint>(5)
                var valid = true
                val kBase = anchor * 10
                for (point in 0 until 5) {
                    val kx = kps[kBase + point * 2]
                    val ky = kps[kBase + point * 2 + 1]
                    if (!kx.isFinite() || !ky.isFinite()) {
                        valid = false
                        break
                    }
                    landmarks += FacePoint((col + kx) * stride, (row + ky) * stride)
                }
                if (!valid) continue

                candidates += FaceDetection(
                    x = centerX - width / 2f,
                    y = centerY - height / 2f,
                    width = width,
                    height = height,
                    landmarks = landmarks,
                    score = score
                )
            }
        }

        return nonMaximumSuppression(candidates, nmsThreshold, topK, maxFaces)
    }

    fun mapToSource(
        detection: FaceDetection,
        sourceWidth: Int,
        sourceHeight: Int,
        inputWidth: Int,
        inputHeight: Int
    ): FaceDetection {
        require(sourceWidth > 0 && sourceHeight > 0 && inputWidth > 0 && inputHeight > 0)
        val sx = sourceWidth.toFloat() / inputWidth.toFloat()
        val sy = sourceHeight.toFloat() / inputHeight.toFloat()
        val x1 = (detection.x * sx).coerceIn(0f, sourceWidth.toFloat())
        val y1 = (detection.y * sy).coerceIn(0f, sourceHeight.toFloat())
        val x2 = ((detection.x + detection.width) * sx).coerceIn(0f, sourceWidth.toFloat())
        val y2 = ((detection.y + detection.height) * sy).coerceIn(0f, sourceHeight.toFloat())
        return FaceDetection(
            x = min(x1, x2),
            y = min(y1, y2),
            width = (x2 - x1).coerceAtLeast(0f),
            height = (y2 - y1).coerceAtLeast(0f),
            landmarks = detection.landmarks.map {
                FacePoint(
                    (it.x * sx).coerceIn(0f, sourceWidth.toFloat()),
                    (it.y * sy).coerceIn(0f, sourceHeight.toFloat())
                )
            },
            score = detection.score
        )
    }

    private fun requireHead(outputs: Map<String, FloatArray>, name: String, expected: Int): FloatArray {
        val values = requireNotNull(outputs[name]) { "Missing YuNet output '$name'" }
        require(values.size == expected) { "YuNet output '$name' has ${values.size} values, expected $expected" }
        return values
    }

    private fun nonMaximumSuppression(
        source: List<FaceDetection>,
        threshold: Float,
        topK: Int,
        maxFaces: Int
    ): List<FaceDetection> {
        if (source.isEmpty()) return emptyList()
        val sorted = source.sortedByDescending(FaceDetection::score).take(topK)
        val kept = ArrayList<FaceDetection>(min(maxFaces, sorted.size))
        for (candidate in sorted) {
            if (kept.none { intersectionOverUnion(candidate, it) > threshold }) {
                kept += candidate
                if (kept.size >= maxFaces) break
            }
        }
        return kept
    }

    internal fun intersectionOverUnion(a: FaceDetection, b: FaceDetection): Float {
        val ax2 = a.x + a.width
        val ay2 = a.y + a.height
        val bx2 = b.x + b.width
        val by2 = b.y + b.height
        val ix1 = max(a.x, b.x)
        val iy1 = max(a.y, b.y)
        val ix2 = min(ax2, bx2)
        val iy2 = min(ay2, by2)
        val iw = (ix2 - ix1).coerceAtLeast(0f)
        val ih = (iy2 - iy1).coerceAtLeast(0f)
        val intersection = iw * ih
        val union = a.area + b.area - intersection
        return if (union <= 0f) 0f else (intersection / union).coerceIn(0f, 1f)
    }
}
