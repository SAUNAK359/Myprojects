package com.saunak.director.storage

import android.graphics.Bitmap
import android.net.Uri
import com.saunak.director.engine.SierraEngine
import com.saunak.director.engine.render.AnipEngine
import com.saunak.director.engine.render.AnipRequest
import com.saunak.director.engine.render.AnipRenderResult
import com.saunak.director.engine.render.AnipSubjectContext
import com.saunak.director.processing.LookSpec
import java.io.File
import kotlin.math.abs
import kotlin.math.roundToInt

internal object AnipPhotoFinish {
    data class Output(val bitmap: Bitmap, val result: AnipRenderResult)

    fun render(
        source: Bitmap,
        look: LookSpec,
        sceneLabel: String,
        objective: String,
        reconstructionIntegrity: Double
    ): Output {
        val engine = SierraEngine.getOrNull()
        val track = engine?.subjects?.latest()
        val subject = track?.let {
            val mask = it.mask
            AnipSubjectContext(
                bounds = it.bounds,
                maskWidth = mask?.width ?: 0,
                maskHeight = mask?.height ?: 0,
                mask = mask?.probabilities,
                maskConfidence = mask?.confidence ?: 0.0,
                depthSeparation = it.depth?.foregroundSeparation ?: 0.0,
                trackConfidence = it.confidence
            )
        }
        val motionRisk = engine?.motion?.latest()?.conservativeRisk ?: 0.45
        val request = AnipRequest(
            sceneLabel = sceneLabel,
            objective = objective,
            look = look.look,
            lookStrength = look.normalizedStrength,
            subjectPresent = subject != null,
            motionRisk = motionRisk,
            reconstructionIntegrity = reconstructionIntegrity.coerceIn(0.0, 1.0)
        )
        val (bitmap, result) = AnipEngine().render(source, request, subject)
        return Output(bitmap, result)
    }
}

/**
 * ANIP finishing for CameraX/JPEG evidence paths. Catastrophic frames never call this extension;
 * callers preserve those byte-for-byte through saveRawJpeg().
 */
suspend fun PhotoRepository.saveAnipFinal(
    file: File,
    look: LookSpec,
    aspectRatio: String = "4:3",
    cropCenterX: Double = 0.5,
    cropCenterY: Double = 0.5,
    sceneLabel: String = "",
    objective: String = ""
): Uri {
    val source = decodeOriented(file, PhotoRepository.PROCESSED_MAX_DIMENSION)
    var anip: Bitmap? = null
    var cropped: Bitmap? = null
    return try {
        val output = AnipPhotoFinish.render(
            source = source,
            look = look,
            sceneLabel = sceneLabel,
            objective = objective,
            reconstructionIntegrity = JPEG_EVIDENCE_INTEGRITY_PRIOR
        )
        anip = output.bitmap
        val ratio = SierraCropPolicy.ratioOrNull(aspectRatio)
        cropped = if (ratio == null) output.bitmap else cropAnipBitmap(
            output.bitmap, ratio.first, ratio.second, cropCenterX, cropCenterY
        )
        saveProcessed(checkNotNull(cropped), file)
    } finally {
        cropped?.let { if (it !== anip && !it.isRecycled) it.recycle() }
        anip?.let { if (!it.isRecycled) it.recycle() }
        if (!source.isRecycled) source.recycle()
    }
}

internal fun cropAnipBitmap(
    source: Bitmap,
    ratioWidth: Int,
    ratioHeight: Int,
    centerX: Double,
    centerY: Double
): Bitmap {
    val target = ratioWidth.toDouble() / ratioHeight.toDouble()
    val current = source.width.toDouble() / source.height.toDouble()
    if (abs(target - current) < 0.015) return source
    val cropWidth: Int
    val cropHeight: Int
    if (current > target) {
        cropHeight = source.height
        cropWidth = (cropHeight * target).roundToInt().coerceIn(1, source.width)
    } else {
        cropWidth = source.width
        cropHeight = (cropWidth / target).roundToInt().coerceIn(1, source.height)
    }
    val safeCx = centerX.coerceIn(0.0, 1.0)
    val safeCy = centerY.coerceIn(0.0, 1.0)
    val left = (safeCx * source.width - cropWidth / 2.0).roundToInt().coerceIn(0, source.width - cropWidth)
    val top = (safeCy * source.height - cropHeight / 2.0).roundToInt().coerceIn(0, source.height - cropHeight)
    return Bitmap.createBitmap(source, left, top, cropWidth, cropHeight)
}

private const val JPEG_EVIDENCE_INTEGRITY_PRIOR = 0.78
