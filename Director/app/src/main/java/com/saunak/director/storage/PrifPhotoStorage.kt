package com.saunak.director.storage

import android.graphics.Bitmap
import android.net.Uri
import com.saunak.director.engine.SierraEngine
import com.saunak.director.processing.LookSpec
import java.io.File

/**
 * Keeps the accepted P-RIF reconstruction in memory through ANIP rendering, crop and final save.
 * [sourceJpeg] remains the safe EXIF source and same-shutter recovery image.
 *
 * P-RIF evidence is rolled back if the final gallery write fails, so Sierra never claims that a
 * reconstructed/rendered photo was delivered when only the safety JPEG survived.
 */
suspend fun PhotoRepository.savePrifReconstruction(
    bitmap: Bitmap,
    sourceJpeg: File,
    look: LookSpec,
    aspectRatio: String,
    cropCenterX: Double,
    cropCenterY: Double,
    sceneLabel: String = "",
    objective: String = "",
    reconstructionIntegrity: Double = 1.0
): Uri {
    require(!bitmap.isRecycled)
    val evidenceLedger = SierraEngine.get().evidence
    val newestPrifTimestamp = evidenceLedger.snapshot()
        .asSequence()
        .filter { it.regionId.value.startsWith("prif/") }
        .maxOfOrNull { it.timestampNs }
    val transactionIds = if (newestPrifTimestamp == null) emptyList() else evidenceLedger.snapshot()
        .filter { it.timestampNs == newestPrifTimestamp && it.regionId.value.startsWith("prif/") }
        .map { it.regionId }

    var anip: Bitmap? = null
    var cropped: Bitmap? = null
    return try {
        val output = AnipPhotoFinish.render(
            source = bitmap,
            look = look,
            sceneLabel = sceneLabel,
            objective = objective,
            reconstructionIntegrity = reconstructionIntegrity
        )
        anip = output.bitmap
        val ratio = SierraCropPolicy.ratioOrNull(aspectRatio)
        cropped = if (ratio == null) output.bitmap else cropAnipBitmap(
            output.bitmap, ratio.first, ratio.second, cropCenterX, cropCenterY
        )
        saveProcessed(checkNotNull(cropped), sourceJpeg)
    } catch (t: Throwable) {
        transactionIds.forEach(evidenceLedger::remove)
        throw t
    } finally {
        cropped?.let { if (it !== anip && !it.isRecycled) it.recycle() }
        anip?.let { if (!it.isRecycled) it.recycle() }
    }
}
