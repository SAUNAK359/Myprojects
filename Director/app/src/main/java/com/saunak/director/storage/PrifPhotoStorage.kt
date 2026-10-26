package com.saunak.director.storage

import android.graphics.Bitmap
import android.net.Uri
import com.saunak.director.engine.SierraEngine
import com.saunak.director.processing.AdaptiveContrastFinisher
import com.saunak.director.processing.LookSpec
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.max

/**
 * Keeps the accepted P-RIF reconstruction protected while ANIP renders a bounded working copy.
 * [sourceJpeg] remains the same-shutter recovery image if finishing fails on a specific device.
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

    var working: Bitmap? = null
    var cropped: Bitmap? = null
    var anip: Bitmap? = null
    var finished: Bitmap? = null
    return try {
        val rendered = withContext(Dispatchers.Default) {
            val maxDimension = prifProcessingMaxDimension()
            val longest = max(bitmap.width, bitmap.height)
            working = if (longest > maxDimension) {
                val scale = maxDimension.toFloat() / longest.toFloat()
                Bitmap.createScaledBitmap(
                    bitmap,
                    maxOf(1, (bitmap.width * scale).toInt()),
                    maxOf(1, (bitmap.height * scale).toInt()),
                    true
                )
            } else bitmap

            val ratio = SierraCropPolicy.ratioOrNull(aspectRatio)
            cropped = if (ratio == null) working else cropAnipBitmap(
                checkNotNull(working), ratio.first, ratio.second, cropCenterX, cropCenterY
            )
            val output = AnipPhotoFinish.render(
                source = checkNotNull(cropped),
                look = look,
                sceneLabel = sceneLabel,
                objective = objective,
                reconstructionIntegrity = reconstructionIntegrity
            )
            anip = output.bitmap
            finished = AdaptiveContrastFinisher.apply(output.bitmap)
            checkNotNull(finished)
        }
        saveProcessed(rendered, sourceJpeg)
    } catch (t: Throwable) {
        transactionIds.forEach(evidenceLedger::remove)
        throw t
    } finally {
        finished?.let { if (it !== anip && it !== cropped && it !== working && it !== bitmap && !it.isRecycled) it.recycle() }
        anip?.let { if (it !== cropped && !it.isRecycled) it.recycle() }
        cropped?.let { if (it !== working && it !== bitmap && !it.isRecycled) it.recycle() }
        working?.let { if (it !== bitmap && !it.isRecycled) it.recycle() }
    }
}

private fun prifProcessingMaxDimension(): Int {
    val heapMb = Runtime.getRuntime().maxMemory() / (1024L * 1024L)
    return when {
        heapMb < 256L -> 1792
        heapMb < 384L -> 2304
        else -> 2880
    }
}
