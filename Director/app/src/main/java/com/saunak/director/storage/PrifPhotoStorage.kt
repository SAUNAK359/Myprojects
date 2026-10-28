package com.saunak.director.storage

import android.graphics.Bitmap
import android.net.Uri
import com.saunak.director.director.FinalRenderAdjustment
import com.saunak.director.director.PhotographicDirection
import com.saunak.director.engine.SierraEngine
import com.saunak.director.processing.LookSpec
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.max

suspend fun PhotoRepository.savePrifReconstruction(
    bitmap: Bitmap,
    sourceJpeg: File,
    look: LookSpec,
    aspectRatio: String,
    cropCenterX: Double,
    cropCenterY: Double,
    sceneLabel: String = "",
    objective: String = "",
    reconstructionIntegrity: Double = 1.0,
    direction: PhotographicDirection? = null,
    adjustment: FinalRenderAdjustment? = null
): Uri {
    require(!bitmap.isRecycled)
    val evidenceLedger = SierraEngine.get().evidence
    val newestPrifTimestamp = evidenceLedger.snapshot().asSequence()
        .filter { it.regionId.value.startsWith("prif/") }
        .maxOfOrNull { it.timestampNs }
    val transactionIds = if (newestPrifTimestamp == null) emptyList() else evidenceLedger.snapshot()
        .filter { it.timestampNs == newestPrifTimestamp && it.regionId.value.startsWith("prif/") }
        .map { it.regionId }

    var working: Bitmap? = null
    var cropped: Bitmap? = null
    var renderedBitmap: Bitmap? = null
    return try {
        val rendered = withContext(Dispatchers.Default) {
            val maxDimension = prifProcessingMaxDimension()
            val longest = max(bitmap.width, bitmap.height)
            working = if (longest > maxDimension) {
                val scale = maxDimension.toFloat() / longest
                Bitmap.createScaledBitmap(bitmap, maxOf(1, (bitmap.width * scale).toInt()), maxOf(1, (bitmap.height * scale).toInt()), true)
            } else bitmap

            val ratio = SierraCropPolicy.ratioOrNull(aspectRatio)
            cropped = if (ratio == null) working else cropAnipBitmap(checkNotNull(working), ratio.first, ratio.second, cropCenterX, cropCenterY)

            // Once crop owns an independent bitmap, the temporary scaled working copy is dead.
            if (cropped !== working && working !== bitmap) {
                working?.let { if (!it.isRecycled) it.recycle() }
                working = null
            }

            val output = AnipPhotoFinish.render(
                source = checkNotNull(cropped),
                look = look,
                sceneLabel = sceneLabel,
                objective = objective,
                reconstructionIntegrity = reconstructionIntegrity,
                direction = direction,
                adjustment = adjustment
            )
            renderedBitmap = output.bitmap

            // ANIP now owns the visual result. Release any temporary crop/scale bitmap immediately;
            // the caller-owned reconstruction bitmap is never recycled here.
            if (renderedBitmap !== cropped && cropped !== bitmap) {
                cropped?.let { if (!it.isRecycled) it.recycle() }
                if (cropped === working) working = null
                cropped = null
            }
            output.bitmap
        }
        saveProcessed(rendered, sourceJpeg)
    } catch (t: Throwable) {
        transactionIds.forEach(evidenceLedger::remove)
        throw t
    } finally {
        renderedBitmap?.let { if (it !== bitmap && !it.isRecycled) it.recycle() }
        cropped?.let { if (it !== bitmap && it !== renderedBitmap && !it.isRecycled) it.recycle() }
        working?.let { if (it !== bitmap && it !== cropped && it !== renderedBitmap && !it.isRecycled) it.recycle() }
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
