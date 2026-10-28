package com.saunak.director.storage

import android.graphics.Bitmap
import android.net.Uri
import com.saunak.director.director.SierraRenderSpec
import com.saunak.director.engine.SierraEngine
import com.saunak.director.processing.SierraRenderExecutor
import java.io.File
import kotlin.math.max
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

suspend fun PhotoRepository.saveSierraRenderFinal(
    file: File,
    renderSpec: SierraRenderSpec,
    aspectRatio: String,
    cropCenterX: Double,
    cropCenterY: Double
): Uri {
    var source: Bitmap? = decodeOriented(file, renderSpec.execution.workingLongEdge.coerceIn(1024, PROCESSED_MAX_DIMENSION))
    var cropped: Bitmap? = null
    var rendered: Bitmap? = null
    return try {
        val ratio = SierraCropPolicy.ratioOrNull(aspectRatio)
        cropped = if (ratio == null) source else cropAnipBitmap(checkNotNull(source), ratio.first, ratio.second, cropCenterX, cropCenterY)
        if (cropped !== source) {
            source?.let { if (!it.isRecycled) it.recycle() }
            source = null
        }
        rendered = withContext(Dispatchers.Default) {
            SierraRenderExecutor.render(
                source = checkNotNull(cropped),
                spec = renderSpec,
                subject = SierraEngine.getOrNull()?.subjects?.latest(),
                protectSource = false
            ).bitmap
        }
        saveProcessed(checkNotNull(rendered), file)
    } finally {
        rendered?.let { if (!it.isRecycled) it.recycle() }
        cropped?.let { if (it !== rendered && !it.isRecycled) it.recycle() }
        source?.let { if (it !== cropped && it !== rendered && !it.isRecycled) it.recycle() }
    }
}

suspend fun PhotoRepository.savePrifSierraRenderFinal(
    bitmap: Bitmap,
    sourceJpeg: File,
    renderSpec: SierraRenderSpec,
    aspectRatio: String,
    cropCenterX: Double,
    cropCenterY: Double
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
    var rendered: Bitmap? = null
    return try {
        withContext(Dispatchers.Default) {
            val maxDimension = minOf(renderSpec.execution.workingLongEdge.coerceIn(1024, 4096), prifSierraProcessingMaxDimension())
            val longest = max(bitmap.width, bitmap.height)
            working = if (longest > maxDimension) {
                val scale = maxDimension.toFloat() / longest
                Bitmap.createScaledBitmap(bitmap, maxOf(1, (bitmap.width * scale).toInt()), maxOf(1, (bitmap.height * scale).toInt()), true)
            } else bitmap
            val ratio = SierraCropPolicy.ratioOrNull(aspectRatio)
            cropped = if (ratio == null) working else cropAnipBitmap(checkNotNull(working), ratio.first, ratio.second, cropCenterX, cropCenterY)
            if (cropped !== working && working !== bitmap) {
                working?.let { if (!it.isRecycled) it.recycle() }
                working = null
            }
            rendered = SierraRenderExecutor.render(
                source = checkNotNull(cropped),
                spec = renderSpec,
                subject = SierraEngine.getOrNull()?.subjects?.latest(),
                protectSource = cropped === bitmap
            ).bitmap
        }
        saveProcessed(checkNotNull(rendered), sourceJpeg)
    } catch (t: Throwable) {
        transactionIds.forEach(evidenceLedger::remove)
        throw t
    } finally {
        rendered?.let { if (it !== bitmap && !it.isRecycled) it.recycle() }
        cropped?.let { if (it !== bitmap && it !== rendered && !it.isRecycled) it.recycle() }
        working?.let { if (it !== bitmap && it !== cropped && it !== rendered && !it.isRecycled) it.recycle() }
    }
}

private fun prifSierraProcessingMaxDimension(): Int {
    val heapMb = Runtime.getRuntime().maxMemory() / (1024L * 1024L)
    return when {
        heapMb < 256L -> 1792
        heapMb < 384L -> 2304
        else -> 3072
    }
}
