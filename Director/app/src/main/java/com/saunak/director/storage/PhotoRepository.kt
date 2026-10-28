package com.saunak.director.storage

import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.StatFs
import android.provider.MediaStore
import androidx.exifinterface.media.ExifInterface
import com.saunak.director.processing.LookProcessor
import com.saunak.director.processing.LookSpec
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Locale
import kotlin.math.max
import kotlin.math.roundToInt

class PhotoRepository(private val context: Context) {
    private val pendingDir = File(context.noBackupFilesDir, "sierra_pending").apply { mkdirs() }
    private val attemptDir = File(context.cacheDir, "sierra_attempts").apply { mkdirs() }
    private val comparisonDir = File(context.cacheDir, "sierra_compare").apply { mkdirs() }
    @Volatile private var lastComparisonReferencePath: String? = null

    fun newPendingReference(sessionId: Long): File = File(pendingDir, "reference_${sessionId}_${System.currentTimeMillis()}.jpg")
    fun newAttempt(sessionId: Long, index: Int): File = File(attemptDir, "attempt_${sessionId}_${index}_${System.nanoTime()}.jpg")
    fun hasWorkingSpace(requiredBytes: Long = 96L * 1024L * 1024L): Boolean = runCatching { StatFs(context.filesDir.absolutePath).availableBytes >= requiredBytes }.getOrDefault(true)

    suspend fun decodeOriented(file: File, maxDimension: Int = AI_DECODE_MAX_DIMENSION): Bitmap = withContext(Dispatchers.IO) {
        require(maxDimension >= 256) { "Invalid decode bound" }
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        check(bounds.outWidth > 0 && bounds.outHeight > 0) { "Could not inspect captured image" }
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / sample > maxDimension * 2) sample *= 2
        val options = BitmapFactory.Options().apply { inSampleSize = sample; inPreferredConfig = Bitmap.Config.ARGB_8888 }
        var bitmap = BitmapFactory.decodeFile(file.absolutePath, options) ?: error("Could not decode captured image")
        val orientation = runCatching { ExifInterface(file).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL) }.getOrDefault(ExifInterface.ORIENTATION_NORMAL)
        val matrix = Matrix()
        val needsTransform = when (orientation) {
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> { matrix.setScale(-1f, 1f); true }
            ExifInterface.ORIENTATION_ROTATE_180 -> { matrix.setRotate(180f); true }
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> { matrix.setScale(1f, -1f); true }
            ExifInterface.ORIENTATION_TRANSPOSE -> { matrix.setRotate(90f); matrix.postScale(-1f, 1f); true }
            ExifInterface.ORIENTATION_ROTATE_90 -> { matrix.setRotate(90f); true }
            ExifInterface.ORIENTATION_TRANSVERSE -> { matrix.setRotate(270f); matrix.postScale(-1f, 1f); true }
            ExifInterface.ORIENTATION_ROTATE_270 -> { matrix.setRotate(270f); true }
            else -> false
        }
        if (needsTransform) {
            val transformed = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            if (transformed !== bitmap) bitmap.recycle()
            bitmap = transformed
        }
        val longest = max(bitmap.width, bitmap.height)
        if (longest > maxDimension) {
            val scale = maxDimension.toFloat() / longest.toFloat()
            val scaled = Bitmap.createScaledBitmap(bitmap, maxOf(1, (bitmap.width * scale).toInt()), maxOf(1, (bitmap.height * scale).toInt()), true)
            if (scaled !== bitmap) bitmap.recycle()
            bitmap = scaled
        }
        bitmap
    }

    suspend fun stageComparisonPreview(file: File): String? = withContext(Dispatchers.IO) {
        runCatching {
            val bitmap = decodeOriented(file, COMPARISON_LONG_EDGE)
            try { stageComparisonPreviewInternal(bitmap) } finally { if (!bitmap.isRecycled) bitmap.recycle() }
        }.getOrNull()
    }

    suspend fun stageComparisonPreview(bitmap: Bitmap): String? = withContext(Dispatchers.IO) {
        runCatching { stageComparisonPreviewInternal(bitmap) }.getOrNull()
    }

    fun consumeComparisonReferencePath(): String? {
        val path = lastComparisonReferencePath
        lastComparisonReferencePath = null
        return path
    }

    private fun stageComparisonPreviewInternal(source: Bitmap): String {
        require(!source.isRecycled)
        val longest = max(source.width, source.height)
        val preview = if (longest > COMPARISON_LONG_EDGE) {
            val scale = COMPARISON_LONG_EDGE.toFloat() / longest
            Bitmap.createScaledBitmap(source, (source.width * scale).toInt().coerceAtLeast(1), (source.height * scale).toInt().coerceAtLeast(1), true)
        } else source
        val file = File(comparisonDir, "before_${System.currentTimeMillis()}_${System.nanoTime()}.jpg")
        try {
            FileOutputStream(file).use { out -> check(preview.compress(Bitmap.CompressFormat.JPEG, COMPARISON_QUALITY, out)) }
        } finally {
            if (preview !== source && !preview.isRecycled) preview.recycle()
        }
        lastComparisonReferencePath = file.absolutePath
        comparisonDir.listFiles()?.filter(File::isFile)?.sortedByDescending { it.lastModified() }?.drop(MAX_COMPARISON_FILES)?.forEach { runCatching { it.delete() } }
        return file.absolutePath
    }

    suspend fun saveFinal(file: File, look: LookSpec, aspectRatio: String = "4:3", cropCenterX: Double = 0.5, cropCenterY: Double = 0.5): Uri {
        val ratio = SierraCropPolicy.ratioOrNull(aspectRatio)
        if (look.isIdentity() && ratio == null) return saveRawJpeg(file)
        val source = decodeOriented(file, processingMaxDimension())
        var cropped: Bitmap? = null
        var rendered: Bitmap? = null
        return try {
            val pair = withContext(Dispatchers.Default) {
                val c = if (ratio == null) source else cropToRatio(source, ratio.first, ratio.second, cropCenterX, cropCenterY)
                val r = LookProcessor.apply(c, look)
                c to r
            }
            cropped = pair.first; rendered = pair.second
            saveProcessed(checkNotNull(rendered), file)
        } finally {
            rendered?.let { if (it !== cropped && !it.isRecycled) it.recycle() }
            cropped?.let { if (it !== source && !it.isRecycled) it.recycle() }
            if (!source.isRecycled) source.recycle()
        }
    }

    suspend fun saveOriginal(file: File): Uri = saveRawJpeg(file, "Pictures/Sierra Vision/Originals", "SIERRA_ORIGINAL")

    suspend fun recentFinals(limit: Int = 60): List<Uri> = withContext(Dispatchers.IO) {
        val collection = MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        val projection = arrayOf(MediaStore.Images.Media._ID, MediaStore.Images.Media.DISPLAY_NAME)
        val selection = "${MediaStore.Images.Media.RELATIVE_PATH}=?"
        val args = arrayOf("Pictures/Sierra Vision/")
        val result = ArrayList<Uri>(limit.coerceAtLeast(1))
        context.contentResolver.query(collection, projection, selection, args, "${MediaStore.Images.Media.DATE_ADDED} DESC")?.use { cursor ->
            val idColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
            val nameColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DISPLAY_NAME)
            while (cursor.moveToNext() && result.size < limit) {
                val name = cursor.getString(nameColumn).orEmpty()
                if (!name.startsWith("SIERRA_") || name.startsWith("SIERRA_ORIGINAL_")) continue
                result += ContentUris.withAppendedId(collection, cursor.getLong(idColumn))
            }
        }
        result
    }

    private fun cropToRatio(source: Bitmap, ratioWidth: Int, ratioHeight: Int, centerX: Double, centerY: Double): Bitmap {
        val target = ratioWidth.toDouble() / ratioHeight.toDouble()
        val current = source.width.toDouble() / source.height.toDouble()
        if (kotlin.math.abs(target - current) < 0.015) return source
        val cropWidth: Int; val cropHeight: Int
        if (current > target) { cropHeight = source.height; cropWidth = (cropHeight * target).roundToInt().coerceIn(1, source.width) }
        else { cropWidth = source.width; cropHeight = (cropWidth / target).roundToInt().coerceIn(1, source.height) }
        val safeCx = centerX.coerceIn(0.0, 1.0); val safeCy = centerY.coerceIn(0.0, 1.0)
        val left = (safeCx * source.width - cropWidth / 2.0).roundToInt().coerceIn(0, source.width - cropWidth)
        val top = (safeCy * source.height - cropHeight / 2.0).roundToInt().coerceIn(0, source.height - cropHeight)
        return Bitmap.createBitmap(source, left, top, cropWidth, cropHeight)
    }

    suspend fun saveRawJpeg(file: File, relativePath: String = "Pictures/Sierra Vision", prefix: String = "SIERRA"): Uri = withContext(Dispatchers.IO) {
        require(file.isFile && file.length() > 0) { "Reference JPEG missing" }
        val uri = createPendingMediaUri(file.nameWithoutExtension, relativePath, prefix)
        try {
            context.contentResolver.openOutputStream(uri, "w")!!.use { out -> file.inputStream().use { it.copyTo(out) } }
            commitMediaUri(uri); uri
        } catch (t: Throwable) { context.contentResolver.delete(uri, null, null); throw t }
    }

    suspend fun saveProcessed(bitmap: Bitmap, sourceFile: File): Uri = withContext(Dispatchers.IO) {
        val uri = createPendingMediaUri(sourceFile.nameWithoutExtension, "Pictures/Sierra Vision", "SIERRA")
        try {
            context.contentResolver.openOutputStream(uri, "w")!!.use { out -> check(bitmap.compress(Bitmap.CompressFormat.JPEG, 96, out)) }
            copySafeExif(sourceFile, uri); commitMediaUri(uri); uri
        } catch (t: Throwable) { context.contentResolver.delete(uri, null, null); throw t }
    }

    private fun createPendingMediaUri(stem: String, relativePath: String, prefix: String): Uri {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(System.currentTimeMillis())
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "${prefix}_${timestamp}_${stem.takeLast(10)}.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, relativePath)
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        return context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: error("Could not create gallery image")
    }

    private fun commitMediaUri(uri: Uri) {
        val values = ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }
        context.contentResolver.update(uri, values, null, null)
    }

    private fun copySafeExif(sourceFile: File, targetUri: Uri) {
        runCatching {
            val src = ExifInterface(sourceFile)
            context.contentResolver.openFileDescriptor(targetUri, "rw")?.use { pfd ->
                val dst = ExifInterface(pfd.fileDescriptor)
                listOf(
                    ExifInterface.TAG_DATETIME, ExifInterface.TAG_DATETIME_ORIGINAL, ExifInterface.TAG_MAKE,
                    ExifInterface.TAG_MODEL, ExifInterface.TAG_FOCAL_LENGTH, ExifInterface.TAG_EXPOSURE_TIME,
                    ExifInterface.TAG_F_NUMBER, ExifInterface.TAG_PHOTOGRAPHIC_SENSITIVITY,
                    ExifInterface.TAG_FLASH, ExifInterface.TAG_WHITE_BALANCE
                ).forEach { tag -> src.getAttribute(tag)?.let { dst.setAttribute(tag, it) } }
                dst.setAttribute(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL.toString())
                dst.saveAttributes()
            }
        }
    }

    suspend fun recoverPendingToGallery(): List<Uri> = withContext(Dispatchers.IO) {
        val saved = mutableListOf<Uri>()
        pendingDir.listFiles { f -> f.isFile && f.extension.equals("jpg", true) }?.sortedBy { it.lastModified() }?.forEach { file ->
            runCatching { saveRawJpeg(file) }.onSuccess { uri -> saved += uri; file.delete() }
        }
        saved
    }

    fun cleanup(files: Iterable<File>, keep: File? = null) { files.forEach { if (it != keep) runCatching { it.delete() } } }
    fun markFinalized(reference: File) { runCatching { reference.delete() } }

    private fun processingMaxDimension(): Int {
        val heapMb = Runtime.getRuntime().maxMemory() / (1024L * 1024L)
        return when { heapMb < 256L -> 2048; heapMb < 384L -> 2560; else -> 3072 }
    }

    companion object {
        const val AI_DECODE_MAX_DIMENSION = 1024
        const val PROCESSED_MAX_DIMENSION = 4096
        private const val COMPARISON_LONG_EDGE = 1280
        private const val COMPARISON_QUALITY = 88
        private const val MAX_COMPARISON_FILES = 8
    }
}

object SierraCropPolicy {
    private val ratios = linkedMapOf(
        "4:3" to (4 to 3), "3:2" to (3 to 2), "4:5" to (4 to 5), "1:1" to (1 to 1),
        "16:9" to (16 to 9), "9:16" to (9 to 16), "2.39:1" to (239 to 100)
    )
    val supported: List<String> = ratios.keys.toList()
    fun ratioOrNull(value: String): Pair<Int, Int>? = ratios[value.trim()]
    fun isSupported(value: String): Boolean = ratioOrNull(value) != null
}
