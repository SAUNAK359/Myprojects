package com.saunak.director.storage

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
import java.text.SimpleDateFormat
import java.util.Locale

class PhotoRepository(private val context: Context) {
    private val pendingDir = File(context.noBackupFilesDir, "director_pending").apply { mkdirs() }
    private val attemptDir = File(context.cacheDir, "director_attempts").apply { mkdirs() }

    fun newPendingReference(sessionId: Long): File = File(pendingDir, "reference_${sessionId}_${System.currentTimeMillis()}.jpg")
    fun newAttempt(sessionId: Long, index: Int): File = File(attemptDir, "attempt_${sessionId}_${index}_${System.nanoTime()}.jpg")
    fun hasWorkingSpace(requiredBytes: Long = 96L * 1024L * 1024L): Boolean = runCatching { StatFs(context.filesDir.absolutePath).availableBytes >= requiredBytes }.getOrDefault(true)

    suspend fun decodeOriented(file: File): Bitmap = withContext(Dispatchers.IO) {
        val source = BitmapFactory.decodeFile(file.absolutePath) ?: error("Could not decode captured image")
        val orientation = runCatching { ExifInterface(file).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL) }.getOrDefault(ExifInterface.ORIENTATION_NORMAL)
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.setScale(-1f, 1f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.setRotate(180f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.setScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> { matrix.setRotate(90f); matrix.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.setRotate(90f)
            ExifInterface.ORIENTATION_TRANSVERSE -> { matrix.setRotate(270f); matrix.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.setRotate(270f)
            else -> return@withContext source
        }
        val corrected = Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
        if (corrected !== source) source.recycle()
        corrected
    }

    suspend fun saveFinal(file: File, look: LookSpec): Uri = if (look.isIdentity()) saveRawJpeg(file) else {
        val source = decodeOriented(file)
        val rendered = LookProcessor.apply(source, look)
        try { saveProcessed(rendered, file) } finally {
            if (rendered !== source && !rendered.isRecycled) rendered.recycle()
            if (!source.isRecycled) source.recycle()
        }
    }

    suspend fun saveRawJpeg(file: File): Uri = withContext(Dispatchers.IO) {
        require(file.isFile && file.length() > 0) { "Reference JPEG missing" }
        val uri = createPendingMediaUri(file.nameWithoutExtension)
        try {
            context.contentResolver.openOutputStream(uri, "w")!!.use { out -> file.inputStream().use { it.copyTo(out) } }
            commitMediaUri(uri)
            uri
        } catch (t: Throwable) { context.contentResolver.delete(uri, null, null); throw t }
    }

    suspend fun saveProcessed(bitmap: Bitmap, sourceFile: File): Uri = withContext(Dispatchers.IO) {
        val uri = createPendingMediaUri(sourceFile.nameWithoutExtension)
        try {
            context.contentResolver.openOutputStream(uri, "w")!!.use { out -> check(bitmap.compress(Bitmap.CompressFormat.JPEG, 96, out)) }
            copySafeExif(sourceFile, uri)
            commitMediaUri(uri)
            uri
        } catch (t: Throwable) { context.contentResolver.delete(uri, null, null); throw t }
    }

    private fun createPendingMediaUri(stem: String): Uri {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(System.currentTimeMillis())
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "DIRECTOR_${timestamp}_${stem.takeLast(10)}.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Director")
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
                val tags = listOf(ExifInterface.TAG_DATETIME, ExifInterface.TAG_DATETIME_ORIGINAL, ExifInterface.TAG_MAKE, ExifInterface.TAG_MODEL, ExifInterface.TAG_FOCAL_LENGTH, ExifInterface.TAG_EXPOSURE_TIME, ExifInterface.TAG_F_NUMBER, ExifInterface.TAG_PHOTOGRAPHIC_SENSITIVITY, ExifInterface.TAG_FLASH, ExifInterface.TAG_WHITE_BALANCE)
                tags.forEach { tag -> src.getAttribute(tag)?.let { dst.setAttribute(tag, it) } }
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
}
