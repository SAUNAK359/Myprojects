package com.saunak.director.storage

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.provider.MediaStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale

class PhotoRepository(private val context: Context) {
    suspend fun decode(file: File): Bitmap = withContext(Dispatchers.IO) { android.graphics.BitmapFactory.decodeFile(file.absolutePath) ?: error("Could not decode captured image") }
    suspend fun saveProcessed(bitmap: Bitmap): Uri = withContext(Dispatchers.IO) {
        val name = "DIRECTOR_${SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(System.currentTimeMillis())}.jpg"
        val values = ContentValues().apply { put(MediaStore.Images.Media.DISPLAY_NAME, name); put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg"); put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Director"); put(MediaStore.Images.Media.IS_PENDING, 1) }
        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: error("Could not create gallery image")
        try {
            resolver.openOutputStream(uri, "w")!!.use { out -> check(bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)) }
            values.clear(); values.put(MediaStore.Images.Media.IS_PENDING, 0); resolver.update(uri, values, null, null); uri
        } catch (t: Throwable) { resolver.delete(uri, null, null); throw t }
    }
    fun cleanup(files: Iterable<File>, except: File? = null) { files.forEach { if (it != except) runCatching { it.delete() } } }
}
