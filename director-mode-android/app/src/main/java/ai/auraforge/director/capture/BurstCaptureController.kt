package ai.auraforge.director.capture

import ai.auraforge.director.quality.ImageQualityAnalyzer
import android.content.ContentValues
import android.content.Context
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import java.io.File
import java.io.FileInputStream
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicBoolean

class BurstCaptureController(
    private val context: Context,
    private val imageCaptureProvider: () -> ImageCapture,
    private val executor: Executor,
    private val qualityAnalyzer: ImageQualityAnalyzer
) {
    private val capturing = AtomicBoolean(false)

    fun capture(
        count: Int = 3,
        externalOutput: Uri? = null,
        onComplete: (Result<Uri>) -> Unit
    ) {
        if (!capturing.compareAndSet(false, true)) return
        val files = mutableListOf<File>()
        captureNext(0, count.coerceIn(1, 3), files) { result ->
            if (result.isFailure) {
                files.forEach { it.delete() }
                capturing.set(false)
                onComplete(Result.failure(result.exceptionOrNull() ?: IllegalStateException("Capture failed")))
                return@captureNext
            }

            runCatching {
                val valid4K = files.filter(::isNative4KFile)
                require(valid4K.isNotEmpty()) {
                    "Camera HAL advertised native 4K, but the captured JPEG was below 3840×2160. The image was not upscaled or saved as fake 4K."
                }
                val best = selectBest(valid4K)
                val uri = if (externalOutput != null) {
                    context.contentResolver.openOutputStream(externalOutput, "w")!!.use { out ->
                        FileInputStream(best).use { it.copyTo(out) }
                    }
                    externalOutput
                } else {
                    saveToGallery(best)
                }
                files.forEach { it.delete() }
                uri
            }.also {
                capturing.set(false)
                onComplete(it)
            }
        }
    }

    private fun captureNext(
        index: Int,
        count: Int,
        files: MutableList<File>,
        done: (Result<Unit>) -> Unit
    ) {
        if (index >= count) {
            done(Result.success(Unit))
            return
        }

        val file = File.createTempFile("director_${System.currentTimeMillis()}_${index}_", ".jpg", context.cacheDir)
        val options = ImageCapture.OutputFileOptions.Builder(file).build()
        imageCaptureProvider().takePicture(
            options,
            executor,
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    files += file
                    captureNext(index + 1, count, files, done)
                }

                override fun onError(exception: ImageCaptureException) {
                    file.delete()
                    done(Result.failure(exception))
                }
            }
        )
    }

    private fun isNative4KFile(file: File): Boolean {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        val longSide = maxOf(bounds.outWidth, bounds.outHeight)
        val shortSide = minOf(bounds.outWidth, bounds.outHeight)
        return longSide >= 3840 && shortSide >= 2160
    }

    private fun selectBest(files: List<File>): File {
        if (files.size == 1) return files.first()
        return files.maxByOrNull { file ->
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(file.absolutePath, bounds)
            val sample = when {
                bounds.outWidth > 6000 -> 12
                bounds.outWidth > 4000 -> 8
                else -> 4
            }
            val opts = BitmapFactory.Options().apply { inSampleSize = sample }
            val bitmap = BitmapFactory.decodeFile(file.absolutePath, opts)
            if (bitmap == null) 0f else qualityAnalyzer.scoreJpegThumbnail(bitmap).also { bitmap.recycle() }
        } ?: files.first()
    }

    private fun saveToGallery(file: File): Uri {
        val resolver = context.contentResolver
        val name = "Director_${System.currentTimeMillis()}.jpg"
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "DCIM/Director")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
        }
        val collection = MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        val uri = resolver.insert(collection, values) ?: error("Unable to create MediaStore item")
        resolver.openOutputStream(uri, "w")!!.use { out ->
            FileInputStream(file).use { it.copyTo(out) }
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val done = ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }
            resolver.update(uri, done, null, null)
        }
        return uri
    }
}
