package com.saunak.director.ai

import android.content.Context
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.RandomAccessFile
import java.security.MessageDigest
import java.util.concurrent.TimeUnit
import kotlin.coroutines.coroutineContext

class ModelManager(private val context: Context) {
    data class Files(val model: File, val projector: File)
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(90, TimeUnit.SECONDS)
        .callTimeout(60, TimeUnit.MINUTES)
        .retryOnConnectionFailure(true)
        .build()
    private val dir = File(context.filesDir, "models/qwen3-vl-2b").apply { mkdirs() }
    private val model = File(dir, "Qwen3VL-2B-Instruct-Q4_K_M.gguf")
    private val projector = File(dir, "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf")
    private val verifiedMarker = File(dir, ".verified-v1")

    companion object {
        const val MODEL_URL = "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3VL-2B-Instruct-Q4_K_M.gguf"
        const val PROJECTOR_URL = "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
        const val MODEL_SHA256 = "089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae"
        const val PROJECTOR_SHA256 = "f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82"
        const val DOWNLOAD_SIZE_LABEL = "about 1.6 GB"
        private const val MIN_FREE_BYTES = 2_000_000_000L
    }

    suspend fun ensureInstalled(onProgress: (Float) -> Unit): Result<Files> = withContext(Dispatchers.IO) {
        try {
            dir.mkdirs()
            val required = if (isInstalled()) 128L * 1024 * 1024 else MIN_FREE_BYTES
            check(dir.usableSpace >= required) { "Not enough free storage for Director AI. Free at least 2 GB and try again." }
            if (!isInstalled()) {
                verifiedMarker.delete()
                downloadIfNeeded(MODEL_URL, model, MODEL_SHA256, 0f, 0.72f, onProgress)
                downloadIfNeeded(PROJECTOR_URL, projector, PROJECTOR_SHA256, 0.72f, 1f, onProgress)
                verifiedMarker.writeText(expectedMarker())
            }
            onProgress(1f)
            Result.success(Files(model, projector))
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (t: Throwable) {
            Result.failure(t)
        }
    }

    fun isInstalled(): Boolean = runCatching {
        verifiedMarker.isFile && verifiedMarker.readText() == expectedMarker() && model.isFile && model.length() > 0L && projector.isFile && projector.length() > 0L
    }.getOrDefault(false)

    fun invalidateVerification() { runCatching { verifiedMarker.delete() } }

    private fun expectedMarker(): String = "$MODEL_SHA256\n$PROJECTOR_SHA256\n"

    private suspend fun downloadIfNeeded(url: String, destination: File, expectedSha: String, start: Float, end: Float, progress: (Float) -> Unit) {
        if (destination.exists() && sha256(destination).equals(expectedSha, true)) { progress(end); return }
        if (destination.exists()) destination.delete()
        val part = File(destination.absolutePath + ".part")
        var restartAttempted = false
        while (true) {
            var existing = part.length().coerceAtLeast(0L)
            val request = Request.Builder().url(url).get().apply { if (existing > 0L) header("Range", "bytes=$existing-") }.build()
            val shouldRestart = client.newCall(request).execute().use { response ->
                if (response.code == 416 && existing > 0L && !restartAttempted) return@use true
                check(response.isSuccessful) { "Director AI download failed: HTTP ${response.code}" }
                val body = checkNotNull(response.body) { "Director AI download returned no data" }
                val resumed = existing > 0L && response.code == 206
                if (existing > 0L && !resumed) { part.delete(); existing = 0L }
                val remaining = body.contentLength().coerceAtLeast(1L)
                val total = if (resumed) existing + remaining else remaining
                RandomAccessFile(part, "rw").use { out ->
                    if (resumed) out.seek(existing) else out.setLength(0L)
                    body.byteStream().buffered().use { input ->
                        val buffer = ByteArray(1024 * 1024)
                        var copied = existing
                        while (true) {
                            coroutineContext.ensureActive()
                            val count = input.read(buffer)
                            if (count < 0) break
                            out.write(buffer, 0, count)
                            copied += count
                            progress(start + (end - start) * (copied.toDouble() / total.toDouble()).coerceIn(0.0, 1.0).toFloat())
                        }
                        out.fd.sync()
                    }
                }
                false
            }
            if (shouldRestart) {
                part.delete(); restartAttempted = true; continue
            }
            break
        }
        check(sha256(part).equals(expectedSha, true)) { part.delete(); "Director AI file verification failed. Please retry the download." }
        if (destination.exists()) destination.delete()
        check(part.renameTo(destination)) { "Could not activate the downloaded Director AI model" }
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
