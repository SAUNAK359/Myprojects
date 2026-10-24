package com.saunak.director.engine.models

import android.content.Context
import android.os.StatFs
import com.saunak.director.BuildConfig
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.security.MessageDigest
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request

/** Policy is explicit so a debug/research build can use non-commercial oracle assets without ever
 * allowing them to leak into a production release. */
data class ModelArtifactPolicy(
    val commercialRelease: Boolean = !BuildConfig.DEBUG,
    val maxArtifactBytes: Long = 2L * 1024L * 1024L * 1024L,
    val minimumFreeBytesAfterInstall: Long = 512L * 1024L * 1024L
) {
    init { require(maxArtifactBytes > 0); require(minimumFreeBytesAfterInstall >= 0) }
}

data class InstalledModelArtifact(
    val manifest: ModelManifest,
    val file: File,
    val bytes: Long
)

/**
 * Transactional, checksum-gated model storage. Downloads never become visible until the complete
 * artifact has passed SHA-256 verification and an atomic/same-directory move succeeds.
 */
class ModelArtifactStore(
    context: Context,
    private val client: OkHttpClient = OkHttpClient(),
    private val policy: ModelArtifactPolicy = ModelArtifactPolicy()
) {
    private val root = File(context.noBackupFilesDir, "sierra/models").apply { mkdirs() }

    suspend fun ensure(manifest: ModelManifest): Result<InstalledModelArtifact> = withContext(Dispatchers.IO) {
        runCatching {
            validatePolicy(manifest)
            val target = artifactFile(manifest)
            if (target.isFile) {
                if (target.length() <= policy.maxArtifactBytes && sha256(target).equals(manifest.sha256, ignoreCase = true)) {
                    return@runCatching InstalledModelArtifact(manifest, target, target.length())
                }
                target.delete()
            }

            val url = requireNotNull(manifest.sourceUrl) { "No artifact URL provisioned for ${manifest.id}" }
            require(url.startsWith("https://")) { "Model artifacts must use HTTPS" }
            val part = File(target.parentFile, target.name + ".part")
            if (part.exists()) part.delete()
            target.parentFile?.mkdirs()

            val request = Request.Builder().url(url).get().build()
            client.newCall(request).execute().use { response ->
                require(response.isSuccessful) { "Model download failed: HTTP ${response.code}" }
                val body = requireNotNull(response.body) { "Model download returned no body" }
                val declared = body.contentLength()
                require(declared < 0 || declared <= policy.maxArtifactBytes) { "Model artifact exceeds configured size limit" }
                ensureFreeSpace(if (declared > 0) declared else 0L)

                var written = 0L
                FileOutputStream(part).use { output ->
                    body.byteStream().use { input ->
                        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                        while (true) {
                            val count = input.read(buffer)
                            if (count < 0) break
                            written += count
                            require(written <= policy.maxArtifactBytes) { "Model artifact exceeded configured size limit while streaming" }
                            output.write(buffer, 0, count)
                        }
                        output.fd.sync()
                    }
                }
            }

            ensureFreeSpace(0L)
            require(sha256(part).equals(manifest.sha256, ignoreCase = true)) { "SHA-256 mismatch for ${manifest.id}" }
            moveIntoPlace(part, target)
            InstalledModelArtifact(manifest, target, target.length())
        }
    }

    fun findVerified(manifest: ModelManifest): InstalledModelArtifact? {
        return runCatching {
            validatePolicy(manifest)
            val file = artifactFile(manifest)
            if (!file.isFile || file.length() > policy.maxArtifactBytes) return null
            if (!sha256(file).equals(manifest.sha256, ignoreCase = true)) return null
            InstalledModelArtifact(manifest, file, file.length())
        }.getOrNull()
    }

    fun remove(manifest: ModelManifest): Boolean {
        val file = artifactFile(manifest)
        val part = File(file.parentFile, file.name + ".part")
        part.delete()
        return !file.exists() || file.delete()
    }

    fun pruneExcept(allowed: Set<Pair<String, String>>) {
        root.listFiles()?.forEach { modelDir ->
            if (!modelDir.isDirectory) return@forEach
            modelDir.listFiles()?.forEach { file ->
                val key = modelDir.name to file.name.substringBeforeLast('.')
                if (key !in allowed) file.delete()
            }
            if (modelDir.listFiles().isNullOrEmpty()) modelDir.delete()
        }
    }

    private fun validatePolicy(manifest: ModelManifest) {
        if (policy.commercialRelease) {
            require(manifest.allowedForCommercialRelease()) {
                "${manifest.id} weights are not approved for commercial redistribution"
            }
        }
    }

    private fun artifactFile(manifest: ModelManifest): File {
        require(manifest.id.matches(Regex("[a-z0-9_.-]+")))
        require(manifest.version.matches(Regex("[A-Za-z0-9_.-]+")))
        val dir = File(root, manifest.id).apply { mkdirs() }
        val candidate = File(dir, "${manifest.version}.onnx")
        require(candidate.canonicalPath.startsWith(root.canonicalPath + File.separator)) { "Unsafe model path" }
        return candidate
    }

    private fun ensureFreeSpace(incomingBytes: Long) {
        val stat = StatFs(root.absolutePath)
        val needed = incomingBytes + policy.minimumFreeBytesAfterInstall
        require(stat.availableBytes >= needed) { "Insufficient free storage for model artifact" }
    }

    private fun moveIntoPlace(from: File, to: File) {
        try {
            Files.move(from.toPath(), to.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        } catch (_: Throwable) {
            Files.move(from.toPath(), to.toPath(), StandardCopyOption.REPLACE_EXISTING)
        }
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
