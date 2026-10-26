package com.saunak.director.ai

import android.content.Context
import java.io.IOException
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject

/**
 * Data-driven Gemini transport configuration plus capability-checked model resolution.
 * No provider model id is compiled into Kotlin. If discovery cannot prove that an installed
 * preference is currently available, Sierra fails closed to its local photographic path.
 */
data class GeminiProviderConfig(
    val interactionsUrl: String,
    val modelsUrl: String,
    val preferredModels: List<String>,
    val resolutionTtlMs: Long,
    val connectTimeoutMs: Long,
    val readTimeoutMs: Long,
    val writeTimeoutMs: Long,
    val callTimeoutMs: Long,
    val maxOutputTokens: Int,
    val thinkingLevel: String
) {
    init {
        require(interactionsUrl.startsWith("https://") && modelsUrl.startsWith("https://"))
        require(preferredModels.isNotEmpty() && preferredModels.all { it.isNotBlank() })
        require(resolutionTtlMs > 0)
        require(connectTimeoutMs > 0 && readTimeoutMs > 0 && writeTimeoutMs > 0 && callTimeoutMs > 0)
        require(maxOutputTokens in 128..8192)
        require(thinkingLevel in setOf("minimal", "low", "medium", "high"))
    }

    companion object {
        private const val ASSET = "sierra/gemini_provider_v1.json"
        private const val SCHEMA = 1

        fun load(context: Context): GeminiProviderConfig {
            val raw = context.applicationContext.assets.open(ASSET).bufferedReader().use { it.readText() }
            val json = JSONObject(raw)
            require(json.getInt("schema_version") == SCHEMA) { "Unsupported Gemini provider schema" }
            val preferred = json.getJSONArray("preferred_models")
            return GeminiProviderConfig(
                interactionsUrl = json.getString("interactions_url").trim().removeSuffix("/"),
                modelsUrl = json.getString("models_url").trim(),
                preferredModels = List(preferred.length()) { index -> preferred.getString(index).trim() },
                resolutionTtlMs = json.getLong("resolution_ttl_ms"),
                connectTimeoutMs = json.getLong("connect_timeout_ms"),
                readTimeoutMs = json.getLong("read_timeout_ms"),
                writeTimeoutMs = json.getLong("write_timeout_ms"),
                callTimeoutMs = json.getLong("call_timeout_ms"),
                maxOutputTokens = json.getInt("max_output_tokens"),
                thinkingLevel = json.getString("thinking_level").trim()
            )
        }
    }
}

class GeminiModelResolver(
    private val config: GeminiProviderConfig,
    private val client: OkHttpClient
) {
    private data class Cached(val model: String, val expiresElapsedMs: Long)
    private val cached = AtomicReference<Cached?>(null)

    @Throws(IOException::class)
    fun resolve(apiKey: String, nowElapsedMs: Long = android.os.SystemClock.elapsedRealtime()): String {
        cached.get()?.takeIf { nowElapsedMs < it.expiresElapsedMs }?.let { return it.model }
        val request = Request.Builder()
            .url(config.modelsUrl)
            .header("x-goog-api-key", apiKey)
            .get()
            .build()
        val available = client.newCall(request).execute().use { response ->
            val payload = response.body?.string().orEmpty()
            if (!response.isSuccessful) throw IOException(providerError(response.code, payload))
            val models = JSONObject(payload).optJSONArray("models") ?: throw IOException("Gemini model discovery returned no models")
            buildSet {
                for (i in 0 until models.length()) {
                    val model = models.optJSONObject(i) ?: continue
                    val name = model.optString("name").removePrefix("models/").trim()
                    if (name.isNotBlank()) add(name)
                }
            }
        }
        val resolved = config.preferredModels.firstOrNull(available::contains)
            ?: throw IOException("No configured Sierra Gemini model is currently available")
        cached.set(Cached(resolved, nowElapsedMs + config.resolutionTtlMs))
        return resolved
    }

    fun invalidate() { cached.set(null) }

    private fun providerError(code: Int, payload: String): String =
        runCatching { JSONObject(payload).optJSONObject("error")?.optString("message") }
            .getOrNull()?.takeIf(String::isNotBlank)?.take(180)
            ?: "Gemini model discovery failed ($code)"
}

fun GeminiProviderConfig.newHttpClient(): OkHttpClient = OkHttpClient.Builder()
    .connectTimeout(connectTimeoutMs, TimeUnit.MILLISECONDS)
    .readTimeout(readTimeoutMs, TimeUnit.MILLISECONDS)
    .writeTimeout(writeTimeoutMs, TimeUnit.MILLISECONDS)
    .callTimeout(callTimeoutMs, TimeUnit.MILLISECONDS)
    .retryOnConnectionFailure(true)
    .build()
