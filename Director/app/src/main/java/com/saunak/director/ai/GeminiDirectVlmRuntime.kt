package com.saunak.director.ai

import android.content.Context
import android.graphics.Bitmap
import android.util.Base64
import com.saunak.director.camera.CameraSnapshot
import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.director.EvaluationOutput
import com.saunak.director.director.PlanOutput
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.math.max
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import okhttp3.Call
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject

/**
 * Beta direct Gemini planner used when the user explicitly configures a Gemini API key in-app.
 * Camera physics/safety remain local: Gemini can only propose the existing constrained PlanOutput.
 */
class GeminiDirectVlmRuntime(
    context: Context,
    private val keyStore: GeminiApiKeyStore = GeminiApiKeyStore(context),
    private val promptBuilder: PromptBuilder = PromptBuilder()
) : VlmRuntime {
    private val json = Json { ignoreUnknownKeys = true; explicitNulls = false }
    private val client = OkHttpClient.Builder()
        .connectTimeout(2, TimeUnit.SECONDS)
        .readTimeout(4, TimeUnit.SECONDS)
        .writeTimeout(4, TimeUnit.SECONDS)
        .callTimeout(5, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .build()

    @Volatile private var maxImageDimension = 704
    @Volatile private var activeCall: Call? = null

    override suspend fun load(modelPath: String, projectorPath: String): Result<Unit> =
        if (isReady()) Result.success(Unit) else Result.failure(IllegalStateException("Add a Gemini API key in Sierra settings"))

    override suspend fun plan(
        image: Bitmap,
        capabilities: CapabilityRegistry,
        camera: CameraSnapshot,
        skillContext: String
    ): Result<PlanOutput> = withContext(Dispatchers.IO) {
        runCatching {
            val key = keyStore.read().takeIf(String::isNotBlank)
                ?: throw IllegalStateException("Gemini API key is not configured")
            val prompt = promptBuilder.planPrompt(capabilities, camera, skillContext)
            val body = buildRequestBody(prompt, encode(image))
            val request = Request.Builder()
                .url("$API_ROOT/models/$MODEL:generateContent")
                .header("x-goog-api-key", key)
                .header("Content-Type", "application/json")
                .post(body.toString().toRequestBody(JSON_MEDIA))
                .build()
            val call = client.newCall(request)
            activeCall = call
            try {
                call.execute().use { response ->
                    val payload = response.body?.string().orEmpty()
                    if (!response.isSuccessful) throw IOException(apiError(response.code, payload))
                    val text = extractText(payload)
                    VlmContractValidator.plan(json.decodeFromString<PlanOutput>(text))
                }
            } finally {
                if (activeCall === call) activeCall = null
            }
        }
    }

    override suspend fun evaluate(
        image: Bitmap,
        objective: String,
        attempt: Int,
        camera: CameraSnapshot,
        previousIssues: List<String>,
        evaluationContext: String,
        capabilities: CapabilityRegistry
    ): Result<EvaluationOutput> = Result.failure(
        UnsupportedOperationException("Sierra v10 final technical QA remains local")
    )

    override fun setMaxImageDimension(pixels: Int) {
        maxImageDimension = pixels.coerceIn(384, 896)
    }

    override fun cancel() {
        activeCall?.cancel()
        activeCall = null
    }

    override fun isReady(): Boolean = keyStore.isConfigured()
    override fun unload() = cancel()
    override fun close() = cancel()

    private fun buildRequestBody(prompt: String, image: ByteArray): JSONObject {
        val parts = JSONArray()
            .put(JSONObject().put("text", prompt))
            .put(
                JSONObject().put(
                    "inlineData",
                    JSONObject()
                        .put("mimeType", "image/jpeg")
                        .put("data", Base64.encodeToString(image, Base64.NO_WRAP))
                )
            )
        return JSONObject()
            .put("contents", JSONArray().put(JSONObject().put("role", "user").put("parts", parts)))
            .put(
                "generationConfig",
                JSONObject()
                    .put("responseMimeType", "application/json")
                    .put("thinkingConfig", JSONObject().put("thinkingLevel", "low"))
            )
    }

    private fun extractText(payload: String): String {
        val root = JSONObject(payload)
        val candidates = root.optJSONArray("candidates")
            ?: throw IOException("Gemini returned no candidates")
        if (candidates.length() == 0) throw IOException("Gemini returned an empty candidate list")
        val parts = candidates.getJSONObject(0).optJSONObject("content")?.optJSONArray("parts")
            ?: throw IOException("Gemini returned no content")
        val text = buildString {
            for (i in 0 until parts.length()) {
                val value = parts.optJSONObject(i)?.optString("text").orEmpty()
                if (value.isNotBlank()) append(value)
            }
        }.trim()
        if (text.isBlank()) throw IOException("Gemini returned no JSON plan")
        return text
    }

    private fun encode(source: Bitmap): ByteArray {
        val longest = max(source.width, source.height)
        val output = if (longest > maxImageDimension) {
            val scale = maxImageDimension.toFloat() / longest.toFloat()
            Bitmap.createScaledBitmap(
                source,
                (source.width * scale).toInt().coerceAtLeast(1),
                (source.height * scale).toInt().coerceAtLeast(1),
                true
            )
        } else source
        return try {
            ByteArrayOutputStream().use { stream ->
                check(output.compress(Bitmap.CompressFormat.JPEG, 78, stream)) { "Could not encode Gemini analysis frame" }
                stream.toByteArray()
            }
        } finally {
            if (output !== source && !output.isRecycled) output.recycle()
        }
    }

    private fun apiError(code: Int, payload: String): String {
        val message = runCatching {
            JSONObject(payload).optJSONObject("error")?.optString("message")
        }.getOrNull()?.takeIf { it.isNotBlank() }
        return message?.take(180) ?: "Gemini request failed ($code)"
    }

    companion object {
        private const val API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
        private const val MODEL = "gemini-3.8-flash"
        private val JSON_MEDIA = "application/json; charset=utf-8".toMediaType()
    }
}
