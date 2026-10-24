package com.saunak.director.ai

import android.content.Context
import android.graphics.Bitmap
import android.util.Base64
import com.saunak.director.camera.CameraSnapshot
import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.director.EvaluationOutput
import com.saunak.director.director.PlanOutput
import com.saunak.director.storage.RecipeMemory
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.max

/**
 * NVIDIA NIM runtime for Director using Nemotron 3.5 Super VL.
 * Hosted prototype endpoint: https://integrate.api.nvidia.com/v1/chat/completions
 * Model: nvidia/nemotron-3.5-super-vl-120b-a12b
 */
class NvidiaNemotronVlmRuntime(
    context: Context,
    private val keyStore: SecureApiKeyStore = SecureApiKeyStore(context),
    private val deviceSkill: DeviceConfigSkill = DeviceConfigSkill(context),
    private val recipeMemory: RecipeMemory = RecipeMemory(context)
) : VlmRuntime {
    private val json = Json { ignoreUnknownKeys = true; isLenient = true }
    private val client = OkHttpClient.Builder()
        .connectTimeout(4, TimeUnit.SECONDS)
        .readTimeout(14, TimeUnit.SECONDS)
        .writeTimeout(8, TimeUnit.SECONDS)
        .callTimeout(16, TimeUnit.SECONDS)
        .build()

    @Volatile private var activeCall: Call? = null
    @Volatile private var maxImageDimension = 768

    override suspend fun load(modelPath: String, projectorPath: String): Result<Unit> =
        if (isReady()) Result.success(Unit)
        else Result.failure(IllegalStateException("NVIDIA API key is not configured"))

    override suspend fun plan(
        image: Bitmap,
        capabilities: CapabilityRegistry,
        camera: CameraSnapshot,
        skillContext: String
    ): Result<PlanOutput> = preservingCancellation {
        val config = deviceSkill.readOrGenerate(capabilities, camera)
        val prompt = """
            You are Director, an elite computational-photography agent controlling a real Android camera.
            Analyze this frame for scene, subject, lighting, motion, depth, composition, mood and probable user intent.
            Produce the best achievable camera recipe using ONLY controls allowed by DEVICE CONFIG.
            Prefer natural professional photography. Never invent a control or parameter.

            DEVICE CONFIG:
            $config

            PHOTOGRAPHY SKILL:
            $skillContext

            RECIPE MEMORY:
            ${recipeMemory.promptHints()}

            CURRENT CAMERA STATE:
            ${camera.compactDescription()}

            Return ONE compact JSON object only, no markdown and no explanation:
            {"scene":"short scene description","objective":"professional target look","branch":"general|portrait|landscape|low_light|motion|group|food|architecture|macro|selfie","reference_score":0.0,"reference_accept":false,"calls":[{"id":1,"v":null,"x":0.5,"y":0.5,"s":null}],"recapture":true}

            Rules: scores 0..1; calls max 8; IDs and parameter ranges MUST come from config; use minimal calls when the current image is already strong.
        """.trimIndent()
        VlmContractValidator.plan(json.decodeFromString(generate(prompt, image)))
    }

    override suspend fun evaluate(
        image: Bitmap,
        objective: String,
        attempt: Int,
        camera: CameraSnapshot,
        previousIssues: List<String>,
        evaluationContext: String,
        capabilities: CapabilityRegistry
    ): Result<EvaluationOutput> = preservingCancellation {
        val config = deviceSkill.readOrGenerate(capabilities, camera)
        val prompt = """
            You are Director's strict photography critic. Evaluate this intermediate capture against the objective.
            Judge technical quality, subject rendering, composition, lighting, color, mood and aesthetics.
            Approve only when no material camera correction is needed. Otherwise return small corrected ABSOLUTE target values legal under DEVICE CONFIG.

            ATTEMPT: $attempt of 3
            TARGET: $objective
            PREVIOUS ISSUES: ${previousIssues.joinToString()}
            CURRENT CAMERA: ${camera.compactDescription()}

            DEVICE CONFIG:
            $config

            EVALUATION SKILL:
            $evaluationContext

            Return ONE compact JSON object only, no markdown and no explanation:
            {"score":0.0,"technical":0.0,"composition":0.0,"subject":0.0,"aesthetic":0.0,"accept":false,"issues":["short issue"],"delta":[{"id":3,"v":0.17,"x":null,"y":null,"s":null}]}

            Rules: scores 0..1; issues max 8; delta max 8; if approved set accept=true and delta=[].
        """.trimIndent()
        VlmContractValidator.evaluation(json.decodeFromString(generate(prompt, image)))
    }

    suspend fun validateConnection(): Result<Unit> = preservingCancellation {
        val key = keyStore.read() ?: error("NVIDIA API key is not configured")
        val content = JSONArray().put(JSONObject().put("type", "text").put("text", "Reply with exactly OK"))
        val body = requestBody(content, maxTokens = 32)
        val request = buildRequest(key, body)
        val call = client.newCall(request)
        activeCall = call
        try {
            await(call).use { response ->
                val raw = response.body?.string().orEmpty()
                if (!response.isSuccessful) throw NvidiaApiException.from(response.code, raw)
                extractText(raw)
            }
        } finally {
            if (activeCall === call) activeCall = null
        }
    }

    private suspend fun generate(prompt: String, bitmap: Bitmap): String {
        val key = keyStore.read() ?: error("NVIDIA API key is not configured")
        val imageBytes = encodeForAi(bitmap)
        val dataUrl = "data:image/jpeg;base64," + Base64.encodeToString(imageBytes, Base64.NO_WRAP)

        val content = JSONArray()
            .put(JSONObject().put("type", "text").put("text", prompt))
            .put(
                JSONObject()
                    .put("type", "image_url")
                    .put("image_url", JSONObject().put("url", dataUrl))
            )

        val body = requestBody(content, maxTokens = 1400)
        val request = buildRequest(key, body)
        val call = client.newCall(request)
        activeCall = call
        try {
            await(call).use { response ->
                val raw = response.body?.string().orEmpty()
                if (!response.isSuccessful) throw NvidiaApiException.from(response.code, raw)
                return cleanJson(extractText(raw))
            }
        } finally {
            if (activeCall === call) activeCall = null
        }
    }

    private fun requestBody(content: JSONArray, maxTokens: Int): JSONObject = JSONObject()
        .put("model", MODEL)
        .put(
            "messages",
            JSONArray().put(JSONObject().put("role", "user").put("content", content))
        )
        .put("temperature", 1.0)
        .put("top_p", 0.95)
        .put("max_tokens", maxTokens)
        .put("stream", false)
        .put(
            "chat_template_kwargs",
            JSONObject().put("enable_thinking", false)
        )

    private fun buildRequest(key: String, body: JSONObject): Request = Request.Builder()
        .url(API_URL)
        .header("Authorization", "Bearer $key")
        .header("Accept", "application/json")
        .header("Content-Type", "application/json")
        .post(body.toString().toRequestBody(JSON_MEDIA))
        .build()

    private fun extractText(raw: String): String {
        val parsed = runCatching { JSONObject(raw) }
            .getOrElse { error("NVIDIA returned an unreadable response") }
        val choices = parsed.optJSONArray("choices") ?: error("NVIDIA returned no choices")
        if (choices.length() == 0) error("NVIDIA returned no choices")
        val choice = choices.optJSONObject(0) ?: error("NVIDIA returned an invalid choice")
        val finish = choice.optString("finish_reason").ifBlank { "unknown" }
        val message = choice.optJSONObject("message") ?: error("NVIDIA returned no message")
        val text = message.optString("content").trim()
        if (text.isBlank()) error("NVIDIA returned empty output (finishReason=$finish)")
        return text
    }

    private fun cleanJson(text: String): String {
        val cleaned = text.trim()
            .removePrefix("```json")
            .removePrefix("```")
            .removeSuffix("```")
            .trim()
        if (cleaned.startsWith("{") && cleaned.endsWith("}")) return cleaned
        val start = cleaned.indexOf('{')
        val end = cleaned.lastIndexOf('}')
        if (start >= 0 && end > start) return cleaned.substring(start, end + 1)
        error("NVIDIA returned non-JSON output")
    }

    private suspend fun await(call: Call): Response = suspendCancellableCoroutine { continuation ->
        continuation.invokeOnCancellation { call.cancel() }
        call.enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                if (!continuation.isActive) return
                val mapped = if (e.message?.contains("timeout", ignoreCase = true) == true) {
                    IOException("NVIDIA response timed out", e)
                } else e
                continuation.resumeWithException(mapped)
            }

            override fun onResponse(call: Call, response: Response) {
                if (continuation.isActive) continuation.resume(response) else response.close()
            }
        })
    }

    private fun encodeForAi(source: Bitmap): ByteArray {
        val longest = max(source.width, source.height)
        val prepared = if (longest > maxImageDimension) {
            val scale = maxImageDimension.toFloat() / longest.toFloat()
            Bitmap.createScaledBitmap(
                source,
                maxOf(1, (source.width * scale).toInt()),
                maxOf(1, (source.height * scale).toInt()),
                true
            )
        } else source

        return try {
            ByteArrayOutputStream().use { out ->
                check(prepared.compress(Bitmap.CompressFormat.JPEG, 80, out))
                out.toByteArray()
            }
        } finally {
            if (prepared !== source && !prepared.isRecycled) prepared.recycle()
        }
    }

    private suspend fun <T> preservingCancellation(block: suspend () -> T): Result<T> = try {
        Result.success(block())
    } catch (t: CancellationException) {
        throw t
    } catch (t: Throwable) {
        Result.failure(t)
    }

    override fun setMaxImageDimension(pixels: Int) {
        maxImageDimension = pixels.coerceIn(512, 896)
    }

    override fun cancel() { activeCall?.cancel() }
    override fun isReady(): Boolean = keyStore.hasKey()
    override fun unload() = Unit

    override fun close() {
        activeCall?.cancel()
        client.dispatcher.executorService.shutdown()
    }

    private class NvidiaApiException private constructor(message: String) : IOException(message) {
        companion object {
            fun from(code: Int, raw: String): NvidiaApiException {
                val provider = runCatching {
                    val error = JSONObject(raw).optJSONObject("error")
                    error?.optString("message")
                        ?.replace(Regex("\\s+"), " ")
                        ?.trim()
                        ?.take(220)
                        ?.takeIf { it.isNotBlank() }
                }.getOrNull()

                val base = when (code) {
                    400 -> "NVIDIA rejected the request (400)"
                    401 -> "NVIDIA API key is invalid or expired (401)"
                    402 -> "NVIDIA API requires credits or billing (402)"
                    403 -> "NVIDIA API access is blocked for this key (403)"
                    404 -> "Nemotron model or endpoint is unavailable (404)"
                    408 -> "NVIDIA request timed out (408)"
                    413 -> "Director frame is too large for NVIDIA (413)"
                    429 -> "NVIDIA free-tier rate limit reached (429)"
                    in 500..599 -> "NVIDIA service is temporarily unavailable ($code)"
                    else -> "NVIDIA request failed ($code)"
                }
                return NvidiaApiException(if (provider == null) base else "$base: $provider")
            }
        }
    }

    companion object {
        private const val API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
        private const val MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
        private val JSON_MEDIA = "application/json; charset=utf-8".toMediaType()
    }
}
