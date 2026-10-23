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

/** Gemini cloud runtime used by Director when the user supplies an API key in Settings. */
class GeminiVlmRuntime(
    context: Context,
    private val keyStore: SecureApiKeyStore = SecureApiKeyStore(context),
    private val deviceSkill: DeviceConfigSkill = DeviceConfigSkill(context),
    private val recipeMemory: RecipeMemory = RecipeMemory(context)
) : VlmRuntime {
    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    /*
     * Image reasoning can take longer than a simple text completion, especially on the free tier.
     * The original 6 s total timeout caused valid requests to be cancelled before Gemini answered.
     */
    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(18, TimeUnit.SECONDS)
        .writeTimeout(10, TimeUnit.SECONDS)
        .callTimeout(20, TimeUnit.SECONDS)
        .build()

    @Volatile private var activeCall: Call? = null
    @Volatile private var maxImageDimension = 896

    override suspend fun load(modelPath: String, projectorPath: String): Result<Unit> =
        if (isReady()) Result.success(Unit) else Result.failure(IllegalStateException("Gemini API key is not configured"))

    override suspend fun plan(
        image: Bitmap,
        capabilities: CapabilityRegistry,
        camera: CameraSnapshot,
        skillContext: String
    ): Result<PlanOutput> = preservingCancellation {
        val config = deviceSkill.readOrGenerate(capabilities, camera)
        val prompt = """
            You are Director, an elite computational-photography agent. Understand the scene, context, subject, lighting, motion, depth, composition, mood and probable user intent. Design the best achievable photograph using ONLY the device controls listed below. Prefer natural, professional photographic results. Reuse a cached recipe only when it fits, then adapt it to the current scene. Never invent a control.

            DEVICE CONFIG SKILL:
            $config

            PHOTOGRAPHY SKILL:
            $skillContext

            RECIPE MEMORY:
            ${recipeMemory.promptHints()}

            CURRENT CAMERA STATE:
            ${camera.compactDescription()}

            Return JSON only using exactly this shape:
            {"scene":"short scene description","objective":"professional target look","branch":"general|portrait|landscape|low_light|motion|group|food|architecture|macro|selfie","reference_score":0.0,"reference_accept":false,"calls":[{"id":1,"v":null,"x":0.5,"y":0.5,"s":null}],"recapture":true}
            calls max 8. IDs and parameter ranges MUST come from config.md. If the current image is already excellent, use minimal or no calls.
        """.trimIndent()
        val raw = generate(prompt, image)
        VlmContractValidator.plan(json.decodeFromString(raw))
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
            You are now Director's strict photography critic. Compare this intermediate result with the target objective. Judge technical quality, subject rendering, composition, lighting, color, mood and aesthetic quality. Approve when the photograph is genuinely professional and no material camera correction is needed. Otherwise return corrected absolute target values that are legal under config.md. This is attempt $attempt of maximum 3.

            TARGET: $objective
            PREVIOUS ISSUES: ${previousIssues.joinToString()}
            CAMERA: ${camera.compactDescription()}
            DEVICE CONFIG:
            $config
            EVALUATION SKILL:
            $evaluationContext

            Return JSON only:
            {"score":0.0,"technical":0.0,"composition":0.0,"subject":0.0,"aesthetic":0.0,"accept":false,"issues":["short issue"],"delta":[{"id":3,"v":0.17,"x":null,"y":null,"s":null}]}
            Scores must be 0..1, issues max 8, delta max 8. If approved, set accept=true and delta=[].
        """.trimIndent()
        val raw = generate(prompt, image)
        VlmContractValidator.evaluation(json.decodeFromString(raw))
    }

    /**
     * Lightweight connectivity check used after saving a key. It does not upload a camera frame.
     * This distinguishes an invalid/restricted key from an image-analysis failure.
     */
    suspend fun validateConnection(): Result<Unit> = preservingCancellation {
        val key = keyStore.read() ?: error("Gemini API key is not configured")
        val parts = JSONArray().put(JSONObject().put("text", "Reply with exactly OK"))
        val content = JSONObject().put("role", "user").put("parts", parts)
        val config = baseGenerationConfig(maxTokens = 64, jsonOutput = false)
        val root = JSONObject()
            .put("contents", JSONArray().put(content))
            .put("generationConfig", config)

        val request = buildRequest(key, root)
        val call = client.newCall(request)
        activeCall = call
        try {
            await(call).use { response ->
                val body = response.body?.string().orEmpty()
                if (!response.isSuccessful) throw GeminiApiException.from(response.code, body)
                val parsed = JSONObject(body)
                val candidates = parsed.optJSONArray("candidates")
                    ?: error("Gemini connection test returned no candidate")
                if (candidates.length() == 0) error("Gemini connection test returned no candidate")
            }
        } finally {
            if (activeCall === call) activeCall = null
        }
    }

    private suspend fun <T> preservingCancellation(block: suspend () -> T): Result<T> = try {
        Result.success(block())
    } catch (t: CancellationException) {
        throw t
    } catch (t: Throwable) {
        Result.failure(t)
    }

    private suspend fun generate(prompt: String, bitmap: Bitmap): String {
        val key = keyStore.read() ?: error("Gemini API key is not configured")
        val imageBytes = encodeForAi(bitmap)

        val imageData = JSONObject()
            .put("mime_type", "image/jpeg")
            .put("data", Base64.encodeToString(imageBytes, Base64.NO_WRAP))

        val parts = JSONArray()
            .put(JSONObject().put("text", prompt))
            .put(JSONObject().put("inline_data", imageData))

        val content = JSONObject()
            .put("role", "user")
            .put("parts", parts)

        val root = JSONObject()
            .put("contents", JSONArray().put(content))
            .put("generationConfig", baseGenerationConfig(maxTokens = 4096, jsonOutput = true))

        val request = buildRequest(key, root)
        val call = client.newCall(request)
        activeCall = call
        try {
            await(call).use { response ->
                val body = response.body?.string().orEmpty()
                if (!response.isSuccessful) throw GeminiApiException.from(response.code, body)
                return extractText(body)
            }
        } finally {
            if (activeCall === call) activeCall = null
        }
    }

    private fun baseGenerationConfig(maxTokens: Int, jsonOutput: Boolean): JSONObject = JSONObject().apply {
        // Gemini 3.8 Flash defaults to medium thinking. Low is better for Director's latency-critical loop.
        put("thinkingConfig", JSONObject().put("thinkingLevel", "low"))
        put("maxOutputTokens", maxTokens)
        if (jsonOutput) put("responseMimeType", "application/json")
    }

    private fun buildRequest(key: String, root: JSONObject): Request = Request.Builder()
        .url("https://generativelanguage.googleapis.com/v1beta/models/$MODEL:generateContent")
        .header("x-goog-api-key", key)
        .header("Content-Type", "application/json")
        .post(root.toString().toRequestBody(JSON_MEDIA))
        .build()

    private fun extractText(body: String): String {
        val parsed = runCatching { JSONObject(body) }
            .getOrElse { error("Gemini returned an unreadable response") }
        val candidates = parsed.optJSONArray("candidates")
            ?: error(blockReason(parsed) ?: "Gemini returned no candidate")
        if (candidates.length() == 0) error(blockReason(parsed) ?: "Gemini returned no candidate")

        val candidate = candidates.optJSONObject(0) ?: error("Gemini returned an invalid candidate")
        val finishReason = candidate.optString("finishReason").ifBlank { "unknown" }
        val content = candidate.optJSONObject("content")
            ?: error("Gemini returned no content (finishReason=$finishReason)")
        val partsArray = content.optJSONArray("parts")
            ?: error("Gemini returned no output parts (finishReason=$finishReason)")

        val text = buildString {
            for (i in 0 until partsArray.length()) {
                val part = partsArray.optJSONObject(i) ?: continue
                if (part.has("text")) append(part.optString("text"))
            }
        }
            .trim()
            .removePrefix("```json")
            .removePrefix("```")
            .removeSuffix("```")
            .trim()

        if (text.isBlank()) {
            error("Gemini returned empty output (finishReason=$finishReason)")
        }
        return text
    }

    private fun blockReason(parsed: JSONObject): String? {
        val feedback = parsed.optJSONObject("promptFeedback") ?: return null
        val reason = feedback.optString("blockReason").ifBlank { return null }
        return "Gemini blocked the request ($reason)"
    }

    private suspend fun await(call: Call): Response = suspendCancellableCoroutine { continuation ->
        continuation.invokeOnCancellation { call.cancel() }
        call.enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                if (continuation.isActive) continuation.resumeWithException(
                    if (e.message?.contains("timeout", ignoreCase = true) == true) {
                        IOException("Gemini request timed out", e)
                    } else e
                )
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
        } else {
            source
        }

        return try {
            ByteArrayOutputStream().use { out ->
                check(prepared.compress(Bitmap.CompressFormat.JPEG, 84, out))
                out.toByteArray()
            }
        } finally {
            if (prepared !== source && !prepared.isRecycled) prepared.recycle()
        }
    }

    override fun setMaxImageDimension(pixels: Int) {
        maxImageDimension = pixels.coerceIn(512, 1024)
    }

    override fun cancel() {
        activeCall?.cancel()
    }

    override fun isReady(): Boolean = keyStore.hasKey()
    override fun unload() = Unit

    override fun close() {
        activeCall?.cancel()
        client.dispatcher.executorService.shutdown()
    }

    private class GeminiApiException private constructor(message: String) : IOException(message) {
        companion object {
            fun from(code: Int, body: String): GeminiApiException {
                val provider = runCatching {
                    JSONObject(body)
                        .optJSONObject("error")
                        ?.optString("message")
                        ?.replace(Regex("\\s+"), " ")
                        ?.trim()
                        ?.take(220)
                        ?.takeIf { it.isNotBlank() }
                }.getOrNull()

                val base = when (code) {
                    400 -> "Gemini rejected the request (400)"
                    401 -> "Gemini API key is invalid or expired (401)"
                    403 -> "Gemini API access is blocked for this key/project (403)"
                    404 -> "Gemini model is unavailable for this key/project (404)"
                    408 -> "Gemini request timed out (408)"
                    429 -> "Gemini quota or rate limit reached (429)"
                    in 500..599 -> "Gemini service is temporarily unavailable ($code)"
                    else -> "Gemini request failed ($code)"
                }
                return GeminiApiException(if (provider == null) base else "$base: $provider")
            }
        }
    }

    companion object {
        private const val MODEL = "gemini-3.8-flash"
        private val JSON_MEDIA = "application/json; charset=utf-8".toMediaType()
    }
}
