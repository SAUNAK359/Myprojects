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

/** OpenAI-compatible multimodal runtime with selectable providers and automatic failover. */
class MultiProviderVlmRuntime(
    context: Context,
    private val keyStore: SecureApiKeyStore = SecureApiKeyStore(context),
    private val settings: AiProviderSettings = AiProviderSettings(context),
    private val deviceSkill: DeviceConfigSkill = DeviceConfigSkill(context),
    private val recipeMemory: RecipeMemory = RecipeMemory(context)
) : VlmRuntime {
    private val json = Json { ignoreUnknownKeys = true; isLenient = true }
    private val client = OkHttpClient.Builder()
        .connectTimeout(4, TimeUnit.SECONDS)
        .readTimeout(18, TimeUnit.SECONDS)
        .writeTimeout(8, TimeUnit.SECONDS)
        .callTimeout(20, TimeUnit.SECONDS)
        .build()

    @Volatile private var activeCall: Call? = null
    @Volatile private var maxImageDimension = 768
    @Volatile private var lastSuccessfulRoute: ResolvedAiRoute? = null

    override suspend fun load(modelPath: String, projectorPath: String): Result<Unit> =
        if (isReady()) Result.success(Unit) else Result.failure(IllegalStateException("No AI provider is configured"))

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

    suspend fun validateConnection(preset: AiProviderPreset = settings.selected()): Result<String> = preservingCancellation {
        val routes = routesFor(preset)
        if (routes.isEmpty()) error("${preset.label} is not configured")
        var last: Throwable? = null
        for (route in routes) {
            try {
                request(route, "Reply with exactly OK", image = null, maxTokens = 32)
                lastSuccessfulRoute = route
                return@preservingCancellation route.preset.label
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                last = t
            }
        }
        throw last ?: IOException("AI provider connection failed")
    }

    fun selectedPreset(): AiProviderPreset = settings.selected()
    fun selectPreset(preset: AiProviderPreset) = settings.select(preset)
    fun presets(): List<AiProviderPreset> = AiProviderPreset.selectable
    fun hasCredential(preset: AiProviderPreset): Boolean = preset.credentialId?.let(keyStore::hasKey) ?: isReady()
    fun saveCredential(preset: AiProviderPreset, key: String) {
        val id = preset.credentialId ?: error("Auto mode has no API key")
        keyStore.save(key, id)
    }
    fun clearCredential(preset: AiProviderPreset) { preset.credentialId?.let(keyStore::clear) }
    fun saveOpenRouterCustomModel(model: String) = settings.saveOpenRouterModel(model)
    fun openRouterCustomModel(): String = settings.openRouterCustomModel()
    fun saveCustomRoute(endpoint: String, model: String) = settings.saveCustom(endpoint, model)
    fun customEndpoint(): String = settings.customEndpoint()
    fun customModel(): String = settings.customModel()
    fun activeRouteLabel(): String = lastSuccessfulRoute?.preset?.label ?: settings.selected().label

    private suspend fun generate(prompt: String, bitmap: Bitmap): String {
        val routes = routesFor(settings.selected())
        if (routes.isEmpty()) error("No configured AI provider for ${settings.selected().label}")
        var last: Throwable? = null
        for (route in routes) {
            try {
                val result = request(route, prompt, bitmap, maxTokens = 1400)
                lastSuccessfulRoute = route
                return cleanJson(result)
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                last = t
                if (settings.selected() != AiProviderPreset.AUTO) break
            }
        }
        throw last ?: IOException("All configured AI providers failed")
    }

    private suspend fun request(route: ResolvedAiRoute, prompt: String, image: Bitmap?, maxTokens: Int): String {
        val content = JSONArray().put(JSONObject().put("type", "text").put("text", prompt))
        if (image != null) {
            val dataUrl = "data:image/jpeg;base64," + Base64.encodeToString(encodeForAi(image), Base64.NO_WRAP)
            content.put(JSONObject().put("type", "image_url").put("image_url", JSONObject().put("url", dataUrl)))
        }
        val body = JSONObject()
            .put("model", route.model)
            .put("messages", JSONArray().put(JSONObject().put("role", "user").put("content", content)))
            .put("temperature", 0.6)
            .put("top_p", 0.95)
            .put("max_tokens", maxTokens)
            .put("stream", false)

        when (route.preset) {
            AiProviderPreset.NVIDIA_NEMOTRON -> body.put("chat_template_kwargs", JSONObject().put("enable_thinking", false))
            AiProviderPreset.GEMINI_38_FLASH -> body.put("reasoning_effort", "low")
            else -> Unit
        }
        if (route.preset != AiProviderPreset.CUSTOM_OPENAI) {
            body.put("response_format", JSONObject().put("type", "json_object"))
        }

        val builder = Request.Builder()
            .url(route.endpoint)
            .header("Authorization", "Bearer ${route.apiKey}")
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
        if (route.preset.name.startsWith("OPENROUTER")) builder.header("X-Title", "Director Camera")
        val call = client.newCall(builder.post(body.toString().toRequestBody(JSON_MEDIA)).build())
        activeCall = call
        try {
            await(call).use { response ->
                val raw = response.body?.string().orEmpty()
                if (!response.isSuccessful) throw ProviderApiException.from(route, response.code, raw)
                return extractText(route, raw)
            }
        } finally {
            if (activeCall === call) activeCall = null
        }
    }

    private fun routesFor(preset: AiProviderPreset): List<ResolvedAiRoute> {
        val candidates = if (preset == AiProviderPreset.AUTO) AiProviderPreset.autoPriority else listOf(preset)
        return candidates.mapNotNull(::resolve)
    }

    private fun resolve(preset: AiProviderPreset): ResolvedAiRoute? {
        if (preset == AiProviderPreset.AUTO) return null
        val credentialId = preset.credentialId ?: return null
        val key = keyStore.read(credentialId) ?: return null
        val endpoint = when (preset) {
            AiProviderPreset.CUSTOM_OPENAI -> settings.customEndpoint()
            else -> preset.endpoint.orEmpty()
        }
        val model = when (preset) {
            AiProviderPreset.CUSTOM_OPENAI -> settings.customModel()
            AiProviderPreset.OPENROUTER_CUSTOM -> settings.openRouterCustomModel()
            else -> preset.model.orEmpty()
        }
        if (!endpoint.startsWith("https://") || model.isBlank()) return null
        return ResolvedAiRoute(preset, endpoint, model, key)
    }

    private fun extractText(route: ResolvedAiRoute, raw: String): String {
        val parsed = runCatching { JSONObject(raw) }.getOrElse { error("${route.preset.label} returned unreadable JSON") }
        val choices = parsed.optJSONArray("choices") ?: error("${route.preset.label} returned no choices")
        if (choices.length() == 0) error("${route.preset.label} returned no choices")
        val message = choices.optJSONObject(0)?.optJSONObject("message") ?: error("${route.preset.label} returned no message")
        val content = message.optString("content").trim()
        if (content.isBlank()) error("${route.preset.label} returned empty output")
        return content
    }

    private fun cleanJson(text: String): String {
        val cleaned = text.trim().removePrefix("```json").removePrefix("```").removeSuffix("```").trim()
        if (cleaned.startsWith("{") && cleaned.endsWith("}")) return cleaned
        val start = cleaned.indexOf('{')
        val end = cleaned.lastIndexOf('}')
        if (start >= 0 && end > start) return cleaned.substring(start, end + 1)
        error("AI provider returned non-JSON output")
    }

    private suspend fun await(call: Call): Response = suspendCancellableCoroutine { continuation ->
        continuation.invokeOnCancellation { call.cancel() }
        call.enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                if (continuation.isActive) continuation.resumeWithException(
                    if (e.message?.contains("timeout", true) == true) IOException("AI provider response timed out", e) else e
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
            Bitmap.createScaledBitmap(source, maxOf(1, (source.width * scale).toInt()), maxOf(1, (source.height * scale).toInt()), true)
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

    override fun setMaxImageDimension(pixels: Int) { maxImageDimension = pixels.coerceIn(512, 896) }
    override fun cancel() { activeCall?.cancel() }
    override fun isReady(): Boolean = routesFor(settings.selected()).isNotEmpty()
    override fun unload() = Unit
    override fun close() { activeCall?.cancel(); client.dispatcher.executorService.shutdown() }

    private class ProviderApiException private constructor(message: String) : IOException(message) {
        companion object {
            fun from(route: ResolvedAiRoute, code: Int, raw: String): ProviderApiException {
                val providerMessage = runCatching {
                    JSONObject(raw).optJSONObject("error")?.optString("message")
                        ?.replace(Regex("\\s+"), " ")?.trim()?.take(180)?.takeIf { it.isNotBlank() }
                }.getOrNull()
                val base = when (code) {
                    400 -> "${route.preset.label} rejected the request (400)"
                    401 -> "${route.preset.label} API key is invalid (401)"
                    402 -> "${route.preset.label} requires credits or billing (402)"
                    403 -> "${route.preset.label} access is blocked (403)"
                    404 -> "${route.preset.label} model or endpoint unavailable (404)"
                    408 -> "${route.preset.label} timed out (408)"
                    413 -> "Image too large for ${route.preset.label} (413)"
                    429 -> "${route.preset.label} rate limit reached (429)"
                    in 500..599 -> "${route.preset.label} is temporarily unavailable ($code)"
                    else -> "${route.preset.label} failed ($code)"
                }
                return ProviderApiException(if (providerMessage == null) base else "$base: $providerMessage")
            }
        }
    }

    companion object {
        private val JSON_MEDIA = "application/json; charset=utf-8".toMediaType()
    }
}
