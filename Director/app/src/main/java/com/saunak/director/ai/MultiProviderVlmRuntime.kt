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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
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
    @Volatile private var maxImageDimension = 640
    @Volatile private var lastSuccessfulRoute: ResolvedAiRoute? = null

    override suspend fun load(modelPath: String, projectorPath: String): Result<Unit> =
        if (isReady()) Result.success(Unit) else Result.failure(IllegalStateException("No AI provider is configured"))

    override suspend fun plan(
        image: Bitmap,
        capabilities: CapabilityRegistry,
        camera: CameraSnapshot,
        skillContext: String
    ): Result<PlanOutput> = preservingCancellation {
        val config = withContext(Dispatchers.IO) { deviceSkill.readOrGenerate(capabilities, camera) }
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

            Return exactly the requested Director plan object. All top-level fields must be present.
            Rules: scores 0..1; calls max 8; IDs and parameter ranges MUST come from config; use minimal calls when the current image is already strong.
        """.trimIndent()
        val raw = generate(prompt, image, OutputKind.PLAN)
        withContext(Dispatchers.Default) {
            val normalized = normalizePlanJson(raw)
            VlmContractValidator.plan(json.decodeFromString(normalized))
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
    ): Result<EvaluationOutput> = preservingCancellation {
        val config = withContext(Dispatchers.IO) { deviceSkill.readOrGenerate(capabilities, camera) }
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

            Return exactly the requested Director evaluation object. All top-level fields must be present.
            Rules: scores 0..1; issues max 8; delta max 8; if approved set accept=true and delta=[].
        """.trimIndent()
        val raw = generate(prompt, image, OutputKind.EVALUATION)
        withContext(Dispatchers.Default) {
            val normalized = normalizeEvaluationJson(raw)
            VlmContractValidator.evaluation(json.decodeFromString(normalized))
        }
    }

    suspend fun validateConnection(preset: AiProviderPreset = settings.selected()): Result<String> = preservingCancellation {
        val routes = withContext(Dispatchers.IO) { routesFor(preset) }
        if (routes.isEmpty()) error("${preset.label} is not configured")
        var last: Throwable? = null
        for (route in routes) {
            try {
                request(route, "Reply with exactly OK", imageDataUrl = null, maxTokens = 32, outputKind = null)
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

    private suspend fun generate(prompt: String, bitmap: Bitmap, outputKind: OutputKind): String {
        val imageDataUrl = withContext(Dispatchers.Default) {
            val jpeg = encodeForAi(bitmap)
            "data:image/jpeg;base64," + Base64.encodeToString(jpeg, Base64.NO_WRAP)
        }
        val routes = withContext(Dispatchers.IO) { routesFor(settings.selected()) }
        if (routes.isEmpty()) error("No configured AI provider for ${settings.selected().label}")

        var last: Throwable? = null
        for (route in routes) {
            try {
                val result = request(route, prompt, imageDataUrl, maxTokens = 1400, outputKind = outputKind)
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

    private suspend fun request(
        route: ResolvedAiRoute,
        prompt: String,
        imageDataUrl: String?,
        maxTokens: Int,
        outputKind: OutputKind?
    ): String {
        val request = withContext(Dispatchers.Default) {
            val content = JSONArray().put(JSONObject().put("type", "text").put("text", prompt))
            if (imageDataUrl != null) {
                content.put(
                    JSONObject()
                        .put("type", "image_url")
                        .put("image_url", JSONObject().put("url", imageDataUrl))
                )
            }
            val body = JSONObject()
                .put("model", route.model)
                .put("messages", JSONArray().put(JSONObject().put("role", "user").put("content", content)))
                .put("temperature", if (outputKind == null) 0.2 else 0.35)
                .put("top_p", 0.95)
                .put("max_tokens", maxTokens)
                .put("stream", false)

            when (route.preset) {
                AiProviderPreset.NVIDIA_NEMOTRON -> body.put("chat_template_kwargs", JSONObject().put("enable_thinking", false))
                AiProviderPreset.GEMINI_38_FLASH -> body.put("reasoning_effort", "low")
                else -> Unit
            }

            if (outputKind != null && route.preset != AiProviderPreset.CUSTOM_OPENAI) {
                body.put("response_format", responseFormat(route, outputKind))
            }

            Request.Builder()
                .url(route.endpoint)
                .header("Authorization", "Bearer ${route.apiKey}")
                .header("Accept", "application/json")
                .header("Content-Type", "application/json")
                .apply {
                    if (route.preset.name.startsWith("OPENROUTER")) header("X-Title", "Director Camera")
                }
                .post(body.toString().toRequestBody(JSON_MEDIA))
                .build()
        }

        val call = client.newCall(request)
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

    private fun responseFormat(route: ResolvedAiRoute, kind: OutputKind): JSONObject {
        val strictSchemaSupported = route.preset == AiProviderPreset.NVIDIA_NEMOTRON ||
            route.preset == AiProviderPreset.OPENROUTER_NEMOTRON_FREE ||
            route.preset == AiProviderPreset.OPENROUTER_NEMOTRON ||
            route.preset == AiProviderPreset.GEMINI_38_FLASH

        if (!strictSchemaSupported) return JSONObject().put("type", "json_object")
        return JSONObject()
            .put("type", "json_schema")
            .put(
                "json_schema",
                JSONObject()
                    .put("name", if (kind == OutputKind.PLAN) "director_plan" else "director_evaluation")
                    .put("strict", true)
                    .put("schema", if (kind == OutputKind.PLAN) planSchema() else evaluationSchema())
            )
    }

    private fun planSchema(): JSONObject {
        val tool = toolCallSchema()
        return JSONObject()
            .put("type", "object")
            .put("additionalProperties", false)
            .put(
                "properties",
                JSONObject()
                    .put("scene", JSONObject().put("type", "string"))
                    .put("objective", JSONObject().put("type", "string"))
                    .put(
                        "branch",
                        JSONObject().put("type", "string").put(
                            "enum",
                            JSONArray(listOf("general", "portrait", "landscape", "low_light", "motion", "group", "food", "architecture", "macro", "selfie"))
                        )
                    )
                    .put("reference_score", JSONObject().put("type", "number"))
                    .put("reference_accept", JSONObject().put("type", "boolean"))
                    .put("calls", JSONObject().put("type", "array").put("maxItems", 8).put("items", tool))
                    .put("recapture", JSONObject().put("type", "boolean"))
            )
            .put("required", JSONArray(listOf("scene", "objective", "branch", "reference_score", "reference_accept", "calls", "recapture")))
    }

    private fun evaluationSchema(): JSONObject {
        return JSONObject()
            .put("type", "object")
            .put("additionalProperties", false)
            .put(
                "properties",
                JSONObject()
                    .put("score", JSONObject().put("type", "number"))
                    .put("technical", JSONObject().put("type", "number"))
                    .put("composition", JSONObject().put("type", "number"))
                    .put("subject", JSONObject().put("type", "number"))
                    .put("aesthetic", JSONObject().put("type", "number"))
                    .put("accept", JSONObject().put("type", "boolean"))
                    .put("issues", JSONObject().put("type", "array").put("maxItems", 8).put("items", JSONObject().put("type", "string")))
                    .put("delta", JSONObject().put("type", "array").put("maxItems", 8).put("items", toolCallSchema()))
            )
            .put("required", JSONArray(listOf("score", "technical", "composition", "subject", "aesthetic", "accept", "issues", "delta")))
    }

    private fun toolCallSchema(): JSONObject = JSONObject()
        .put("type", "object")
        .put("additionalProperties", false)
        .put(
            "properties",
            JSONObject()
                .put("id", JSONObject().put("type", "integer"))
                .put("v", nullableSchema("number"))
                .put("x", nullableSchema("number"))
                .put("y", nullableSchema("number"))
                .put("s", nullableSchema("string"))
        )
        .put("required", JSONArray(listOf("id", "v", "x", "y", "s")))

    private fun nullableSchema(type: String): JSONObject = JSONObject().put(
        "anyOf",
        JSONArray().put(JSONObject().put("type", type)).put(JSONObject().put("type", "null"))
    )

    private fun normalizePlanJson(raw: String): String {
        val obj = parseObject(raw)
        val scene = obj.optString("scene").trim()
        val objective = obj.optString("objective").trim()
        require(scene.isNotBlank()) { "Invalid scene" }
        require(objective.isNotBlank()) { "Invalid objective" }

        if (obj.optString("branch").isBlank()) obj.put("branch", "general")
        if (!obj.has("reference_score") || obj.isNull("reference_score")) obj.put("reference_score", 0.5)
        if (!obj.has("reference_accept") || obj.isNull("reference_accept")) obj.put("reference_accept", false)
        if (!obj.has("calls") || obj.optJSONArray("calls") == null) obj.put("calls", JSONArray())
        if (!obj.has("recapture") || obj.isNull("recapture")) obj.put("recapture", true)
        normalizeToolArray(obj.optJSONArray("calls"))
        return obj.toString()
    }

    private fun normalizeEvaluationJson(raw: String): String {
        val obj = parseObject(raw)
        val score = obj.optDouble("score", Double.NaN)
        require(score.isFinite()) { "Invalid evaluation score" }
        if (!obj.has("technical") || obj.isNull("technical")) obj.put("technical", score)
        if (!obj.has("composition") || obj.isNull("composition")) obj.put("composition", score)
        if (!obj.has("subject") || obj.isNull("subject")) obj.put("subject", score)
        if (!obj.has("aesthetic") || obj.isNull("aesthetic")) obj.put("aesthetic", score)
        if (!obj.has("accept") || obj.isNull("accept")) obj.put("accept", false)
        if (!obj.has("issues") || obj.optJSONArray("issues") == null) obj.put("issues", JSONArray())
        if (!obj.has("delta") || obj.optJSONArray("delta") == null) obj.put("delta", JSONArray())
        normalizeToolArray(obj.optJSONArray("delta"))
        return obj.toString()
    }

    private fun normalizeToolArray(array: JSONArray?) {
        if (array == null) return
        for (i in 0 until array.length()) {
            val call = array.optJSONObject(i) ?: continue
            if (!call.has("v")) call.put("v", JSONObject.NULL)
            if (!call.has("x")) call.put("x", JSONObject.NULL)
            if (!call.has("y")) call.put("y", JSONObject.NULL)
            if (!call.has("s")) call.put("s", JSONObject.NULL)
        }
    }

    private fun parseObject(raw: String): JSONObject = runCatching { JSONObject(cleanJson(raw)) }
        .getOrElse { throw IllegalArgumentException("AI returned an invalid analysis response", it) }

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

    private fun isConfigured(preset: AiProviderPreset): Boolean {
        if (preset == AiProviderPreset.AUTO) return AiProviderPreset.autoPriority.any(::isConfigured)
        val id = preset.credentialId ?: return false
        if (!keyStore.hasKey(id)) return false
        val endpoint = when (preset) {
            AiProviderPreset.CUSTOM_OPENAI -> settings.customEndpoint()
            else -> preset.endpoint.orEmpty()
        }
        val model = when (preset) {
            AiProviderPreset.CUSTOM_OPENAI -> settings.customModel()
            AiProviderPreset.OPENROUTER_CUSTOM -> settings.openRouterCustomModel()
            else -> preset.model.orEmpty()
        }
        return endpoint.startsWith("https://") && model.isNotBlank()
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
        require(!source.isRecycled) { "Analysis frame is no longer available" }
        val longest = max(source.width, source.height)
        require(longest > 0) { "Analysis frame is invalid" }
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
            ByteArrayOutputStream(192 * 1024).use { out ->
                check(prepared.compress(Bitmap.CompressFormat.JPEG, 72, out)) { "Could not encode analysis frame" }
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
    } catch (oom: OutOfMemoryError) {
        Result.failure(IOException("Director ran out of memory preparing the AI frame", oom))
    } catch (t: Throwable) {
        Result.failure(t)
    }

    override fun setMaxImageDimension(pixels: Int) { maxImageDimension = pixels.coerceIn(512, 704) }
    override fun cancel() { activeCall?.cancel() }
    override fun isReady(): Boolean = isConfigured(settings.selected())
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

    private enum class OutputKind { PLAN, EVALUATION }

    companion object {
        private val JSON_MEDIA = "application/json; charset=utf-8".toMediaType()
    }
}
