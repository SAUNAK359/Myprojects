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
import kotlin.math.max
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import okhttp3.Call
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject

/** Direct Gemini Interactions runtime with stored device foundation and structured SSE streaming. */
class GeminiDirectVlmRuntime(
    context: Context,
    private val keyStore: GeminiApiKeyStore = GeminiApiKeyStore(context),
    private val promptBuilder: PromptBuilder = PromptBuilder()
) : VlmRuntime {
    private val appContext = context.applicationContext
    private val json = Json { ignoreUnknownKeys = false; explicitNulls = false }
    private val themeCatalog = ThemePromptCatalog(appContext)
    private val provider = GeminiProviderConfig.load(appContext)
    private val client = provider.newHttpClient()
    private val modelResolver = GeminiModelResolver(provider, client)
    private val foundationPrefs = appContext.getSharedPreferences("sierra_device_foundation_v1", Context.MODE_PRIVATE)
    private val foundationMutex = Mutex()

    @Volatile private var maxImageDimension = 512
    @Volatile private var activeCall: Call? = null
    @Volatile private var lastDiagnostics = VlmDiagnostics("sierra-unresolved")
    @Volatile private var lastFoundationProfile: DeviceFoundationProfile? = null
    @Volatile private var progressListener: ((VlmStreamProgress) -> Unit)? = null
    @Volatile private var lastPlanInteractionId: String? = null

    override suspend fun load(modelPath: String, projectorPath: String): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            val key = keyStore.read().takeIf(String::isNotBlank) ?: error("Add the Sierra Intelligence API key in Sierra settings")
            modelResolver.resolve(key)
        }.map { Unit }
    }

    override suspend fun syncDeviceFoundation(profile: DeviceFoundationProfile): Result<Unit> = withContext(Dispatchers.IO) {
        foundationMutex.withLock {
            lastFoundationProfile = profile
            if (foundationIsFresh(profile)) Result.success(Unit) else createFoundation(profile)
        }
    }

    private fun foundationIsFresh(profile: DeviceFoundationProfile): Boolean {
        val storedDigest = foundationPrefs.getString(KEY_FOUNDATION_DIGEST, null)
        val storedId = foundationPrefs.getString(KEY_FOUNDATION_ID, null)
        val storedAt = foundationPrefs.getLong(KEY_FOUNDATION_AT, 0L)
        return storedDigest == profile.digest && !storedId.isNullOrBlank() &&
            System.currentTimeMillis() - storedAt in 0 until FOUNDATION_REFRESH_MS
    }

    private fun createFoundation(profile: DeviceFoundationProfile): Result<Unit> {
        val started = android.os.SystemClock.elapsedRealtime()
        var resolvedModel = "sierra-unresolved"
        return runCatching {
            val key = keyStore.read().takeIf(String::isNotBlank) ?: error("Sierra Intelligence API key is not configured")
            resolvedModel = modelResolver.resolve(key)
            val body = JSONObject()
                .put("model", resolvedModel)
                .put("input", """
                    You are Sierra, the photographic authority and final colorist inside Sierra Vision.
                    Store this DEVICE FOUNDATION as the hard physical/API/render-executor boundary for future camera plans and final render specifications.
                    Sierra beta may enforce physical feasibility but must not creatively replace Sierra's valid decisions.
                    Device limits change execution, not photographic intent. Never invent unsupported camera or render capabilities.
                    Preserve identity, geometry, text, material truth and scene evidence.

                    ${profile.payload}
                """.trimIndent())
                .put("generation_config", JSONObject().put("thinking_level", provider.thinkingLevel).put("max_output_tokens", 128))
                .put("store", true)
            val request = baseRequest(key).post(body.toString().toRequestBody(JSON_MEDIA)).build()
            val call = client.newCall(request)
            activeCall = call
            try {
                call.execute().use { response ->
                    val payload = response.body?.string().orEmpty()
                    if (!response.isSuccessful) throw IOException(apiError(response.code, payload))
                    val id = JSONObject(payload).optString("id").takeIf(String::isNotBlank)
                        ?: throw IOException("Sierra device foundation returned no interaction id")
                    foundationPrefs.edit().putString(KEY_FOUNDATION_ID, id).putString(KEY_FOUNDATION_DIGEST, profile.digest)
                        .putLong(KEY_FOUNDATION_AT, System.currentTimeMillis()).apply()
                    lastPlanInteractionId = null
                }
            } finally { if (activeCall === call) activeCall = null }
        }.onSuccess {
            lastDiagnostics = VlmDiagnostics(resolvedModel, android.os.SystemClock.elapsedRealtime() - started, null)
        }.onFailure {
            lastDiagnostics = VlmDiagnostics(resolvedModel, android.os.SystemClock.elapsedRealtime() - started, it.message?.take(160))
        }
    }

    override suspend fun plan(image: Bitmap, capabilities: CapabilityRegistry, camera: CameraSnapshot, skillContext: String): Result<PlanOutput> {
        suspend fun execute(): Result<PlanOutput> = streamRequest(
            promptBuilder.planPrompt(capabilities, camera, skillContext, themeCatalog.compactDescription()), image,
            SierraGeminiSchemas.plan(), currentFoundationId(),
            onInteractionCompleted = { lastPlanInteractionId = it }
        ) { raw -> VlmContractValidator.plan(json.decodeFromString<PlanOutput>(raw)).also(::validateThemeReferences) }

        val first = execute()
        if (first.isSuccess || !foundationContextFailure(first.exceptionOrNull())) return first
        val profile = lastFoundationProfile ?: return first
        val rebuilt = withContext(Dispatchers.IO) { foundationMutex.withLock { clearFoundation(); createFoundation(profile) } }
        if (rebuilt.isFailure) return Result.failure(rebuilt.exceptionOrNull() ?: IllegalStateException("Device foundation rebuild failed"))
        return execute()
    }

    override suspend fun evaluate(
        image: Bitmap,
        objective: String,
        attempt: Int,
        camera: CameraSnapshot,
        previousIssues: List<String>,
        evaluationContext: String,
        capabilities: CapabilityRegistry
    ): Result<EvaluationOutput> = streamRequest(
        promptBuilder.evaluationPrompt(objective, attempt, camera, previousIssues, evaluationContext, capabilities), image,
        SierraGeminiSchemas.evaluation(), lastPlanInteractionId ?: currentFoundationId()
    ) { raw -> VlmContractValidator.evaluation(json.decodeFromString<EvaluationOutput>(raw)) }

    private suspend fun <T> streamRequest(
        prompt: String,
        image: Bitmap,
        schema: JSONObject,
        previousInteractionId: String?,
        onInteractionCompleted: (String?) -> Unit = {},
        decode: (String) -> T
    ): Result<T> = withContext(Dispatchers.IO) {
        val started = android.os.SystemClock.elapsedRealtime()
        var resolvedModel = "sierra-unresolved"
        runCatching {
            val key = keyStore.read().takeIf(String::isNotBlank) ?: error("Sierra Intelligence API key is not configured")
            resolvedModel = modelResolver.resolve(key)
            val body = buildInteractionBody(resolvedModel, prompt, encode(image), schema, previousInteractionId).put("stream", true)
            val request = baseRequest(key).header("Accept", "text/event-stream").post(body.toString().toRequestBody(JSON_MEDIA)).build()
            val streamer = GeminiInteractionStreamer(
                client = client,
                onProgress = { progressListener?.invoke(it) },
                onActiveCall = { call -> activeCall = call }
            )
            try {
                val streamed = streamer.execute(request)
                val decoded = decode(streamed.text)
                onInteractionCompleted(streamed.interactionId)
                decoded
            } catch (http: GeminiHttpException) {
                if (http.statusCode == 400 || http.statusCode == 404) modelResolver.invalidate()
                throw IOException("http=${http.statusCode};${apiError(http.statusCode, http.payload)}")
            }
        }.onSuccess {
            lastDiagnostics = VlmDiagnostics(resolvedModel, android.os.SystemClock.elapsedRealtime() - started, null)
            progressListener?.invoke(VlmStreamProgress("complete", "Sierra result validated"))
        }.onFailure {
            lastDiagnostics = VlmDiagnostics(resolvedModel, android.os.SystemClock.elapsedRealtime() - started, it.message?.take(160))
            progressListener?.invoke(VlmStreamProgress("failed", it.message?.take(120) ?: "Sierra stream failed"))
        }
    }

    private fun validateThemeReferences(plan: PlanOutput) {
        val direction = plan.photographicDirection
        if (!direction.abstain) {
            require(direction.selectedThemeId.isNotBlank()) { "Sierra omitted the theme id" }
            themeCatalog.requireInstalled(direction.selectedThemeId)
        } else if (direction.selectedThemeId.isNotBlank()) themeCatalog.requireInstalled(direction.selectedThemeId)
        direction.themeBlend.forEach { themeCatalog.requireInstalled(it.themeId) }
    }

    private fun baseRequest(key: String): Request.Builder = Request.Builder()
        .url(provider.interactionsUrl)
        .header("x-goog-api-key", key)
        .header("Content-Type", "application/json")
        .header("Api-Revision", API_REVISION)

    private fun buildInteractionBody(model: String, prompt: String, image: ByteArray, schema: JSONObject, previousInteractionId: String?): JSONObject {
        val input = JSONArray()
            .put(JSONObject().put("type", "text").put("text", prompt))
            .put(JSONObject().put("type", "image").put("data", Base64.encodeToString(image, Base64.NO_WRAP))
                .put("mime_type", "image/jpeg").put("resolution", "medium"))
        return JSONObject().put("model", model).put("input", input).put("store", true)
            .apply { previousInteractionId?.let { put("previous_interaction_id", it) } }
            .put("generation_config", JSONObject().put("thinking_level", provider.thinkingLevel).put("max_output_tokens", provider.maxOutputTokens))
            .put("response_format", JSONObject().put("type", "text").put("mime_type", "application/json").put("schema", schema))
    }

    private fun encode(source: Bitmap): ByteArray {
        val longest = max(source.width, source.height)
        val output = if (longest > maxImageDimension) {
            val scale = maxImageDimension.toFloat() / longest
            Bitmap.createScaledBitmap(source, (source.width * scale).toInt().coerceAtLeast(1), (source.height * scale).toInt().coerceAtLeast(1), true)
        } else source
        return try { ByteArrayOutputStream().use { stream -> check(output.compress(Bitmap.CompressFormat.JPEG, 80, stream)); stream.toByteArray() } }
        finally { if (output !== source && !output.isRecycled) output.recycle() }
    }

    override fun setMaxImageDimension(pixels: Int) { maxImageDimension = pixels.coerceIn(384, 896) }
    override fun setStreamProgressListener(listener: ((VlmStreamProgress) -> Unit)?) { progressListener = listener }
    override fun cancel() { activeCall?.cancel(); activeCall = null }
    override fun isReady(): Boolean = keyStore.isConfigured()
    override fun waitForSemanticPlan(): Boolean = true
    override fun supportsFinalReview(): Boolean = true
    override fun diagnostics(): VlmDiagnostics = lastDiagnostics
    override fun unload() = cancel()
    override fun close() { progressListener = null; cancel() }

    private fun currentFoundationId(): String? = foundationPrefs.getString(KEY_FOUNDATION_ID, null)?.takeIf(String::isNotBlank)
    private fun clearFoundation() {
        foundationPrefs.edit().remove(KEY_FOUNDATION_ID).remove(KEY_FOUNDATION_DIGEST).remove(KEY_FOUNDATION_AT).apply()
        lastPlanInteractionId = null
    }
    private fun foundationContextFailure(t: Throwable?): Boolean {
        val m = t?.message?.lowercase().orEmpty()
        return "previous_interaction" in m || ("interaction" in m && ("not found" in m || "expired" in m)) || m.startsWith("http=404")
    }
    private fun apiError(code: Int, payload: String): String = runCatching { JSONObject(payload).optJSONObject("error")?.optString("message") }
        .getOrNull()?.takeIf(String::isNotBlank)?.take(180) ?: "Sierra interaction failed ($code)"

    companion object {
        private val JSON_MEDIA = "application/json; charset=utf-8".toMediaType()
        private const val API_REVISION = "2026-05-20"
        private const val KEY_FOUNDATION_ID = "interaction_id"
        private const val KEY_FOUNDATION_DIGEST = "profile_digest"
        private const val KEY_FOUNDATION_AT = "created_at_ms"
        private const val FOUNDATION_REFRESH_MS = 20L * 60L * 60L * 1000L
    }
}
