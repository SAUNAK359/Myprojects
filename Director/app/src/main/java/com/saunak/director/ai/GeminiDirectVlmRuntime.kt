package com.saunak.director.ai

import android.content.Context
import android.graphics.Bitmap
import com.saunak.director.camera.CameraSnapshot
import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.director.EvaluationOutput
import com.saunak.director.director.PlanOutput
import com.saunak.director.director.PreShutterValidationOutput
import com.saunak.director.director.live.LiveDirectorValidator
import com.saunak.director.director.live.SierraDirectorSpec
import java.io.IOException
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
import org.json.JSONObject

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
    @Volatile private var latestDirectorSpec: SierraDirectorSpec? = null
    @Volatile private var lastDirectorContract: DirectorSchemaCompiler.Contract? = null

    override suspend fun load(modelPath: String, projectorPath: String): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching { modelResolver.resolve(keyStore.read().takeIf(String::isNotBlank) ?: error("Add the Sierra Intelligence API key in Sierra settings")) }.map { Unit }
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
        return storedDigest == profile.digest && !storedId.isNullOrBlank() && System.currentTimeMillis() - storedAt in 0 until FOUNDATION_REFRESH_MS
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
                    Store this DEVICE FOUNDATION as the hard physical/API/render-executor boundary for future camera plans and render specifications.
                    Sierra beta may enforce physical feasibility but must not creatively replace Sierra's valid decisions.
                    Device limits change execution, not photographic intent. Never invent unsupported camera or render capabilities.
                    Preserve identity, geometry, text, material truth and scene evidence.
                    ${profile.payload}
                """.trimIndent())
                .put("generation_config", JSONObject().put("thinking_level", provider.thinkingLevel).put("max_output_tokens", 128))
                .put("store", true)
            val call = client.newCall(baseRequest(key).post(body.toString().toRequestBody(JSON_MEDIA)).build())
            activeCall = call
            try {
                call.execute().use { response ->
                    val payload = response.body?.string().orEmpty()
                    if (!response.isSuccessful) throw IOException(apiError(response.code, payload))
                    val id = JSONObject(payload).optString("id").takeIf(String::isNotBlank) ?: throw IOException("Sierra device foundation returned no interaction id")
                    foundationPrefs.edit().putString(KEY_FOUNDATION_ID, id).putString(KEY_FOUNDATION_DIGEST, profile.digest).putLong(KEY_FOUNDATION_AT, System.currentTimeMillis()).apply()
                    lastPlanInteractionId = null
                    latestDirectorSpec = null
                    lastDirectorContract = null
                }
            } finally { if (activeCall === call) activeCall = null }
        }.onSuccess {
            lastDiagnostics = VlmDiagnostics(resolvedModel, android.os.SystemClock.elapsedRealtime() - started, null)
        }.onFailure {
            lastDiagnostics = VlmDiagnostics(resolvedModel, android.os.SystemClock.elapsedRealtime() - started, it.message?.take(160))
        }
    }

    override suspend fun plan(image: Bitmap, capabilities: CapabilityRegistry, camera: CameraSnapshot, skillContext: String): Result<PlanOutput> {
        latestDirectorSpec = null
        lastPlanInteractionId = null
        val directorContract = withContext(Dispatchers.IO) { DirectorSchemaCompiler().compile() }
        lastDirectorContract = directorContract
        suspend fun execute(): Result<PlanOutput> {
            lastDirectorContract = directorContract
            val result = streamRequest(
                promptBuilder.planPrompt(capabilities, camera, skillContext, themeCatalog.compactDescription(), directorContract),
                image,
                SierraGeminiSchemas.plan(directorContract),
                currentFoundationId(),
                onInteractionCompleted = { lastPlanInteractionId = it }
            ) { raw ->
                val decoded = VlmContractValidator.plan(json.decodeFromString<PlanOutput>(raw))
                val director = requireNotNull(decoded.directorSpec) { "Sierra omitted required Director spec" }
                val render = requireNotNull(decoded.renderSpec) { "Sierra omitted required Render spec" }
                val validatedDirector = LiveDirectorValidator.validate(director, directorContract)
                val validatedRender = VlmContractValidator.renderSpec(render)
                // Themes are optional descriptive priors in v11 production. RenderSpec is the
                // actual creative contract, so an unknown/custom/blank theme must never reject an
                // otherwise valid photograph.
                decoded.copy(directorSpec = validatedDirector, renderSpec = validatedRender)
            }
            result.onSuccess { latestDirectorSpec = it.directorSpec }
            return result
        }

        val first = execute()
        if (first.isSuccess || !foundationContextFailure(first.exceptionOrNull())) return first
        val profile = lastFoundationProfile ?: return first
        val rebuilt = withContext(Dispatchers.IO) { foundationMutex.withLock { clearFoundation(); createFoundation(profile) } }
        if (rebuilt.isFailure) return Result.failure(rebuilt.exceptionOrNull() ?: IllegalStateException("Device foundation rebuild failed"))
        return execute()
    }

    override suspend fun validatePreShutter(
        image: Bitmap,
        originalPlan: PlanOutput,
        camera: CameraSnapshot,
        validationContext: String,
        capabilities: CapabilityRegistry
    ): Result<PreShutterValidationOutput> {
        val interaction = lastPlanInteractionId ?: return Result.failure(IllegalStateException("No validated Sierra planning interaction for pre-shutter call"))
        val contract = lastDirectorContract ?: return Result.failure(IllegalStateException("Director capability contract unavailable"))
        return streamRequest(
            promptBuilder.preShutterValidationPrompt(originalPlan, camera, validationContext, capabilities, contract),
            image,
            SierraGeminiSchemas.preShutterValidation(contract),
            interaction
        ) { raw ->
            PreShutterValidator.validate(json.decodeFromString<PreShutterValidationOutput>(raw), capabilities, contract)
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
    ): Result<EvaluationOutput> = Result.failure(UnsupportedOperationException("Post-capture Gemini evaluation disabled; Render Spec is authored in call #1"))

    private suspend fun <T> streamRequest(
        prompt: String,
        image: Bitmap,
        schema: JSONObject,
        previousInteractionId: String?,
        onInteractionCompleted: (String?) -> Unit = {},
        decode: (String) -> T
    ): Result<T> {
        val started = android.os.SystemClock.elapsedRealtime()
        var resolvedModel = "sierra-unresolved"
        return runCatching {
            val key = keyStore.read().takeIf(String::isNotBlank) ?: error("Sierra Intelligence API key is not configured")
            resolvedModel = withContext(Dispatchers.IO) { modelResolver.resolve(key) }
            val requestBody = StreamingGeminiRequestBody(
                model = resolvedModel,
                prompt = prompt,
                image = image,
                maxImageDimension = maxImageDimension,
                schema = schema,
                previousInteractionId = previousInteractionId,
                thinkingLevel = provider.thinkingLevel,
                maxOutputTokens = provider.maxOutputTokens,
                mediaType = JSON_MEDIA
            )
            val request = baseRequest(key).header("Accept", "text/event-stream").post(requestBody).build()
            val streamer = GeminiInteractionStreamer(client, { progressListener?.invoke(it) }, { call -> activeCall = call })
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

    private fun baseRequest(key: String): Request.Builder = Request.Builder().url(provider.interactionsUrl).header("x-goog-api-key", key).header("Content-Type", "application/json")

    override fun setMaxImageDimension(pixels: Int) { maxImageDimension = pixels.coerceIn(384, 896) }
    override fun setStreamProgressListener(listener: ((VlmStreamProgress) -> Unit)?) { progressListener = listener }
    override fun cancel() {
        activeCall?.cancel()
        activeCall = null
        latestDirectorSpec = null
        lastPlanInteractionId = null
        lastDirectorContract = null
    }
    override fun isReady(): Boolean = keyStore.isConfigured()
    override fun waitForSemanticPlan(): Boolean = true
    override fun supportsPreShutterValidation(): Boolean = true
    override fun supportsFinalReview(): Boolean = false
    override fun diagnostics(): VlmDiagnostics = lastDiagnostics
    override fun lastDirectorSpec(): SierraDirectorSpec? = latestDirectorSpec
    override fun unload() = cancel()
    override fun close() { progressListener = null; cancel() }

    private fun currentFoundationId(): String? = foundationPrefs.getString(KEY_FOUNDATION_ID, null)?.takeIf(String::isNotBlank)
    private fun clearFoundation() {
        foundationPrefs.edit().remove(KEY_FOUNDATION_ID).remove(KEY_FOUNDATION_DIGEST).remove(KEY_FOUNDATION_AT).apply()
        lastPlanInteractionId = null
        latestDirectorSpec = null
        lastDirectorContract = null
    }
    private fun foundationContextFailure(t: Throwable?): Boolean {
        val m = t?.message?.lowercase().orEmpty()
        return "previous_interaction" in m || ("interaction" in m && ("not found" in m || "expired" in m)) || m.startsWith("http=404")
    }
    private fun apiError(code: Int, payload: String): String = runCatching { JSONObject(payload).optJSONObject("error")?.optString("message") }.getOrNull()?.takeIf(String::isNotBlank)?.take(180) ?: "Sierra interaction failed ($code)"

    companion object {
        private val JSON_MEDIA = "application/json; charset=utf-8".toMediaType()
        private const val KEY_FOUNDATION_ID = "interaction_id"
        private const val KEY_FOUNDATION_DIGEST = "profile_digest"
        private const val KEY_FOUNDATION_AT = "created_at_ms"
        private const val FOUNDATION_REFRESH_MS = 20L * 60L * 60L * 1000L
    }
}
