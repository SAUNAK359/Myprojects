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
import kotlinx.coroutines.withContext
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import okhttp3.Call
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject

/**
 * Optional personal-beta semantic supervisor used behind Sierra branding.
 * Transport/model preferences are asset-driven and the model is capability-checked at runtime.
 * Any discovery/network/schema failure is surfaced to the orchestrator, which falls back locally.
 */
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

    @Volatile private var maxImageDimension = 512
    @Volatile private var activeCall: Call? = null
    @Volatile private var lastDiagnostics = VlmDiagnostics("sierra-unresolved")

    override suspend fun load(modelPath: String, projectorPath: String): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            val key = keyStore.read().takeIf(String::isNotBlank) ?: error("Add the Sierra Intelligence API key in Sierra settings")
            modelResolver.resolve(key)
        }.map { Unit }
    }

    override suspend fun plan(
        image: Bitmap,
        capabilities: CapabilityRegistry,
        camera: CameraSnapshot,
        skillContext: String
    ): Result<PlanOutput> = request(
        prompt = promptBuilder.planPrompt(capabilities, camera, skillContext, themeCatalog.compactDescription()),
        image = image,
        schema = planSchema(),
        decode = { raw ->
            val plan = VlmContractValidator.plan(json.decodeFromString<PlanOutput>(raw))
            validateThemeReferences(plan)
            plan
        }
    )

    override suspend fun evaluate(
        image: Bitmap,
        objective: String,
        attempt: Int,
        camera: CameraSnapshot,
        previousIssues: List<String>,
        evaluationContext: String,
        capabilities: CapabilityRegistry
    ): Result<EvaluationOutput> = request(
        prompt = promptBuilder.evaluationPrompt(objective, attempt, camera, previousIssues, evaluationContext, capabilities),
        image = image,
        schema = evaluationSchema(),
        decode = { VlmContractValidator.evaluation(json.decodeFromString<EvaluationOutput>(it)) }
    )

    private fun validateThemeReferences(plan: PlanOutput) {
        val direction = plan.photographicDirection
        if (!direction.abstain) {
            require(direction.selectedThemeId.isNotBlank()) { "Sierra omitted the theme id" }
            themeCatalog.requireInstalled(direction.selectedThemeId)
        } else if (direction.selectedThemeId.isNotBlank()) {
            themeCatalog.requireInstalled(direction.selectedThemeId)
        }
        direction.themeBlend.forEach { themeCatalog.requireInstalled(it.themeId) }
    }

    private suspend fun <T> request(
        prompt: String,
        image: Bitmap,
        schema: JSONObject,
        decode: (String) -> T
    ): Result<T> = withContext(Dispatchers.IO) {
        val started = android.os.SystemClock.elapsedRealtime()
        var resolvedModel = "sierra-unresolved"
        runCatching {
            val key = keyStore.read().takeIf(String::isNotBlank) ?: error("Sierra Intelligence API key is not configured")
            resolvedModel = modelResolver.resolve(key)
            val body = buildInteractionBody(resolvedModel, prompt, encode(image), schema)
            val request = Request.Builder()
                .url(provider.interactionsUrl)
                .header("x-goog-api-key", key)
                .header("Content-Type", "application/json")
                .post(body.toString().toRequestBody(JSON_MEDIA))
                .build()
            val call = client.newCall(request)
            activeCall = call
            try {
                call.execute().use { response ->
                    val payload = response.body?.string().orEmpty()
                    if (!response.isSuccessful) {
                        if (response.code == 404 || response.code == 400) modelResolver.invalidate()
                        throw IOException(apiError(response.code, payload))
                    }
                    decode(extractInteractionText(payload))
                }
            } finally {
                if (activeCall === call) activeCall = null
            }
        }.onSuccess {
            lastDiagnostics = VlmDiagnostics(resolvedModel, android.os.SystemClock.elapsedRealtime() - started, null)
        }.onFailure {
            lastDiagnostics = VlmDiagnostics(resolvedModel, android.os.SystemClock.elapsedRealtime() - started, it.message?.take(160))
        }
    }

    // Planning normally uses 384–640 px. Final same-capture comparison is allowed up to 896 px so
    // each evidence/render half retains enough material, skin and text detail for the critic.
    override fun setMaxImageDimension(pixels: Int) { maxImageDimension = pixels.coerceIn(384, 896) }
    override fun cancel() { activeCall?.cancel(); activeCall = null }
    override fun isReady(): Boolean = keyStore.isConfigured()
    override fun waitForSemanticPlan(): Boolean = false
    override fun supportsFinalReview(): Boolean = true
    override fun diagnostics(): VlmDiagnostics = lastDiagnostics
    override fun unload() = cancel()
    override fun close() = cancel()

    private fun buildInteractionBody(model: String, prompt: String, image: ByteArray, schema: JSONObject): JSONObject {
        val input = JSONArray()
            .put(JSONObject().put("type", "text").put("text", prompt))
            .put(JSONObject()
                .put("type", "image")
                .put("data", Base64.encodeToString(image, Base64.NO_WRAP))
                .put("mime_type", "image/jpeg")
                .put("resolution", "medium"))
        return JSONObject()
            .put("model", model)
            .put("input", input)
            .put("generation_config", JSONObject()
                .put("thinking_level", provider.thinkingLevel)
                .put("max_output_tokens", provider.maxOutputTokens))
            .put("response_format", JSONObject()
                .put("type", "text")
                .put("mime_type", "application/json")
                .put("schema", schema))
    }

    private fun planSchema(): JSONObject = JSONObject("""{
      "type":"object","required":["scene","objective","branch","reference_score","reference_accept","calls","recapture","aspect_ratio","crop_center_x","crop_center_y","photographic_direction","planner"],
      "properties":{
        "scene":{"type":"string"},"objective":{"type":"string"},"branch":{"type":"string"},
        "reference_score":{"type":"number"},"reference_accept":{"type":"boolean"},
        "calls":{"type":"array","maxItems":8,"items":{"type":"object","required":["id","v","x","y","s"],"properties":{"id":{"type":"integer"},"v":{"type":["number","null"]},"x":{"type":["number","null"]},"y":{"type":["number","null"]},"s":{"type":["string","null"]}}}},
        "recapture":{"type":"boolean"},"aspect_ratio":{"type":"string"},"crop_center_x":{"type":"number"},"crop_center_y":{"type":"number"},"planner":{"type":"string"},
        "photographic_direction":{"type":"object","required":["story","mood","primary_subject","narrative_role","selected_theme_id","theme_strength","theme_confidence","theme_blend","atmosphere_priority","skin_naturalness","highlight_protection","shadow_depth","midtone_contrast","local_contrast","color_intensity","warmth_bias","subject_relight_ev","background_suppression_ev","grain","vignette","abstain","reason_codes"],"properties":{
          "story":{"type":"string"},"mood":{"type":"string"},"primary_subject":{"type":"string"},"narrative_role":{"type":"string"},"selected_theme_id":{"type":"string"},"theme_strength":{"type":"number"},"theme_confidence":{"type":"number"},
          "theme_blend":{"type":"array","maxItems":3,"items":{"type":"object","required":["theme_id","weight"],"properties":{"theme_id":{"type":"string"},"weight":{"type":"number"}}}},
          "atmosphere_priority":{"type":"number"},"skin_naturalness":{"type":"number"},"highlight_protection":{"type":"number"},"shadow_depth":{"type":"number"},"midtone_contrast":{"type":"number"},"local_contrast":{"type":"number"},"color_intensity":{"type":"number"},"warmth_bias":{"type":"number"},"subject_relight_ev":{"type":"number"},"background_suppression_ev":{"type":"number"},"grain":{"type":"number"},"vignette":{"type":"number"},"abstain":{"type":"boolean"},"reason_codes":{"type":"array","maxItems":8,"items":{"type":"string"}}
        }}
      }}""")

    private fun evaluationSchema(): JSONObject = JSONObject("""{
      "type":"object","required":["score","technical","composition","subject","aesthetic","intent_match","naturalness","accept","issues","delta","render_adjustment"],
      "properties":{"score":{"type":"number"},"technical":{"type":"number"},"composition":{"type":"number"},"subject":{"type":"number"},"aesthetic":{"type":"number"},"intent_match":{"type":"number"},"naturalness":{"type":"number"},"accept":{"type":"boolean"},"issues":{"type":"array","maxItems":8,"items":{"type":"string"}},"delta":{"type":"array","maxItems":4,"items":{"type":"object","required":["id","v","x","y","s"],"properties":{"id":{"type":"integer"},"v":{"type":["number","null"]},"x":{"type":["number","null"]},"y":{"type":["number","null"]},"s":{"type":["string","null"]}}}},"render_adjustment":{"type":"object","required":["theme_strength_delta","midtone_contrast_delta","local_contrast_delta","shadow_depth_delta","highlight_protection_delta","subject_relight_delta_ev","background_suppression_delta_ev","warmth_delta","color_intensity_delta"],"properties":{"theme_strength_delta":{"type":"number"},"midtone_contrast_delta":{"type":"number"},"local_contrast_delta":{"type":"number"},"shadow_depth_delta":{"type":"number"},"highlight_protection_delta":{"type":"number"},"subject_relight_delta_ev":{"type":"number"},"background_suppression_delta_ev":{"type":"number"},"warmth_delta":{"type":"number"},"color_intensity_delta":{"type":"number"}}}}
    }""")

    private fun extractInteractionText(payload: String): String {
        val root = JSONObject(payload)
        if (root.optString("status") != "completed") throw IOException("Sierra interaction did not complete")
        val steps = root.optJSONArray("steps") ?: throw IOException("Sierra returned no interaction steps")
        for (stepIndex in steps.length() - 1 downTo 0) {
            val step = steps.optJSONObject(stepIndex) ?: continue
            if (step.optString("type") != "model_output") continue
            val content = step.optJSONArray("content") ?: continue
            val text = buildString {
                for (i in 0 until content.length()) {
                    val block = content.optJSONObject(i) ?: continue
                    if (block.optString("type") == "text") append(block.optString("text"))
                }
            }.trim()
            if (text.isNotBlank()) return text
        }
        throw IOException("Sierra returned no structured model output")
    }

    private fun encode(source: Bitmap): ByteArray {
        val longest = max(source.width, source.height)
        val output = if (longest > maxImageDimension) {
            val scale = maxImageDimension.toFloat() / longest
            Bitmap.createScaledBitmap(source, (source.width * scale).toInt().coerceAtLeast(1), (source.height * scale).toInt().coerceAtLeast(1), true)
        } else source
        return try {
            ByteArrayOutputStream().use { stream ->
                check(output.compress(Bitmap.CompressFormat.JPEG, 80, stream))
                stream.toByteArray()
            }
        } finally {
            if (output !== source && !output.isRecycled) output.recycle()
        }
    }

    private fun apiError(code: Int, payload: String): String =
        runCatching { JSONObject(payload).optJSONObject("error")?.optString("message") }
            .getOrNull()?.takeIf(String::isNotBlank)?.take(180)
            ?: "Sierra interaction failed ($code)"

    companion object {
        private val JSON_MEDIA = "application/json; charset=utf-8".toMediaType()
    }
}
