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

class GeminiDirectVlmRuntime(
    context: Context,
    private val keyStore: GeminiApiKeyStore = GeminiApiKeyStore(context),
    private val promptBuilder: PromptBuilder = PromptBuilder()
) : VlmRuntime {
    private val json = Json { ignoreUnknownKeys = true; explicitNulls = false }
    private val themeCatalog = ThemePromptCatalog(context)
    private val client = OkHttpClient.Builder()
        .connectTimeout(2, TimeUnit.SECONDS)
        .readTimeout(12, TimeUnit.SECONDS)
        .writeTimeout(4, TimeUnit.SECONDS)
        .callTimeout(14, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .build()

    @Volatile private var maxImageDimension = 512
    @Volatile private var activeCall: Call? = null
    @Volatile private var lastDiagnostics = VlmDiagnostics("gemini-3.8-flash")

    override suspend fun load(modelPath: String, projectorPath: String): Result<Unit> =
        if (isReady()) Result.success(Unit) else Result.failure(IllegalStateException("Add a Gemini API key in Sierra settings"))

    override suspend fun plan(image: Bitmap, capabilities: CapabilityRegistry, camera: CameraSnapshot, skillContext: String): Result<PlanOutput> =
        request(
            prompt = promptBuilder.planPrompt(capabilities, camera, skillContext, themeCatalog.compactDescription()),
            image = image,
            schema = planSchema(),
            decode = { VlmContractValidator.plan(json.decodeFromString<PlanOutput>(it)) }
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

    private suspend fun <T> request(prompt: String, image: Bitmap, schema: JSONObject, decode: (String) -> T): Result<T> = withContext(Dispatchers.IO) {
        val started = android.os.SystemClock.elapsedRealtime()
        runCatching {
            val key = keyStore.read().takeIf(String::isNotBlank) ?: error("Gemini API key is not configured")
            val body = buildRequestBody(prompt, encode(image), schema)
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
                    decode(extractText(payload))
                }
            } finally {
                if (activeCall === call) activeCall = null
            }
        }.onSuccess {
            lastDiagnostics = VlmDiagnostics(MODEL, android.os.SystemClock.elapsedRealtime() - started, null)
        }.onFailure {
            lastDiagnostics = VlmDiagnostics(MODEL, android.os.SystemClock.elapsedRealtime() - started, it.message?.take(160))
        }
    }

    override fun setMaxImageDimension(pixels: Int) { maxImageDimension = pixels.coerceIn(384, 640) }
    override fun cancel() { activeCall?.cancel(); activeCall = null }
    override fun isReady(): Boolean = keyStore.isConfigured()
    override fun waitForSemanticPlan(): Boolean = true
    override fun supportsFinalReview(): Boolean = true
    override fun diagnostics(): VlmDiagnostics = lastDiagnostics
    override fun unload() = cancel()
    override fun close() = cancel()

    private fun buildRequestBody(prompt: String, image: ByteArray, schema: JSONObject): JSONObject {
        val parts = JSONArray()
            .put(JSONObject().put("text", prompt))
            .put(JSONObject().put("inlineData", JSONObject().put("mimeType", "image/jpeg").put("data", Base64.encodeToString(image, Base64.NO_WRAP))))
        return JSONObject()
            .put("contents", JSONArray().put(JSONObject().put("role", "user").put("parts", parts)))
            .put("generationConfig", JSONObject()
                .put("responseMimeType", "application/json")
                .put("responseJsonSchema", schema)
                .put("thinkingConfig", JSONObject().put("thinkingLevel", "low"))
                .put("maxOutputTokens", 1150))
    }

    private fun planSchema(): JSONObject = JSONObject("""{
      "type":"object","required":["scene","objective","branch","reference_score","reference_accept","calls","recapture","aspect_ratio","crop_center_x","crop_center_y","photographic_direction","planner"],
      "properties":{
        "scene":{"type":"string"},"objective":{"type":"string"},"branch":{"type":"string"},
        "reference_score":{"type":"number"},"reference_accept":{"type":"boolean"},
        "calls":{"type":"array","items":{"type":"object","required":["id","v","x","y","s"],"properties":{"id":{"type":"integer"},"v":{"type":["number","null"]},"x":{"type":["number","null"]},"y":{"type":["number","null"]},"s":{"type":["string","null"]}}}},
        "recapture":{"type":"boolean"},"aspect_ratio":{"type":"string"},"crop_center_x":{"type":"number"},"crop_center_y":{"type":"number"},"planner":{"type":"string"},
        "photographic_direction":{"type":"object","required":["story","mood","primary_subject","narrative_role","selected_theme_id","theme_strength","theme_confidence","theme_blend","atmosphere_priority","skin_naturalness","highlight_protection","shadow_depth","midtone_contrast","local_contrast","color_intensity","warmth_bias","subject_relight_ev","background_suppression_ev","grain","vignette","abstain","reason_codes"],"properties":{
          "story":{"type":"string"},"mood":{"type":"string"},"primary_subject":{"type":"string"},"narrative_role":{"type":"string"},"selected_theme_id":{"type":"string"},"theme_strength":{"type":"number"},"theme_confidence":{"type":"number"},
          "theme_blend":{"type":"array","items":{"type":"object","required":["theme_id","weight"],"properties":{"theme_id":{"type":"string"},"weight":{"type":"number"}}}},
          "atmosphere_priority":{"type":"number"},"skin_naturalness":{"type":"number"},"highlight_protection":{"type":"number"},"shadow_depth":{"type":"number"},"midtone_contrast":{"type":"number"},"local_contrast":{"type":"number"},"color_intensity":{"type":"number"},"warmth_bias":{"type":"number"},"subject_relight_ev":{"type":"number"},"background_suppression_ev":{"type":"number"},"grain":{"type":"number"},"vignette":{"type":"number"},"abstain":{"type":"boolean"},"reason_codes":{"type":"array","items":{"type":"string"}}
        }}
      }}""")

    private fun evaluationSchema(): JSONObject = JSONObject("""{
      "type":"object","required":["score","technical","composition","subject","aesthetic","intent_match","naturalness","accept","issues","delta","render_adjustment"],
      "properties":{"score":{"type":"number"},"technical":{"type":"number"},"composition":{"type":"number"},"subject":{"type":"number"},"aesthetic":{"type":"number"},"intent_match":{"type":"number"},"naturalness":{"type":"number"},"accept":{"type":"boolean"},"issues":{"type":"array","items":{"type":"string"}},"delta":{"type":"array","items":{"type":"object","required":["id","v","x","y","s"],"properties":{"id":{"type":"integer"},"v":{"type":["number","null"]},"x":{"type":["number","null"]},"y":{"type":["number","null"]},"s":{"type":["string","null"]}}}},"render_adjustment":{"type":"object","required":["theme_strength_delta","midtone_contrast_delta","local_contrast_delta","shadow_depth_delta","highlight_protection_delta","subject_relight_delta_ev","background_suppression_delta_ev","warmth_delta","color_intensity_delta"],"properties":{"theme_strength_delta":{"type":"number"},"midtone_contrast_delta":{"type":"number"},"local_contrast_delta":{"type":"number"},"shadow_depth_delta":{"type":"number"},"highlight_protection_delta":{"type":"number"},"subject_relight_delta_ev":{"type":"number"},"background_suppression_delta_ev":{"type":"number"},"warmth_delta":{"type":"number"},"color_intensity_delta":{"type":"number"}}}}
    }""")

    private fun extractText(payload: String): String {
        val candidates = JSONObject(payload).optJSONArray("candidates") ?: throw IOException("Gemini returned no candidates")
        val parts = candidates.optJSONObject(0)?.optJSONObject("content")?.optJSONArray("parts") ?: throw IOException("Gemini returned no content")
        return buildString { for (i in 0 until parts.length()) append(parts.optJSONObject(i)?.optString("text").orEmpty()) }.trim().ifBlank { throw IOException("Gemini returned no JSON") }
    }

    private fun encode(source: Bitmap): ByteArray {
        val longest = max(source.width, source.height)
        val output = if (longest > maxImageDimension) {
            val scale = maxImageDimension.toFloat() / longest
            Bitmap.createScaledBitmap(source, (source.width * scale).toInt().coerceAtLeast(1), (source.height * scale).toInt().coerceAtLeast(1), true)
        } else source
        return try { ByteArrayOutputStream().use { stream -> check(output.compress(Bitmap.CompressFormat.JPEG, 72, stream)); stream.toByteArray() } }
        finally { if (output !== source && !output.isRecycled) output.recycle() }
    }

    private fun apiError(code: Int, payload: String): String = runCatching { JSONObject(payload).optJSONObject("error")?.optString("message") }.getOrNull()?.takeIf(String::isNotBlank)?.take(180) ?: "Gemini request failed ($code)"

    companion object {
        private const val API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
        private const val MODEL = "gemini-3.8-flash"
        private val JSON_MEDIA = "application/json; charset=utf-8".toMediaType()
    }
}
