package com.saunak.director.ai

import android.content.Context
import android.graphics.Bitmap
import android.util.Base64
import com.saunak.director.camera.CameraSnapshot
import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.director.EvaluationOutput
import com.saunak.director.director.PlanOutput
import com.saunak.director.storage.RecipeMemory
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
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit
import kotlin.math.max

/** Gemini cloud runtime used by Director when the user supplies an API key in Settings. */
class GeminiVlmRuntime(
    context: Context,
    private val keyStore: SecureApiKeyStore = SecureApiKeyStore(context),
    private val deviceSkill: DeviceConfigSkill = DeviceConfigSkill(context),
    private val recipeMemory: RecipeMemory = RecipeMemory(context)
) : VlmRuntime {
    private val json = Json { ignoreUnknownKeys = true; isLenient = true }
    private val client = OkHttpClient.Builder()
        .connectTimeout(2, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.SECONDS)
        .writeTimeout(4, TimeUnit.SECONDS)
        .callTimeout(6, TimeUnit.SECONDS)
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
    ): Result<PlanOutput> = runCatching {
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
    ): Result<EvaluationOutput> = runCatching {
        val config = deviceSkill.readOrGenerate(capabilities, camera)
        val prompt = """
            You are now Director's strict photography critic. Compare this intermediate result with the target objective. Judge technical quality, subject rendering, composition, lighting, color, mood and aesthetic quality. Approve when the photograph is genuinely professional and no material camera correction is needed. Otherwise return only small corrective deltas that are legal under config.md. This is attempt $attempt of maximum 3.

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
        VlmContractValidator.evaluation(json.decodeFromString(generate(prompt, image)))
    }

    private suspend fun generate(prompt: String, bitmap: Bitmap): String = withContext(Dispatchers.IO) {
        val key = keyStore.read() ?: error("Gemini API key is not configured")
        val imageBytes = encodeForAi(bitmap)
        val root = JSONObject().apply {
            put("contents", JSONArray().put(JSONObject().apply {
                put("role", "user")
                put("parts", JSONArray()
                    .put(JSONObject().put("text", prompt))
                    .put(JSONObject().put("inline_data", JSONObject()
                        .put("mime_type", "image/jpeg")
                        .put("data", Base64.encodeToString(imageBytes, Base64.NO_WRAP))))
            }))
            put("generationConfig", JSONObject()
                .put("temperature", 0.25)
                .put("topP", 0.9)
                .put("maxOutputTokens", 900)
                .put("responseMimeType", "application/json"))
        }
        val request = Request.Builder()
            .url("https://generativelanguage.googleapis.com/v1beta/models/$MODEL:generateContent")
            .header("x-goog-api-key", key)
            .header("Content-Type", "application/json")
            .post(root.toString().toRequestBody(JSON_MEDIA))
            .build()
        val call = client.newCall(request)
        activeCall = call
        try {
            call.execute().use { response ->
                val body = response.body?.string().orEmpty()
                if (!response.isSuccessful) error("Gemini request failed (${response.code})")
                val parsed = JSONObject(body)
                val candidates = parsed.optJSONArray("candidates") ?: error("Gemini returned no candidate")
                val parts = candidates.getJSONObject(0).getJSONObject("content").getJSONArray("parts")
                buildString {
                    for (i in 0 until parts.length()) append(parts.getJSONObject(i).optString("text"))
                }.trim().removePrefix("```json").removePrefix("```").removeSuffix("```").trim()
            }
        } finally {
            if (activeCall === call) activeCall = null
        }
    }

    private fun encodeForAi(source: Bitmap): ByteArray {
        val longest = max(source.width, source.height)
        val prepared = if (longest > maxImageDimension) {
            val scale = maxImageDimension.toFloat() / longest.toFloat()
            Bitmap.createScaledBitmap(source, maxOf(1, (source.width * scale).toInt()), maxOf(1, (source.height * scale).toInt()), true)
        } else source
        return try {
            ByteArrayOutputStream().use { out ->
                check(prepared.compress(Bitmap.CompressFormat.JPEG, 84, out))
                out.toByteArray()
            }
        } finally {
            if (prepared !== source && !prepared.isRecycled) prepared.recycle()
        }
    }

    override fun setMaxImageDimension(pixels: Int) { maxImageDimension = pixels.coerceIn(512, 1024) }
    override fun cancel() { activeCall?.cancel() }
    override fun isReady(): Boolean = keyStore.hasKey()
    override fun unload() = Unit
    override fun close() { activeCall?.cancel(); client.dispatcher.executorService.shutdown() }

    companion object {
        private const val MODEL = "gemini-3.8-flash"
        private val JSON_MEDIA = "application/json; charset=utf-8".toMediaType()
    }
}
