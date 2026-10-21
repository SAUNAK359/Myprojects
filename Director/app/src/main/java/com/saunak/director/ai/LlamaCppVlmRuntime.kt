package com.saunak.director.ai

import android.graphics.Bitmap
import com.saunak.director.camera.CameraSnapshot
import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.director.EvaluationOutput
import com.saunak.director.director.PlanOutput
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

class LlamaCppVlmRuntime(private val promptBuilder: PromptBuilder = PromptBuilder()) : VlmRuntime {
    private val ready = AtomicBoolean(false)
    private val maxImageDimension = AtomicInteger(768)
    private val json = Json { ignoreUnknownKeys = false; isLenient = false; explicitNulls = false }

    override suspend fun load(modelPath: String, projectorPath: String): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            val cores = Runtime.getRuntime().availableProcessors().coerceIn(2, 8)
            NativeVlmBridge.loadModel(modelPath, projectorPath, 4096, cores)
            ready.set(true)
        }
    }
    override suspend fun plan(image: Bitmap, capabilities: CapabilityRegistry, camera: CameraSnapshot, skillContext: String): Result<PlanOutput> = infer(image, promptBuilder.planPrompt(capabilities, camera, skillContext), 160).mapCatching { json.decodeFromString(PlanOutput.serializer(), extractJson(it)) }
    override suspend fun evaluate(image: Bitmap, objective: String, attempt: Int, camera: CameraSnapshot, previousIssues: List<String>, evaluationContext: String): Result<EvaluationOutput> = infer(image, promptBuilder.evaluationPrompt(objective, attempt, camera, previousIssues, evaluationContext), 128).mapCatching { json.decodeFromString(EvaluationOutput.serializer(), extractJson(it)) }
    private suspend fun infer(image: Bitmap, prompt: String, maxTokens: Int): Result<String> = withContext(Dispatchers.Default) {
        runCatching {
            check(ready.get()) { "VLM is not loaded" }
            val safe = ImagePreprocessor.forVlm(image, maxImageDimension.get().coerceIn(384, 1024))
            try {
                val pixels = IntArray(safe.width * safe.height)
                safe.getPixels(pixels, 0, safe.width, 0, 0, safe.width, safe.height)
                NativeVlmBridge.generate(pixels, safe.width, safe.height, prompt, maxTokens, 0.1f, 20, 0.8f)[0]
            } finally { if (safe !== image) safe.recycle() }
        }
    }
    private fun extractJson(raw: String): String {
        val start = raw.indexOf('{'); val end = raw.lastIndexOf('}')
        require(start >= 0 && end > start) { "VLM output did not contain a JSON object" }
        return raw.substring(start, end + 1)
    }
    override fun setMaxImageDimension(pixels: Int) { maxImageDimension.set(pixels.coerceIn(384, 1024)) }
    override fun cancel() = NativeVlmBridge.cancel()
    override fun isReady(): Boolean = ready.get()
    override fun close() { ready.set(false); NativeVlmBridge.close() }
}
