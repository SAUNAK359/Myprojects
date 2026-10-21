package com.saunak.director.ai

import android.graphics.Bitmap
import com.saunak.director.camera.CameraSnapshot
import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.director.EvaluationOutput
import com.saunak.director.director.PlanOutput
import kotlinx.coroutines.CancellableContinuation
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.json.Json
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

class LlamaCppVlmRuntime(private val promptBuilder: PromptBuilder = PromptBuilder()) : VlmRuntime {
    private val ready = AtomicBoolean(false)
    private val maxImageDimension = AtomicInteger(768)
    private val json = Json { ignoreUnknownKeys = false; isLenient = false; explicitNulls = false }
    private val inferenceExecutor = Executors.newSingleThreadExecutor { r -> Thread(r, "director-vlm").apply { priority = Thread.NORM_PRIORITY } }

    override suspend fun load(modelPath: String, projectorPath: String): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            val cores = Runtime.getRuntime().availableProcessors().coerceIn(2, 6)
            NativeVlmBridge.loadModel(modelPath, projectorPath, 4096, cores)
            ready.set(true)
        }
    }

    override suspend fun plan(image: Bitmap, capabilities: CapabilityRegistry, camera: CameraSnapshot, skillContext: String): Result<PlanOutput> = runCatching {
        withTimeout(PLAN_TIMEOUT_MS) {
            val raw = infer(image, promptBuilder.planPrompt(capabilities, camera, skillContext), PLAN_GRAMMAR, 220)
            VlmContractValidator.plan(json.decodeFromString(PlanOutput.serializer(), extractJson(raw)))
        }
    }

    override suspend fun evaluate(image: Bitmap, objective: String, attempt: Int, camera: CameraSnapshot, previousIssues: List<String>, evaluationContext: String, capabilities: CapabilityRegistry): Result<EvaluationOutput> = runCatching {
        withTimeout(EVAL_TIMEOUT_MS) {
            val raw = infer(image, promptBuilder.evaluationPrompt(objective, attempt, camera, previousIssues, evaluationContext, capabilities), EVAL_GRAMMAR, 180)
            VlmContractValidator.evaluation(json.decodeFromString(EvaluationOutput.serializer(), extractJson(raw)))
        }
    }

    private suspend fun infer(image: Bitmap, prompt: String, grammar: String, maxTokens: Int): String {
        check(ready.get()) { "VLM is not loaded" }
        val safe = ImagePreprocessor.forVlm(image, maxImageDimension.get().coerceIn(384, 1024))
        try {
            val pixels = IntArray(safe.width * safe.height)
            safe.getPixels(pixels, 0, safe.width, 0, 0, safe.width, safe.height)
            return nativeGenerate(pixels, safe.width, safe.height, prompt, grammar, maxTokens)[0]
        } finally { if (safe !== image) safe.recycle() }
    }

    private suspend fun nativeGenerate(pixels: IntArray, width: Int, height: Int, prompt: String, grammar: String, maxTokens: Int): Array<String> = suspendCancellableCoroutine { cont: CancellableContinuation<Array<String>> ->
        inferenceExecutor.execute {
            try {
                val result = NativeVlmBridge.generate(pixels, width, height, prompt, grammar, maxTokens, 0.08f, 20, 0.85f)
                if (cont.isActive) cont.resume(result)
            } catch (t: Throwable) { if (cont.isActive) cont.resumeWithException(t) }
        }
        cont.invokeOnCancellation { NativeVlmBridge.cancel() }
    }

    private fun extractJson(raw: String): String {
        val start = raw.indexOf('{'); val end = raw.lastIndexOf('}')
        require(start >= 0 && end > start) { "VLM output did not contain JSON" }
        return raw.substring(start, end + 1)
    }

    override fun setMaxImageDimension(pixels: Int) { maxImageDimension.set(pixels.coerceIn(384, 1024)) }
    override fun cancel() = NativeVlmBridge.cancel()
    override fun isReady(): Boolean = ready.get()
    override fun close() { ready.set(false); NativeVlmBridge.cancel(); runCatching { NativeVlmBridge.close() }; inferenceExecutor.shutdownNow() }

    companion object {
        private const val PLAN_TIMEOUT_MS = 14_000L
        private const val EVAL_TIMEOUT_MS = 12_000L
        private val JSON_STRING = "\\\"([^\\\"\\\\]|\\\\[\\\"\\\\/bfnrt]|\\\\u[0-9a-fA-F]{4})*\\\""
        private val NUMBER = "-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?"
        private val CALL = "\\{\\\"id\\\":[0-9]+,\\\"v\\\":($NUMBER|null),\\\"x\\\":($NUMBER|null),\\\"y\\\":($NUMBER|null),\\\"s\\\":($JSON_STRING|null)\\}"
        private val CALLS = "\\[($CALL(,$CALL)*)?\\]"
        private val ISSUES = "\\[($JSON_STRING(,$JSON_STRING)*)?\\]"
        val PLAN_GRAMMAR: String = """
root ::= "{" "\\\"scene\\\":" string ",\\\"objective\\\":" string ",\\\"branch\\\":" string ",\\\"reference_score\\\":" number ",\\\"reference_accept\\\":" boolean ",\\\"calls\\\":" calls ",\\\"recapture\\\":" boolean "}"
string ::= $JSON_STRING
number ::= $NUMBER
boolean ::= "true" | "false"
call ::= $CALL
calls ::= $CALLS
        """.trimIndent()
        val EVAL_GRAMMAR: String = """
root ::= "{" "\\\"score\\\":" number ",\\\"technical\\\":" number ",\\\"composition\\\":" number ",\\\"subject\\\":" number ",\\\"aesthetic\\\":" number ",\\\"accept\\\":" boolean ",\\\"issues\\\":" issues ",\\\"delta\\\":" calls "}"
string ::= $JSON_STRING
number ::= $NUMBER
boolean ::= "true" | "false"
call ::= $CALL
calls ::= $CALLS
issues ::= $ISSUES
        """.trimIndent()
    }
}
