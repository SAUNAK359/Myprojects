package com.saunak.director.ai

import java.io.IOException
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.suspendCancellableCoroutine
import okhttp3.Call
import okhttp3.Callback
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import org.json.JSONObject

internal data class GeminiStreamResult(val text: String, val interactionId: String?)

internal class GeminiInteractionStreamer(
    private val client: OkHttpClient,
    private val onProgress: (VlmStreamProgress) -> Unit,
    private val onActiveCall: (Call?) -> Unit
) {
    suspend fun execute(request: Request): GeminiStreamResult = suspendCancellableCoroutine { continuation ->
        val call = client.newCall(request)
        onActiveCall(call)
        continuation.invokeOnCancellation {
            call.cancel()
            onActiveCall(null)
        }
        call.enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                onActiveCall(null)
                if (continuation.isActive) continuation.resumeWithException(e)
            }

            override fun onResponse(call: Call, response: Response) {
                try {
                    val result = parse(response)
                    if (continuation.isActive) continuation.resume(result)
                } catch (t: Throwable) {
                    if (continuation.isActive) continuation.resumeWithException(t)
                } finally {
                    response.close()
                    onActiveCall(null)
                }
            }
        })
    }

    private fun parse(response: Response): GeminiStreamResult {
        if (!response.isSuccessful) {
            val payload = response.body?.string().orEmpty()
            throw GeminiHttpException(response.code, payload)
        }
        val source = response.body?.source() ?: throw IOException("Sierra stream returned no body")
        val text = StringBuilder()
        var completed = false
        var interactionId: String? = null
        var firstEvent = false
        onProgress(VlmStreamProgress("request", "Sierra request sent"))
        while (!source.exhausted()) {
            val line = source.readUtf8Line() ?: break
            if (!line.startsWith("data:")) continue
            val data = line.removePrefix("data:").trim()
            if (data.isBlank() || data == "[DONE]") continue
            if (!firstEvent) {
                firstEvent = true
                onProgress(VlmStreamProgress("stream", "Sierra response stream active"))
            }
            val event = JSONObject(data)
            when (event.optString("event_type")) {
                "interaction.created" -> {
                    interactionId = event.optJSONObject("interaction")?.optString("id")?.takeIf(String::isNotBlank)
                    onProgress(VlmStreamProgress("stream", "Sierra is reasoning over the photograph", text.length, interactionId))
                }
                "step.start" -> appendModelOutputStart(event, text)
                "step.delta" -> {
                    val delta = event.optJSONObject("delta")
                    if (delta?.optString("type") == "text") delta.optString("text").takeIf(String::isNotEmpty)?.let(text::append)
                    onProgress(VlmStreamProgress("structured_output", "Sierra structured result streaming", text.length, interactionId))
                }
                "interaction.completed" -> {
                    val interaction = event.optJSONObject("interaction")
                    completed = interaction?.optString("status") == "completed"
                    interactionId = interaction?.optString("id")?.takeIf(String::isNotBlank) ?: interactionId
                }
                "error" -> throw IOException(event.optJSONObject("error")?.optString("message")?.takeIf(String::isNotBlank)
                    ?: "Sierra streaming interaction failed")
            }
        }
        if (!completed) throw IOException("Sierra stream ended before interaction.completed")
        val raw = text.toString().trim()
        if (raw.isBlank()) throw IOException("Sierra returned no structured model output")
        onProgress(VlmStreamProgress("validation", "Sierra result complete • validating", raw.length, interactionId))
        return GeminiStreamResult(raw, interactionId)
    }

    private fun appendModelOutputStart(event: JSONObject, out: StringBuilder) {
        val step = event.optJSONObject("step") ?: return
        if (step.optString("type") != "model_output") return
        val content = step.optJSONArray("content") ?: return
        for (i in 0 until content.length()) {
            val block = content.optJSONObject(i) ?: continue
            if (block.optString("type") == "text") out.append(block.optString("text"))
        }
    }
}

internal class GeminiHttpException(val statusCode: Int, val payload: String) : IOException("Gemini HTTP $statusCode")
