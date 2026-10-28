package com.saunak.director.ai

import android.graphics.Bitmap
import android.util.Base64
import android.util.Base64OutputStream
import java.io.IOException
import kotlin.math.max
import okhttp3.MediaType
import okhttp3.RequestBody
import okio.BufferedSink
import org.json.JSONObject

/**
 * Streams the analysis bitmap directly through JPEG -> Base64 -> Okio. This intentionally avoids
 * simultaneously materializing a JPEG ByteArray, a ~4/3-sized Base64 String and a second full JSON
 * request String on the Java heap.
 */
internal class StreamingGeminiRequestBody(
    private val model: String,
    private val prompt: String,
    private val image: Bitmap,
    private val maxImageDimension: Int,
    private val schema: JSONObject,
    private val previousInteractionId: String?,
    private val thinkingLevel: String,
    private val maxOutputTokens: Int,
    private val mediaType: MediaType
) : RequestBody() {
    override fun contentType(): MediaType = mediaType

    override fun writeTo(sink: BufferedSink) {
        require(!image.isRecycled) { "Sierra analysis bitmap was released before upload" }
        val prefix = buildString {
            append("{\"model\":").append(JSONObject.quote(model))
            append(",\"input\":[{\"type\":\"text\",\"text\":").append(JSONObject.quote(prompt))
            append("},{\"type\":\"image\",\"data\":\"")
        }
        val generation = JSONObject()
            .put("thinking_level", thinkingLevel)
            .put("max_output_tokens", maxOutputTokens)
        val responseFormat = JSONObject()
            .put("type", "text")
            .put("mime_type", "application/json")
            .put("schema", schema)
        val suffix = buildString {
            append("\",\"mime_type\":\"image/jpeg\",\"resolution\":\"medium\"}],\"store\":true")
            previousInteractionId?.takeIf(String::isNotBlank)?.let {
                append(",\"previous_interaction_id\":").append(JSONObject.quote(it))
            }
            append(",\"generation_config\":").append(generation.toString())
            append(",\"response_format\":").append(responseFormat.toString())
            append(",\"stream\":true}")
        }

        sink.writeUtf8(prefix)
        val longest = max(image.width, image.height)
        val encoded = if (longest > maxImageDimension) {
            val scale = maxImageDimension.toFloat() / longest.toFloat()
            Bitmap.createScaledBitmap(
                image,
                (image.width * scale).toInt().coerceAtLeast(1),
                (image.height * scale).toInt().coerceAtLeast(1),
                true
            )
        } else image
        try {
            val flags = Base64.NO_WRAP or Base64.NO_CLOSE
            Base64OutputStream(sink.outputStream(), flags).use { base64 ->
                if (!encoded.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, base64)) {
                    throw IOException("Could not JPEG-encode Sierra analysis frame")
                }
            }
            sink.writeUtf8(suffix)
        } finally {
            if (encoded !== image && !encoded.isRecycled) encoded.recycle()
        }
    }

    companion object {
        private const val JPEG_QUALITY = 80
    }
}
