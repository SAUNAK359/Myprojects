package com.saunak.director.backend

import android.content.Context
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import java.io.Closeable
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

class SierraBackendClient(
    context: Context,
    baseUrl: String,
    private val sessions: SierraSessionStore = SierraSessionStore(context)
) : SierraBackendGateway, Closeable {
    private val root = baseUrl.trim().trimEnd('/')
    private val json = Json { ignoreUnknownKeys = true; isLenient = true; encodeDefaults = true }
    private val refreshMutex = Mutex()
    private val client = OkHttpClient.Builder()
        .connectTimeout(2_500, TimeUnit.MILLISECONDS)
        .readTimeout(6_000, TimeUnit.MILLISECONDS)
        .writeTimeout(6_000, TimeUnit.MILLISECONDS)
        .callTimeout(7_000, TimeUnit.MILLISECONDS)
        .retryOnConnectionFailure(true)
        .build()

    override val configured: Boolean
        get() = root.startsWith("https://")

    override val authenticated: Boolean
        get() = sessions.read()?.refreshToken?.isNotBlank() == true

    override suspend fun signUp(request: SierraSignUpRequest): Result<SierraTokenPair> = result {
        requireConfigured()
        val pair = postPublic("/v1/auth/signup", json.encodeToString(request), 201)
            .let { json.decodeFromString<SierraTokenPair>(it) }
        sessions.save(pair)
        pair
    }

    override suspend fun signIn(request: SierraSignInRequest): Result<SierraTokenPair> = result {
        requireConfigured()
        val pair = postPublic("/v1/auth/login", json.encodeToString(request), 200)
            .let { json.decodeFromString<SierraTokenPair>(it) }
        sessions.save(pair)
        pair
    }

    override suspend fun session(): Result<SierraUserView> = result {
        val raw = executeAuthenticated { token ->
            Request.Builder().url(url("/v1/me")).header("Authorization", "Bearer $token").get().build()
        }
        val user = json.decodeFromString<SierraUserView>(raw)
        sessions.read()?.let { current ->
            sessions.save(
                SierraTokenPair(
                    accessToken = current.accessToken,
                    refreshToken = current.refreshToken,
                    accessExpiresInSeconds = ((current.accessExpiresAtMs - System.currentTimeMillis()) / 1_000L)
                        .coerceAtLeast(1L).toInt(),
                    user = user
                )
            )
        }
        user
    }

    override suspend fun credits(): Result<SierraCredits> = result {
        val raw = executeAuthenticated { token ->
            Request.Builder().url(url("/v1/me/credits")).header("Authorization", "Bearer $token").get().build()
        }
        json.decodeFromString<SierraCredits>(raw)
    }

    override suspend fun createAnalysisJob(request: SierraAnalysisJobRequest): Result<SierraJobView> = result {
        require(request.imageJpeg.isNotEmpty()) { "analysis image is empty" }
        val jpeg = "image/jpeg".toMediaType()
        val text = "text/plain; charset=utf-8".toMediaType()
        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("image", "sierra-analysis.jpg", request.imageJpeg.toRequestBody(jpeg))
            .addFormDataPart("camera_context_json", null, request.cameraContextJson.toRequestBody(text))
            .addFormDataPart("device_capabilities_json", null, request.deviceCapabilitiesJson.toRequestBody(text))
            .addFormDataPart("local_quality_json", null, request.localQualityJson.toRequestBody(text))
            .addFormDataPart("skill_context", null, request.skillContext.toRequestBody(text))
            .build()
        val raw = executeAuthenticated { token ->
            Request.Builder()
                .url(url("/v1/jobs"))
                .header("Authorization", "Bearer $token")
                .header("Idempotency-Key", request.idempotencyKey)
                .post(body)
                .build()
        }
        json.decodeFromString<SierraJobView>(raw)
    }

    override suspend fun job(jobId: String): Result<SierraJobView> = result {
        val raw = executeAuthenticated { token ->
            Request.Builder()
                .url(url("/v1/jobs/${safeId(jobId)}"))
                .header("Authorization", "Bearer $token")
                .get()
                .build()
        }
        json.decodeFromString<SierraJobView>(raw)
    }

    override suspend fun complete(jobId: String, captureId: String): Result<SierraJobView> = result {
        val body = json.encodeToString(SierraCompleteRequest(captureId)).toRequestBody(JSON_MEDIA)
        val raw = executeAuthenticated { token ->
            Request.Builder()
                .url(url("/v1/jobs/${safeId(jobId)}/complete"))
                .header("Authorization", "Bearer $token")
                .post(body)
                .build()
        }
        json.decodeFromString<SierraJobView>(raw)
    }

    override suspend fun cancel(jobId: String): Result<SierraJobView> = result {
        val raw = executeAuthenticated { token ->
            Request.Builder()
                .url(url("/v1/jobs/${safeId(jobId)}/cancel"))
                .header("Authorization", "Bearer $token")
                .post(EMPTY_BODY)
                .build()
        }
        json.decodeFromString<SierraJobView>(raw)
    }

    override suspend fun signOut(): Result<Unit> = result {
        val stored = sessions.read()
        sessions.clear()
        if (configured && stored != null) {
            runCatching {
                postPublic(
                    "/v1/auth/logout",
                    json.encodeToString(SierraRefreshRequest(stored.refreshToken)),
                    204
                )
            }
        }
        Unit
    }

    fun cachedUser(): SierraUserView? = sessions.read()?.user

    private suspend fun executeAuthenticated(factory: (String) -> Request): String {
        requireConfigured()
        var token = validAccessToken()
        var response = execute(factory(token))
        if (response.code == 401) {
            response.close()
            token = refreshAccessToken(force = true)
            response = execute(factory(token))
        }
        response.use { value ->
            val raw = value.body?.string().orEmpty()
            if (!value.isSuccessful) throw apiError(value.code, raw)
            return raw
        }
    }

    private suspend fun validAccessToken(): String {
        val stored = sessions.read() ?: throw SierraAuthException("Sign in to use Sierra AI")
        if (stored.accessExpiresAtMs > System.currentTimeMillis() + 30_000L) return stored.accessToken
        return refreshAccessToken(force = false)
    }

    private suspend fun refreshAccessToken(force: Boolean): String = refreshMutex.withLock {
        val current = sessions.read() ?: throw SierraAuthException("Sierra session is unavailable")
        if (!force && current.accessExpiresAtMs > System.currentTimeMillis() + 30_000L) {
            return@withLock current.accessToken
        }
        val raw = postPublic(
            "/v1/auth/refresh",
            json.encodeToString(SierraRefreshRequest(current.refreshToken)),
            200
        )
        val pair = json.decodeFromString<SierraTokenPair>(raw)
        sessions.save(pair)
        pair.accessToken
    }

    private suspend fun postPublic(path: String, body: String, expectedCode: Int): String {
        requireConfigured()
        val response = execute(
            Request.Builder().url(url(path)).post(body.toRequestBody(JSON_MEDIA)).build()
        )
        response.use { value ->
            val raw = value.body?.string().orEmpty()
            if (value.code != expectedCode) throw apiError(value.code, raw)
            return raw
        }
    }

    private suspend fun execute(request: Request): Response = suspendCancellableCoroutine { continuation ->
        val call = client.newCall(request)
        continuation.invokeOnCancellation { call.cancel() }
        call.enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                if (continuation.isActive) continuation.resumeWithException(e)
            }

            override fun onResponse(call: Call, response: Response) {
                if (continuation.isActive) continuation.resume(response) else response.close()
            }
        })
    }

    private fun apiError(code: Int, raw: String): IOException {
        val detail = runCatching {
            json.parseToJsonElement(raw).jsonObject["detail"]?.jsonPrimitive?.contentOrNull
        }.getOrNull()?.take(220)
        val message = detail ?: when (code) {
            401 -> "Sign in to use Sierra AI"
            402 -> "No Sierra AI credits available"
            409 -> "Sierra job state changed; try again"
            413 -> "Analysis image is too large"
            429 -> "Too many requests; try again shortly"
            in 500..599 -> "Sierra server is temporarily unavailable"
            else -> "Sierra server request failed ($code)"
        }
        return if (code == 401) SierraAuthException(message) else SierraApiException(code, message)
    }

    private fun url(path: String): String = root + path

    private fun safeId(value: String): String {
        require(value.matches(Regex("[A-Za-z0-9-]{8,80}"))) { "invalid Sierra job id" }
        return value
    }

    private fun requireConfigured() {
        check(configured) { "Sierra server URL is not configured" }
    }

    private suspend fun <T> result(block: suspend () -> T): Result<T> = try {
        Result.success(block())
    } catch (t: CancellationException) {
        throw t
    } catch (t: Throwable) {
        Result.failure(t)
    }

    override fun close() {
        client.dispatcher.executorService.shutdown()
        client.connectionPool.evictAll()
    }

    companion object {
        private val JSON_MEDIA = "application/json; charset=utf-8".toMediaType()
        private val EMPTY_BODY = ByteArray(0).toRequestBody(null)
    }
}

open class SierraApiException(val statusCode: Int, message: String) : IOException(message)
class SierraAuthException(message: String) : SierraApiException(401, message)
