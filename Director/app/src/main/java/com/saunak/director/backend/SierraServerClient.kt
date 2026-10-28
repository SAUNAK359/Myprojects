package com.saunak.director.backend

import android.content.Context
import android.os.Build
import com.saunak.director.BuildConfig
import com.saunak.director.director.live.SierraDirectorSpec
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody

@Serializable
data class SierraSignupPayload(val display_name: String, val email: String, val phone: String, val dob: String, val password: String, val device: SierraDeviceProfile)
@Serializable data class SierraLoginPayload(val email: String, val password: String)
@Serializable data class SierraRefreshPayload(val refresh_token: String)
@Serializable data class SierraLogoutPayload(val refresh_token: String)
@Serializable data class SierraUserDto(val id: String, val display_name: String, val available_credits: Int)
@Serializable data class SierraTokenPairDto(val access_token: String, val refresh_token: String, val user: SierraUserDto)
@Serializable data class SierraCreditDto(val balance: Int, val reserved: Int, val available: Int)
@Serializable data class SierraDeviceSyncDto(val id: String)

@Serializable
data class SierraPlanDto(
    val scene: String,
    val objective: String,
    val branch: String,
    val reference_score: Double = 0.5,
    val reference_accept: Boolean = false,
    val calls: List<SierraToolCallDto> = emptyList(),
    val recapture: Boolean = true,
    val aspect_ratio: String = "4:3",
    val crop_center_x: Double = 0.5,
    val crop_center_y: Double = 0.5,
    val director_spec: SierraDirectorSpec? = null,
    val planner: String = "sierra_ai_server"
)

@Serializable
data class SierraToolCallDto(
    val id: Int,
    @SerialName("v") val value: Double? = null,
    val x: Double? = null,
    val y: Double? = null,
    @SerialName("s") val stringValue: String? = null
)

@Serializable
data class SierraJobDto(
    val id: String,
    val state: String,
    val plan: SierraPlanDto? = null,
    val error_code: String? = null,
    val error_message: String? = null,
    val credits: SierraCreditDto
)

class SierraServerClient(context: Context) {
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
    private val sessions = SierraSessionStore(context)
    private val refreshMutex = Mutex()
    private val baseUrl = BuildConfig.SIERRA_SERVER_BASE_URL.trim().removeSuffix("/")
    private val client = OkHttpClient.Builder()
        .connectTimeout(3, TimeUnit.SECONDS)
        .readTimeout(8, TimeUnit.SECONDS)
        .writeTimeout(6, TimeUnit.SECONDS)
        .callTimeout(10, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .build()

    val configured: Boolean get() = baseUrl.startsWith("https://")
    val authenticated: Boolean get() = sessions.read() != null
    fun session(): SierraSessionStore.Session? = sessions.read()

    suspend fun signup(name: String, email: String, phone: String, dobIso: String, password: String, device: SierraDeviceProfile): Result<SierraSessionStore.Session> = runCatching {
        require(configured) { "Sierra Server is not configured" }
        val response: SierraTokenPairDto = postPublic("/v1/auth/signup", json.encodeToString(SierraSignupPayload(name, email, phone, dobIso, password, device)))
        persist(response)
    }

    suspend fun login(email: String, password: String): Result<SierraSessionStore.Session> = runCatching {
        require(configured) { "Sierra Server is not configured" }
        val response: SierraTokenPairDto = postPublic("/v1/auth/login", json.encodeToString(SierraLoginPayload(email, password)))
        persist(response)
    }

    suspend fun logout(): Result<Unit> = runCatching {
        val session = sessions.read()
        if (session != null && configured) runCatching { postPublic<Unit>("/v1/auth/logout", json.encodeToString(SierraLogoutPayload(session.refreshToken)), allowEmptyBody = true) }
        sessions.clear()
    }

    fun logoutLocal() = sessions.clear()

    suspend fun syncDevice(profile: SierraDeviceProfile): Result<Unit> = runCatching {
        authenticatedJson<SierraDeviceSyncDto>("PUT", "/v1/me/devices/current", json.encodeToString(profile)); Unit
    }

    suspend fun credits(): Result<SierraCreditDto> = runCatching {
        val value: SierraCreditDto = authenticatedJson("GET", "/v1/me/credits", null)
        sessions.updateCredits(value.available); value
    }

    suspend fun createAnalysisJob(
        imageJpeg: ByteArray,
        cameraContextJson: String,
        deviceCapabilitiesJson: String,
        localQualityJson: String,
        skillContext: String,
        idempotencyKey: String
    ): Result<SierraJobDto> = runCatching {
        require(configured) { "Sierra Server is not configured" }
        val multipart = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("camera_context_json", cameraContextJson)
            .addFormDataPart("device_capabilities_json", deviceCapabilitiesJson)
            .addFormDataPart("local_quality_json", localQualityJson)
            .addFormDataPart("skill_context", skillContext)
            .addFormDataPart("image", "analysis.jpg", imageJpeg.toRequestBody("image/jpeg".toMediaType()))
            .build()
        authenticatedRequest(Request.Builder().url(baseUrl + "/v1/jobs").header("Idempotency-Key", idempotencyKey).post(multipart))
    }

    suspend fun job(jobId: String): Result<SierraJobDto> = runCatching { authenticatedJson("GET", "/v1/jobs/$jobId", null) }

    suspend fun complete(jobId: String, captureId: String): Result<SierraJobDto> = runCatching {
        val body = "{\"capture_id\":${json.encodeToString(captureId)}}"
        val value: SierraJobDto = authenticatedJson("POST", "/v1/jobs/$jobId/complete", body)
        sessions.updateCredits(value.credits.available); value
    }

    suspend fun cancel(jobId: String): Result<Unit> = runCatching { authenticatedJson<SierraJobDto>("POST", "/v1/jobs/$jobId/cancel", "{}"); Unit }

    private fun persist(dto: SierraTokenPairDto): SierraSessionStore.Session {
        val session = SierraSessionStore.Session(dto.access_token, dto.refresh_token, dto.user.id, dto.user.display_name, dto.user.available_credits)
        sessions.save(session); return session
    }

    private suspend inline fun <reified T> postPublic(path: String, body: String, allowEmptyBody: Boolean = false): T = withContext(Dispatchers.IO) {
        val request = Request.Builder().url(baseUrl + path).header("User-Agent", userAgent()).post(body.toRequestBody(JSON_MEDIA)).build()
        execute<T>(request, allowEmptyBody)
    }

    private suspend inline fun <reified T> authenticatedJson(method: String, path: String, body: String?): T {
        val builder = Request.Builder().url(baseUrl + path).header("User-Agent", userAgent())
        when (method) {
            "GET" -> builder.get()
            "PUT" -> builder.put((body ?: "{}").toRequestBody(JSON_MEDIA))
            "POST" -> builder.post((body ?: "{}").toRequestBody(JSON_MEDIA))
            else -> error("Unsupported HTTP method")
        }
        return authenticatedRequest(builder)
    }

    private suspend inline fun <reified T> authenticatedRequest(builder: Request.Builder): T = withContext(Dispatchers.IO) {
        var session = sessions.read() ?: throw IOException("Sign in to use Sierra AI")
        var request = builder.header("Authorization", "Bearer ${session.accessToken}").header("User-Agent", userAgent()).build()
        var response = client.newCall(request).execute()
        if (response.code == 401) {
            response.close()
            session = refreshMutex.withLock { refreshSession(session.refreshToken) }
            request = builder.header("Authorization", "Bearer ${session.accessToken}").header("User-Agent", userAgent()).build()
            response = client.newCall(request).execute()
        }
        response.use {
            val text = it.body?.string().orEmpty()
            if (!it.isSuccessful) throw IOException(serverError(it.code, text))
            json.decodeFromString<T>(text)
        }
    }

    private fun refreshSession(refreshToken: String): SierraSessionStore.Session {
        val request = Request.Builder().url(baseUrl + "/v1/auth/refresh").header("User-Agent", userAgent()).post(json.encodeToString(SierraRefreshPayload(refreshToken)).toRequestBody(JSON_MEDIA)).build()
        return client.newCall(request).execute().use {
            val text = it.body?.string().orEmpty()
            if (!it.isSuccessful) { sessions.clear(); throw IOException("Sierra session expired") }
            persist(json.decodeFromString<SierraTokenPairDto>(text))
        }
    }

    private inline fun <reified T> execute(request: Request, allowEmptyBody: Boolean = false): T = client.newCall(request).execute().use {
        val text = it.body?.string().orEmpty()
        if (!it.isSuccessful) throw IOException(serverError(it.code, text))
        if (allowEmptyBody && text.isBlank() && T::class == Unit::class) { @Suppress("UNCHECKED_CAST") return Unit as T }
        json.decodeFromString<T>(text)
    }

    private fun serverError(code: Int, body: String): String {
        val detail = runCatching { body.substringAfter("\"detail\":").trim().trimStart('"').substringBefore('"') }.getOrNull()?.takeIf { it.isNotBlank() }
        return detail ?: "Sierra Server request failed ($code)"
    }

    private fun userAgent(): String = "SierraVision/${BuildConfig.VERSION_NAME} Android/${Build.VERSION.SDK_INT}"

    companion object { private val JSON_MEDIA = "application/json; charset=utf-8".toMediaType() }
}
