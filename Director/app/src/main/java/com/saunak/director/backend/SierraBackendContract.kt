package com.saunak.director.backend

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Network-neutral contracts for the future Sierra laptop control plane.
 *
 * This module intentionally contains no URL, token, demo user, credit balance, or implementation.
 * The laptop backend will remain authoritative for identity, credits, job lifecycle and cloud
 * inference. Sierra Vision can therefore ship Sierra β now without pretending local state is
 * payment/auth authority.
 */
interface SierraBackendGateway {
    val configured: Boolean

    suspend fun session(): Result<SierraUserSession>
    suspend fun startPhotoJob(request: StartPhotoJobRequest): Result<PhotoJobLease>
    suspend fun submitAnalysis(request: AnalysisRequest): Result<AnalysisResponse>
    suspend fun completePhotoJob(request: CompletePhotoJobRequest): Result<PhotoJobReceipt>
    suspend fun cancelPhotoJob(jobId: String, reason: String): Result<Unit>
}

@Serializable
data class SierraUserSession(
    @SerialName("user_id") val userId: String,
    @SerialName("display_name") val displayName: String? = null,
    @SerialName("credits_available") val creditsAvailable: Int,
    @SerialName("account_status") val accountStatus: String,
    @SerialName("access_token_expires_at_ms") val accessTokenExpiresAtMs: Long
)

@Serializable
data class StartPhotoJobRequest(
    @SerialName("device_id") val deviceId: String,
    @SerialName("app_version") val appVersion: String,
    @SerialName("requested_mode") val requestedMode: String = "sierra_ai"
)

@Serializable
data class PhotoJobLease(
    @SerialName("job_id") val jobId: String,
    @SerialName("reservation_id") val reservationId: String,
    @SerialName("state") val state: PhotoJobState,
    @SerialName("expires_at_ms") val expiresAtMs: Long,
    @SerialName("credits_reserved") val creditsReserved: Int = 1
)

@Serializable
data class AnalysisRequest(
    @SerialName("job_id") val jobId: String,
    @SerialName("image_jpeg") val imageJpeg: ByteArray,
    @SerialName("camera_context_json") val cameraContextJson: String,
    @SerialName("device_capabilities_json") val deviceCapabilitiesJson: String,
    @SerialName("local_quality_json") val localQualityJson: String
)

@Serializable
data class AnalysisResponse(
    @SerialName("job_id") val jobId: String,
    @SerialName("state") val state: PhotoJobState,
    @SerialName("plan_json") val planJson: String,
    @SerialName("provider_route_id") val providerRouteId: String? = null,
    @SerialName("latency_ms") val latencyMs: Long? = null
)

@Serializable
data class CompletePhotoJobRequest(
    @SerialName("job_id") val jobId: String,
    @SerialName("reservation_id") val reservationId: String,
    @SerialName("final_capture_succeeded") val finalCaptureSucceeded: Boolean,
    @SerialName("capture_timestamp_ms") val captureTimestampMs: Long
)

@Serializable
data class PhotoJobReceipt(
    @SerialName("job_id") val jobId: String,
    @SerialName("state") val state: PhotoJobState,
    @SerialName("credits_consumed") val creditsConsumed: Int,
    @SerialName("credits_remaining") val creditsRemaining: Int
)

@Serializable
enum class PhotoJobState {
    @SerialName("created") CREATED,
    @SerialName("credit_reserved") CREDIT_RESERVED,
    @SerialName("image_received") IMAGE_RECEIVED,
    @SerialName("analyzing") ANALYZING,
    @SerialName("ready") READY,
    @SerialName("capture_confirmed") CAPTURE_CONFIRMED,
    @SerialName("completed") COMPLETED,
    @SerialName("cancelled") CANCELLED,
    @SerialName("expired") EXPIRED,
    @SerialName("failed") FAILED
}

object SierraCreditPolicy {
    const val CREDITS_PER_SUCCESSFUL_AI_PICTURE = 1

    fun shouldConsumeCredit(finalCaptureSucceeded: Boolean, cloudJobReachedReady: Boolean): Boolean =
        finalCaptureSucceeded && cloudJobReachedReady
}
