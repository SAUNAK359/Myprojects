package com.saunak.director.backend

import com.saunak.director.director.PlanOutput
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

interface SierraBackendGateway {
    val configured: Boolean
    val authenticated: Boolean

    suspend fun signUp(request: SierraSignUpRequest): Result<SierraTokenPair>
    suspend fun signIn(request: SierraSignInRequest): Result<SierraTokenPair>
    suspend fun session(): Result<SierraUserView>
    suspend fun credits(): Result<SierraCredits>
    suspend fun createAnalysisJob(request: SierraAnalysisJobRequest): Result<SierraJobView>
    suspend fun job(jobId: String): Result<SierraJobView>
    suspend fun complete(jobId: String, captureId: String): Result<SierraJobView>
    suspend fun cancel(jobId: String): Result<SierraJobView>
    suspend fun signOut(): Result<Unit>
}

@Serializable
data class SierraSignUpRequest(
    @SerialName("display_name") val displayName: String,
    val email: String,
    val phone: String,
    val dob: String,
    val password: String
)

@Serializable
data class SierraSignInRequest(
    val email: String,
    val password: String
)

@Serializable
data class SierraUserView(
    val id: String,
    @SerialName("display_name") val displayName: String,
    val email: String,
    val phone: String,
    val role: String,
    @SerialName("credit_balance") val creditBalance: Int,
    @SerialName("credit_reserved") val creditReserved: Int,
    @SerialName("available_credits") val availableCredits: Int,
    @SerialName("created_at") val createdAt: String
)

@Serializable
data class SierraTokenPair(
    @SerialName("access_token") val accessToken: String,
    @SerialName("refresh_token") val refreshToken: String,
    @SerialName("token_type") val tokenType: String = "bearer",
    @SerialName("access_expires_in_seconds") val accessExpiresInSeconds: Int,
    val user: SierraUserView
)

@Serializable
data class SierraRefreshRequest(
    @SerialName("refresh_token") val refreshToken: String
)

@Serializable
data class SierraCredits(
    val balance: Int,
    val reserved: Int,
    val available: Int
)

data class SierraAnalysisJobRequest(
    val imageJpeg: ByteArray,
    val idempotencyKey: String,
    val cameraContextJson: String,
    val deviceCapabilitiesJson: String,
    val localQualityJson: String,
    val skillContext: String
)

@Serializable
data class SierraJobView(
    val id: String,
    val state: SierraJobState,
    val plan: PlanOutput? = null,
    @SerialName("error_code") val errorCode: String? = null,
    @SerialName("error_message") val errorMessage: String? = null,
    @SerialName("created_at") val createdAt: String,
    @SerialName("ready_expires_at") val readyExpiresAt: String? = null,
    val credits: SierraCredits
)

@Serializable
data class SierraCompleteRequest(
    @SerialName("capture_id") val captureId: String
)

@Serializable
enum class SierraJobState {
    @SerialName("queued") QUEUED,
    @SerialName("processing") PROCESSING,
    @SerialName("ready") READY,
    @SerialName("completed") COMPLETED,
    @SerialName("cancelled") CANCELLED,
    @SerialName("expired") EXPIRED,
    @SerialName("failed") FAILED
}

object SierraCreditPolicy {
    const val TRIAL_CREDITS = 5
    const val CREDITS_PER_SUCCESSFUL_AI_PICTURE = 1

    fun shouldConsumeCredit(finalCaptureSucceeded: Boolean, cloudJobReachedReady: Boolean): Boolean =
        finalCaptureSucceeded && cloudJobReachedReady
}
