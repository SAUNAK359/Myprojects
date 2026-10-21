package com.saunak.director.director

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class ToolCall(@SerialName("id") val id: Int, @SerialName("v") val value: Double? = null, @SerialName("s") val stringValue: String? = null)

@Serializable
data class PlanOutput(
    @SerialName("scene") val scene: String,
    @SerialName("objective") val objective: String,
    @SerialName("branch") val skillBranch: String,
    @SerialName("calls") val toolCalls: List<ToolCall> = emptyList(),
    @SerialName("capture") val capture: Boolean = true
)

@Serializable
data class EvaluationOutput(
    @SerialName("score") val score: Double,
    @SerialName("technical") val technical: Double = score,
    @SerialName("composition") val composition: Double = score,
    @SerialName("subject") val subject: Double = score,
    @SerialName("aesthetic") val aesthetic: Double = score,
    @SerialName("accept") val accept: Boolean,
    @SerialName("issues") val issues: List<String> = emptyList(),
    @SerialName("delta") val deltaCalls: List<ToolCall> = emptyList()
)

enum class DirectorStage { CAMERA_ONLY, AI_DOWNLOAD_REQUIRED, LOADING_AI, OBSERVING, PLANNING, APPLYING, CAPTURING, CHECKING, CORRECTING, SAVING, ERROR }

data class DirectorUiState(
    val stage: DirectorStage = DirectorStage.CAMERA_ONLY,
    val aiAvailable: Boolean = false,
    val status: String = "Camera ready",
    val attempt: Int = 0,
    val lastScore: Double? = null,
    val modelProgress: Float? = null,
    val requestModelDownload: Boolean = false,
    val error: String? = null
)
