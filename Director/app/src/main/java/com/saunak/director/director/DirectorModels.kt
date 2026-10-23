package com.saunak.director.director

import com.saunak.director.camera.CameraUiState
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class ToolCall(
    @SerialName("id") val id: Int,
    @SerialName("v") val value: Double? = null,
    @SerialName("x") val x: Double? = null,
    @SerialName("y") val y: Double? = null,
    @SerialName("s") val stringValue: String? = null
)

@Serializable
data class PlanOutput(
    @SerialName("scene") val scene: String,
    @SerialName("objective") val objective: String,
    @SerialName("branch") val skillBranch: String,
    @SerialName("reference_score") val referenceScore: Double = 0.5,
    @SerialName("reference_accept") val referenceAccept: Boolean = false,
    @SerialName("calls") val toolCalls: List<ToolCall> = emptyList(),
    @SerialName("recapture") val recapture: Boolean = true,
    @SerialName("aspect_ratio") val aspectRatio: String = "4:3",
    @SerialName("crop_center_x") val cropCenterX: Double = 0.5,
    @SerialName("crop_center_y") val cropCenterY: Double = 0.5,
    @SerialName("planner") val planner: String = "sierra_ai"
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

enum class DirectorStage {
    CAMERA_ONLY,
    READY,
    CAPTURING,
    ANALYZING,
    PLANNING,
    APPLYING,
    RECAPTURING,
    CHECKING,
    CORRECTING,
    PREPARED,
    FINAL_CAPTURING,
    SAVING,
    ERROR,
    OBSERVING
}

enum class SierraMode { AI, BETA }

data class DirectorUiState(
    val stage: DirectorStage = DirectorStage.CAMERA_ONLY,
    val aiAvailable: Boolean = false,
    val status: String = "Camera ready",
    val attempt: Int = 0,
    val lastScore: Double? = null,
    val error: String? = null,
    val camera: CameraUiState = CameraUiState(),
    val lastSavedUri: String? = null,
    val hasApiKey: Boolean = false,
    val recipeCount: Int = 0,
    val showSettings: Boolean = false,
    val showTools: Boolean = false,
    val gridEnabled: Boolean = false,
    val levelEnabled: Boolean = false,
    val sierraMode: SierraMode = SierraMode.BETA,
    val aspectRatio: String = "4:3",
    val aspectRatioOverride: String? = null,
    val sceneLabel: String? = null,
    val objective: String? = null,
    val privateMode: Boolean = false,
    val keepOriginal: Boolean = false
)
