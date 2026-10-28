package com.saunak.director.director

import com.saunak.director.camera.CameraUiState
import com.saunak.director.director.live.LiveDirectorState
import com.saunak.director.director.live.SierraDirectorSpec
import com.saunak.director.tools.ToolExecutionReceipt
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class ToolCall(@SerialName("id") val id: Int, @SerialName("v") val value: Double? = null, @SerialName("x") val x: Double? = null, @SerialName("y") val y: Double? = null, @SerialName("s") val stringValue: String? = null)

@Serializable
data class ThemeBlendComponent(@SerialName("theme_id") val themeId: String, @SerialName("weight") val weight: Double)

@Serializable
data class PhotographicDirection(
    @SerialName("story") val story: String = "",
    @SerialName("mood") val mood: String = "",
    @SerialName("primary_subject") val primarySubject: String = "",
    @SerialName("narrative_role") val narrativeRole: String = "",
    @SerialName("selected_theme_id") val selectedThemeId: String = "",
    @SerialName("theme_strength") val themeStrength: Double = 0.0,
    @SerialName("theme_confidence") val themeConfidence: Double = 0.0,
    @SerialName("theme_blend") val themeBlend: List<ThemeBlendComponent> = emptyList(),
    @SerialName("atmosphere_priority") val atmospherePriority: Double = 0.5,
    @SerialName("skin_naturalness") val skinNaturalness: Double = 0.8,
    @SerialName("highlight_protection") val highlightProtection: Double = 0.6,
    @SerialName("shadow_depth") val shadowDepth: Double = 0.5,
    @SerialName("midtone_contrast") val midtoneContrast: Double = 0.5,
    @SerialName("local_contrast") val localContrast: Double = 0.5,
    @SerialName("color_intensity") val colorIntensity: Double = 0.5,
    @SerialName("warmth_bias") val warmthBias: Double = 0.0,
    @SerialName("subject_relight_ev") val subjectRelightEv: Double = 0.0,
    @SerialName("background_suppression_ev") val backgroundSuppressionEv: Double = 0.0,
    @SerialName("grain") val grain: Double = 0.0,
    @SerialName("vignette") val vignette: Double = 0.0,
    @SerialName("abstain") val abstain: Boolean = false,
    @SerialName("reason_codes") val reasonCodes: List<String> = emptyList()
)

@Serializable
data class FinalRenderAdjustment(
    @SerialName("theme_strength_delta") val themeStrengthDelta: Double = 0.0,
    @SerialName("midtone_contrast_delta") val midtoneContrastDelta: Double = 0.0,
    @SerialName("local_contrast_delta") val localContrastDelta: Double = 0.0,
    @SerialName("shadow_depth_delta") val shadowDepthDelta: Double = 0.0,
    @SerialName("highlight_protection_delta") val highlightProtectionDelta: Double = 0.0,
    @SerialName("subject_relight_delta_ev") val subjectRelightDeltaEv: Double = 0.0,
    @SerialName("background_suppression_delta_ev") val backgroundSuppressionDeltaEv: Double = 0.0,
    @SerialName("warmth_delta") val warmthDelta: Double = 0.0,
    @SerialName("color_intensity_delta") val colorIntensityDelta: Double = 0.0
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
    @SerialName("photographic_direction") val photographicDirection: PhotographicDirection = PhotographicDirection(),
    @SerialName("director_spec") val directorSpec: SierraDirectorSpec? = null,
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
    @SerialName("delta") val deltaCalls: List<ToolCall> = emptyList(),
    @SerialName("intent_match") val intentMatch: Double = score,
    @SerialName("naturalness") val naturalness: Double = score,
    @SerialName("render_adjustment") val renderAdjustment: FinalRenderAdjustment = FinalRenderAdjustment(),
    @SerialName("render_spec") val renderSpec: SierraRenderSpec? = null
)

enum class DirectorStage { CAMERA_ONLY, READY, CAPTURING, ANALYZING, PLANNING, APPLYING, RECAPTURING, CHECKING, CORRECTING, PREPARED, FINAL_CAPTURING, SAVING, ERROR, OBSERVING }
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
    val recipeCount: Int = 0,
    val showSettings: Boolean = false,
    val showTools: Boolean = false,
    val showGallery: Boolean = false,
    val galleryItems: List<String> = emptyList(),
    val gridEnabled: Boolean = false,
    val levelEnabled: Boolean = false,
    val sierraMode: SierraMode = SierraMode.BETA,
    val aspectRatio: String = "4:3",
    val aspectRatioOverride: String? = null,
    val sceneLabel: String? = null,
    val objective: String? = null,
    val privateMode: Boolean = false,
    val keepOriginal: Boolean = false,
    val serverConfigured: Boolean = false,
    val signedIn: Boolean = false,
    val guestMode: Boolean = false,
    val accountName: String? = null,
    val creditsAvailable: Int? = null,
    val authBusy: Boolean = false,
    val authError: String? = null,
    val aiProvider: String? = null,
    val aiLatencyMs: Long? = null,
    val aiTheme: String? = null,
    val aiThemeStrength: Double? = null,
    val aiFallbackReason: String? = null,
    val finalReviewStatus: String? = null,
    val aiStreamPhase: String? = null,
    val executionReceipts: List<ToolExecutionReceipt> = emptyList(),
    val liveDirector: LiveDirectorState = LiveDirectorState()
)
