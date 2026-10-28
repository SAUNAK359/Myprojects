package com.saunak.director.director.live

import com.saunak.director.director.DirectorStrategy
import com.saunak.director.director.InterventionPolicy
import com.saunak.director.engine.subject.NormalizedPoint
import com.saunak.director.engine.subject.NormalizedRect
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
enum class GoalPriority { HARD, IMPORTANT, PREFERENCE }

@Serializable
enum class GoalState { UNKNOWN, FAIL, ACCEPTABLE, IDEAL }

@Serializable
enum class DirectorActor { PHOTOGRAPHER, SUBJECT, OBJECT, ANY, NONE }

@Serializable
data class DirectorRect(
    val left: Double,
    val top: Double,
    val right: Double,
    val bottom: Double
) {
    init {
        require(listOf(left, top, right, bottom).all(Double::isFinite))
        require(left in 0.0..1.0 && top in 0.0..1.0 && right in 0.0..1.0 && bottom in 0.0..1.0)
        require(right > left && bottom > top)
    }
    val width: Double get() = right - left
    val height: Double get() = bottom - top
    val centerX: Double get() = (left + right) * 0.5
    val centerY: Double get() = (top + bottom) * 0.5
    fun normalizedRect() = NormalizedRect(left, top, right, bottom)

    companion object {
        fun from(value: NormalizedRect) = DirectorRect(value.left, value.top, value.right, value.bottom)
    }
}

@Serializable
data class DirectorKeypointTarget(
    val index: Int,
    val x: Double,
    val y: Double,
    @SerialName("confidence_min") val confidenceMin: Double = 0.35
) {
    init {
        require(index >= 0)
        require(x in 0.0..1.0 && y in 0.0..1.0)
        require(confidenceMin in 0.0..1.0)
    }
}

@Serializable
data class PrimaryEntityTarget(
    @SerialName("entity_id") val entityId: String,
    @SerialName("semantic_label") val semanticLabel: String,
    @SerialName("entity_kind") val entityKind: String,
    @SerialName("current_box") val currentBox: DirectorRect,
    @SerialName("target_box") val targetBox: DirectorRect,
    @SerialName("movable_actor") val movableActor: DirectorActor = DirectorActor.ANY,
    @SerialName("min_iou") val minIou: Double = 0.58,
    @SerialName("center_tolerance") val centerTolerance: Double = 0.055,
    @SerialName("scale_tolerance") val scaleTolerance: Double = 0.10,
    @SerialName("visibility_min") val visibilityMin: Double = 0.78
) {
    init {
        require(entityId.isNotBlank())
        require(semanticLabel.isNotBlank())
        require(entityKind.isNotBlank())
        require(minIou in 0.0..1.0)
        require(centerTolerance in 0.005..0.30)
        require(scaleTolerance in 0.01..0.50)
        require(visibilityMin in 0.0..1.0)
    }
}

@Serializable
data class DirectorGoalNode(
    val id: String,
    val type: String,
    val priority: GoalPriority,
    val required: Boolean,
    val weight: Double,
    @SerialName("entity_id") val entityId: String? = null,
    @SerialName("target_min") val targetMin: Double? = null,
    @SerialName("target_max") val targetMax: Double? = null,
    @SerialName("ideal_tolerance") val idealTolerance: Double = 0.35,
    @SerialName("acceptable_tolerance") val acceptableTolerance: Double = 1.0,
    @SerialName("min_confidence") val minConfidence: Double = 0.45,
    @SerialName("target_keypoints") val targetKeypoints: List<DirectorKeypointTarget> = emptyList(),
    val actor: DirectorActor = DirectorActor.NONE,
    @SerialName("instruction_low") val instructionLow: String = "",
    @SerialName("instruction_high") val instructionHigh: String = "",
    @SerialName("instruction_hold") val instructionHold: String = "Hold there",
    @SerialName("depends_on") val dependsOn: List<String> = emptyList(),
    val invalidates: List<String> = emptyList(),
    @SerialName("stability_ms") val stabilityMs: Long = 450L
) {
    init {
        require(id.isNotBlank() && type.isNotBlank())
        require(weight.isFinite() && weight in 0.0..10.0)
        require(targetMin == null || targetMin.isFinite())
        require(targetMax == null || targetMax.isFinite())
        require(targetMin == null || targetMax == null || targetMax >= targetMin)
        require(idealTolerance.isFinite() && idealTolerance in 0.0..1.0)
        require(acceptableTolerance.isFinite() && acceptableTolerance in idealTolerance..2.0)
        require(minConfidence in 0.0..1.0)
        require(stabilityMs in 100L..2500L)
    }
}

@Serializable
data class DirectorReadinessPolicy(
    @SerialName("important_score_min") val importantScoreMin: Double = 0.84,
    @SerialName("required_node_floor") val requiredNodeFloor: Double = 0.60,
    @SerialName("valid_fraction_min") val validFractionMin: Double = 0.82,
    @SerialName("base_stability_ms") val baseStabilityMs: Long = 500L,
    @SerialName("max_stability_ms") val maxStabilityMs: Long = 950L,
    @SerialName("enter_factor") val enterFactor: Double = 1.0,
    @SerialName("exit_factor") val exitFactor: Double = 1.25
) {
    init {
        require(importantScoreMin in 0.0..1.0)
        require(requiredNodeFloor in 0.0..1.0)
        require(validFractionMin in 0.5..1.0)
        require(baseStabilityMs in 150L..1500L)
        require(maxStabilityMs in baseStabilityMs..2500L)
        require(enterFactor in 0.5..1.5)
        require(exitFactor in enterFactor..2.0)
    }
}

@Serializable
data class SierraDirectorSpec(
    @SerialName("schema_version") val schemaVersion: String,
    @SerialName("schema_hash") val schemaHash: String,
    @SerialName("goal_summary") val goalSummary: String,
    val strategy: DirectorStrategy = DirectorStrategy.GENERAL,
    @SerialName("intervention_policy") val interventionPolicy: InterventionPolicy = InterventionPolicy.DIRECTED,
    @SerialName("primary_entity") val primaryEntity: PrimaryEntityTarget,
    val goals: List<DirectorGoalNode>,
    val readiness: DirectorReadinessPolicy = DirectorReadinessPolicy()
) {
    init {
        require(schemaVersion.isNotBlank() && schemaHash.isNotBlank())
        require(goalSummary.isNotBlank())
        require(goals.isNotEmpty())
        require(goals.map { it.id }.distinct().size == goals.size)
    }
}

data class GoalMeasurement(
    val nodeId: String,
    val value: Double?,
    val confidence: Double,
    val timestampNs: Long,
    val source: String,
    val errorRatio: Double? = null,
    val state: GoalState = GoalState.UNKNOWN
)

data class GoalEvaluation(
    val node: DirectorGoalNode,
    val measurement: GoalMeasurement,
    val satisfaction: Double,
    val stable: Boolean
)

data class DirectorOverlayState(
    val targetBox: DirectorRect? = null,
    val currentBox: DirectorRect? = null,
    val targetKeypoints: List<DirectorKeypointTarget> = emptyList(),
    val currentKeypoints: List<NormalizedPoint> = emptyList(),
    val sourceWidth: Int = 0,
    val sourceHeight: Int = 0,
    val confidence: Double = 0.0,
    val activeGoalType: String? = null,
    val showTarget: Boolean = false
)

data class CoachingInstruction(
    val nodeId: String,
    val text: String,
    val actor: DirectorActor,
    val urgency: Double,
    val directionX: Double = 0.0,
    val directionY: Double = 0.0,
    val achieved: Boolean = false
)

data class LiveDirectorState(
    val running: Boolean = false,
    val entityLocked: Boolean = false,
    val readinessScore: Double = 0.0,
    val hardPass: Boolean = false,
    val stable: Boolean = false,
    val captureReady: Boolean = false,
    val stableSinceNs: Long? = null,
    val instruction: CoachingInstruction? = null,
    val overlay: DirectorOverlayState = DirectorOverlayState(),
    val evaluations: List<GoalEvaluation> = emptyList(),
    val status: String = "Director idle"
)

object DirectorGoalTypes {
    const val ENTITY_PLACEMENT = "entity_placement"
    const val SUBJECT_SCALE = "subject_scale"
    const val ENTITY_VISIBILITY = "entity_visibility"
    const val SUBJECT_LUMINANCE = "subject_luminance"
    const val LIGHT_BALANCE_X = "light_balance_x"
    const val SUBJECT_BACKGROUND_EV = "subject_background_ev"
    const val BACKGROUND_DISTRACTION = "background_distraction"
    const val MOTION_STABILITY = "motion_stability"
    const val TECHNICAL_SHARPNESS = "technical_sharpness"
    const val BACKGROUND_SEPARATION = "background_separation"
    const val POSE_KEYPOINT_ALIGNMENT = "pose_keypoint_alignment"
    const val FACE_PLACEMENT = "face_placement"
}
