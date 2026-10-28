package com.saunak.director.ai

import android.os.SystemClock
import com.saunak.director.director.DirectorStrategy
import com.saunak.director.director.InterventionPolicy
import com.saunak.director.director.live.DirectorGoalTypes
import com.saunak.director.engine.SierraEngine
import com.saunak.director.engine.models.SierraShippingManifests
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject

/** Single capability-aware source of truth for Gemini Director output and local validation. */
class DirectorSchemaCompiler(private val engine: SierraEngine = SierraEngine.get()) {
    data class Contract(
        val version: String,
        val hash: String,
        val allowedGoalTypes: Set<String>,
        val jsonSchema: JSONObject,
        val promptContract: String
    )

    fun compile(forceRefresh: Boolean = false): Contract {
        val now = SystemClock.elapsedRealtime()
        if (!forceRefresh) cached?.takeIf { now - it.createdAtMs in 0 until CACHE_TTL_MS }?.let { return it.contract }
        synchronized(CACHE_LOCK) {
            val lockedNow = SystemClock.elapsedRealtime()
            if (!forceRefresh) cached?.takeIf { lockedNow - it.createdAtMs in 0 until CACHE_TTL_MS }?.let { return it.contract }
            val contract = compileVerified()
            cached = CachedContract(lockedNow, contract)
            return contract
        }
    }

    private fun compileVerified(): Contract {
        val pose = engine.modelArtifacts.findVerified(SierraShippingManifests.rtmposeSmall) != null
        val depth = engine.modelArtifacts.findVerified(SierraShippingManifests.depthAnythingV2Small) != null
        val segmentation = engine.modelArtifacts.findVerified(SierraShippingManifests.personSegmentation) != null
        val face = engine.modelArtifacts.findVerified(SierraShippingManifests.yunet2026May) != null

        val goals = linkedSetOf(
            DirectorGoalTypes.ENTITY_PLACEMENT,
            DirectorGoalTypes.SUBJECT_SCALE,
            DirectorGoalTypes.ENTITY_VISIBILITY,
            DirectorGoalTypes.SUBJECT_LUMINANCE,
            DirectorGoalTypes.LIGHT_BALANCE_X,
            DirectorGoalTypes.SUBJECT_BACKGROUND_EV,
            DirectorGoalTypes.BACKGROUND_DISTRACTION,
            DirectorGoalTypes.MOTION_STABILITY,
            DirectorGoalTypes.TECHNICAL_SHARPNESS
        )
        if (depth) goals += DirectorGoalTypes.BACKGROUND_SEPARATION
        if (pose) goals += DirectorGoalTypes.POSE_KEYPOINT_ALIGNMENT
        if (face) goals += DirectorGoalTypes.FACE_PLACEMENT

        val canonical = buildString {
            append(VERSION).append('|')
            append("face=").append(face).append('|')
            append("pose=").append(pose).append('|')
            append("depth=").append(depth).append('|')
            append("seg=").append(segmentation).append('|')
            append(goals.sorted().joinToString(","))
        }
        val hash = sha256(canonical)
        return Contract(
            version = VERSION,
            hash = hash,
            allowedGoalTypes = goals,
            jsonSchema = schema(VERSION, hash, goals),
            promptContract = buildString {
                appendLine("SIERRA DIRECTOR CONTRACT $VERSION hash=$hash")
                appendLine("allowed_goal_types=${goals.sorted().joinToString()}")
                appendLine("strategies=${DirectorStrategy.entries.joinToString()}")
                appendLine("intervention_policies=${InterventionPolicy.entries.joinToString()}")
                appendLine("face_landmarks=$face; pose_landmarks=$pose; relative_depth=$depth; person_mask=$segmentation")
                appendLine("Gemini defines targets and strategy; Sierra beta only measures, compares, coaches and gates readiness.")
                appendLine("Unknown evidence is never success. Do not emit a goal type absent from allowed_goal_types.")
                appendLine("Use normalized final-composition coordinates 0..1. Never output screen pixels or metric distance unless a metric sensor capability is explicitly present.")
                appendLine("Every target must include ideal/acceptable tolerance, confidence floor, priority and stability duration.")
                appendLine("Primary entity current_box must locate the exact entity in THIS frame; target_box is where the same entity should be in the intended photograph.")
                appendLine("CANDID with OBSERVE/MINIMAL must preserve the moment: no hard/required pose goal and no blocking subject-reposition goal. Prefer photographer movement, framing, timing, exposure and motion safety.")
                appendLine("BACKGROUND_DISTRACTION is an observable clutter/saliency target; do not imply generative object removal. Sierra coaches physical reframing/repositioning only.")
                appendLine("SUBJECT_BACKGROUND_EV is measured image-space luminance separation, not a fabricated light meter reading.")
                appendLine("Use at most $MAX_GOALS goal nodes and at most $MAX_GRAPH_EDGES dependencies/invalidations per node; include only materially useful goals.")
            }.trim()
        )
    }

    private fun schema(version: String, hash: String, allowed: Set<String>): JSONObject {
        fun number(min: Double? = null, max: Double? = null) = JSONObject().put("type", "number").apply { min?.let { put("minimum", it) }; max?.let { put("maximum", it) } }
        fun integer(min: Int? = null) = JSONObject().put("type", "integer").apply { min?.let { put("minimum", it) } }
        fun string() = JSONObject().put("type", "string")
        fun bool() = JSONObject().put("type", "boolean")
        fun enum(values: List<String>) = JSONObject().put("type", "string").put("enum", JSONArray(values))
        fun nullableNumber() = JSONObject().put("type", JSONArray(listOf("number", "null")))
        fun nullableString() = JSONObject().put("type", JSONArray(listOf("string", "null")))
        fun array(item: JSONObject, max: Int? = null) = JSONObject().put("type", "array").put("items", item).apply { max?.let { put("maxItems", it) } }
        fun obj(required: List<String>, props: Map<String, JSONObject>) = JSONObject().put("type", "object").put("required", JSONArray(required)).put("properties", JSONObject(props)).put("additionalProperties", false)

        val rect = obj(listOf("left","top","right","bottom"), mapOf(
            "left" to number(0.0,1.0), "top" to number(0.0,1.0), "right" to number(0.0,1.0), "bottom" to number(0.0,1.0)
        ))
        val entity = obj(
            listOf("entity_id","semantic_label","entity_kind","current_box","target_box","movable_actor","min_iou","center_tolerance","scale_tolerance","visibility_min"),
            mapOf(
                "entity_id" to string(), "semantic_label" to string(), "entity_kind" to string(), "current_box" to rect, "target_box" to rect,
                "movable_actor" to enum(listOf("PHOTOGRAPHER","SUBJECT","OBJECT","ANY","NONE")), "min_iou" to number(0.0,1.0),
                "center_tolerance" to number(0.005,0.30), "scale_tolerance" to number(0.01,0.50), "visibility_min" to number(0.0,1.0)
            )
        )
        val keypoint = obj(listOf("index","x","y","confidence_min"), mapOf(
            "index" to integer(0), "x" to number(0.0,1.0), "y" to number(0.0,1.0), "confidence_min" to number(0.0,1.0)
        ))
        val goal = obj(
            listOf("id","type","priority","required","weight","entity_id","target_min","target_max","ideal_tolerance","acceptable_tolerance","min_confidence","target_keypoints","actor","instruction_low","instruction_high","instruction_hold","depends_on","invalidates","stability_ms"),
            mapOf(
                "id" to string(), "type" to enum(allowed.sorted()), "priority" to enum(listOf("HARD","IMPORTANT","PREFERENCE")),
                "required" to bool(), "weight" to number(0.0,10.0), "entity_id" to nullableString(), "target_min" to nullableNumber(), "target_max" to nullableNumber(),
                "ideal_tolerance" to number(0.0,1.0), "acceptable_tolerance" to number(0.0,2.0), "min_confidence" to number(0.0,1.0),
                "target_keypoints" to array(keypoint, MAX_POSE_KEYPOINTS), "actor" to enum(listOf("PHOTOGRAPHER","SUBJECT","OBJECT","ANY","NONE")),
                "instruction_low" to string(), "instruction_high" to string(), "instruction_hold" to string(), "depends_on" to array(string(), MAX_GRAPH_EDGES),
                "invalidates" to array(string(), MAX_GRAPH_EDGES), "stability_ms" to integer(100)
            )
        )
        val readiness = obj(
            listOf("important_score_min","required_node_floor","valid_fraction_min","base_stability_ms","max_stability_ms","enter_factor","exit_factor"),
            mapOf(
                "important_score_min" to number(0.0,1.0), "required_node_floor" to number(0.0,1.0), "valid_fraction_min" to number(0.5,1.0),
                "base_stability_ms" to integer(150), "max_stability_ms" to integer(150), "enter_factor" to number(0.5,1.5), "exit_factor" to number(0.5,2.0)
            )
        )
        return obj(
            listOf("schema_version","schema_hash","goal_summary","strategy","intervention_policy","primary_entity","goals","readiness"),
            mapOf(
                "schema_version" to enum(listOf(version)),
                "schema_hash" to enum(listOf(hash)),
                "goal_summary" to string(),
                "strategy" to enum(DirectorStrategy.entries.map { it.name }),
                "intervention_policy" to enum(InterventionPolicy.entries.map { it.name }),
                "primary_entity" to entity,
                "goals" to array(goal, MAX_GOALS),
                "readiness" to readiness
            )
        )
    }

    private data class CachedContract(val createdAtMs: Long, val contract: Contract)

    companion object {
        const val VERSION = "director.v2"
        const val MAX_GOALS = 14
        const val MAX_GRAPH_EDGES = 8
        const val MAX_POSE_KEYPOINTS = 17
        private const val CACHE_TTL_MS = 15_000L
        private val CACHE_LOCK = Any()
        @Volatile private var cached: CachedContract? = null
        private fun sha256(value: String): String = MessageDigest.getInstance("SHA-256").digest(value.toByteArray()).joinToString("") { "%02x".format(it) }
    }
}
