package com.saunak.director.storage

import android.content.Context
import android.os.Build
import com.saunak.director.BuildConfig
import com.saunak.director.director.DirectorUiState
import com.saunak.director.director.PipelineEvent
import com.saunak.director.tools.ToolExecutionReceipt
import com.saunak.director.tools.ToolExecutionStatus
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/**
 * Persists observable decisions and execution evidence only. Hidden model reasoning is never stored.
 * Records are bounded JSON files so production diagnosis does not grow app-private storage without limit.
 */
class ShotAuditStore(context: Context) {
    private val dir = File(context.applicationContext.filesDir, "director/audit").apply { mkdirs() }

    @Synchronized
    fun persist(state: DirectorUiState, events: List<PipelineEvent>): File? = runCatching {
        val savedUri = state.lastSavedUri ?: return null
        val now = System.currentTimeMillis()
        val receipts = state.executionReceipts
        val live = state.liveDirector
        val record = JSONObject().apply {
            put("schema_version", 2)
            put("created_at_ms", now)
            put("build", JSONObject().apply {
                put("application_id", BuildConfig.APPLICATION_ID)
                put("version_name", BuildConfig.VERSION_NAME)
                put("version_code", BuildConfig.VERSION_CODE)
                put("debug", BuildConfig.DEBUG)
                put("android_sdk", Build.VERSION.SDK_INT)
                put("manufacturer", Build.MANUFACTURER)
                put("model", Build.MODEL)
            })
            put("shot", JSONObject().apply {
                put("mode", state.sierraMode.name)
                put("scene", state.sceneLabel)
                put("objective", state.objective)
                put("aspect_ratio", state.aspectRatio)
                put("aspect_ratio_override", state.aspectRatioOverride)
                put("keep_original", state.keepOriginal)
                put("private_mode", state.privateMode)
                put("final_uri", savedUri)
                put("final_review_status", state.finalReviewStatus)
            })
            put("ai", JSONObject().apply {
                put("available", state.aiAvailable)
                put("provider", state.aiProvider)
                put("latency_ms", state.aiLatencyMs)
                put("stream_phase", state.aiStreamPhase)
                put("fallback_reason", state.aiFallbackReason)
                put("theme", state.aiTheme)
                put("theme_strength", state.aiThemeStrength)
            })
            put("camera", JSONObject().apply {
                put("lens_facing", state.camera.lensFacing)
                put("zoom", state.camera.zoom)
                put("zoom_min", state.camera.minZoom)
                put("zoom_max", state.camera.maxZoom)
                put("exposure_supported", state.camera.exposureSupported)
                put("exposure_index", state.camera.exposureIndex)
                put("exposure_step_ev", state.camera.exposureStepEv)
                put("torch_supported", state.camera.torchSupported)
                put("torch_enabled", state.camera.torchEnabled)
                put("flash_mode", state.camera.flashMode.wireName)
                put("extensions", JSONArray(state.camera.extensions.map { it.wireName }.sorted()))
            })
            put("live_director", JSONObject().apply {
                put("running", live.running)
                put("entity_locked", live.entityLocked)
                put("readiness_score", live.readinessScore)
                put("hard_pass", live.hardPass)
                put("stable", live.stable)
                put("capture_ready", live.captureReady)
                put("stable_since_ns", live.stableSinceNs)
                put("status", live.status)
                put("active_instruction", live.instruction?.text)
                put("goal_evaluations", JSONArray(live.evaluations.take(20).map { evaluation ->
                    JSONObject().apply {
                        put("id", evaluation.node.id)
                        put("type", evaluation.node.type)
                        put("priority", evaluation.node.priority.name)
                        put("required", evaluation.node.required)
                        put("state", evaluation.measurement.state.name)
                        put("value", evaluation.measurement.value)
                        put("confidence", evaluation.measurement.confidence)
                        put("source", evaluation.measurement.source)
                        put("satisfaction", evaluation.satisfaction)
                        put("stable", evaluation.stable)
                    }
                }))
            })
            put("execution", JSONObject().apply {
                put("receipt_count", receipts.size)
                put("applied_count", receipts.count { it.status == ToolExecutionStatus.APPLIED })
                put("clamped_count", receipts.count { it.status == ToolExecutionStatus.CLAMPED })
                put("failed_count", receipts.count { it.status == ToolExecutionStatus.FAILED })
                put("rejected_count", receipts.count { it.status == ToolExecutionStatus.REJECTED })
                put("receipts", JSONArray(receipts.map(::receiptJson)))
            })
            put("timeline", JSONArray(events.takeLast(MAX_EVENTS).map(::eventJson)))
        }
        val file = File(dir, "shot-$now.json")
        file.writeText(record.toString(2))
        prune()
        file
    }.getOrNull()

    private fun receiptJson(value: ToolExecutionReceipt): JSONObject = JSONObject().apply {
        put("timestamp_ms", value.timestampMs)
        put("status", value.status.name)
        put("requested", value.requested.toString())
        put("resolved", value.resolved?.toString())
        put("actual", value.actual)
        put("reason", value.reason)
    }

    private fun eventJson(value: PipelineEvent): JSONObject = JSONObject().apply {
        put("id", value.id)
        put("timestamp_ms", value.timestampMs)
        put("actor", value.actor.name)
        put("stage", value.stage.name)
        put("state", value.state.name)
        put("title", value.title)
        put("detail", value.detail)
    }

    private fun prune() {
        val files = dir.listFiles()?.filter { it.isFile && it.name.endsWith(".json") }?.sortedByDescending { it.lastModified() }.orEmpty()
        files.drop(MAX_RECORDS).forEach { runCatching { it.delete() } }
    }

    companion object {
        private const val MAX_RECORDS = 80
        private const val MAX_EVENTS = 120
    }
}
