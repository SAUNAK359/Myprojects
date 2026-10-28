package com.saunak.director.storage

import android.content.Context
import com.saunak.director.director.DirectorUiState
import com.saunak.director.director.PipelineEvent
import com.saunak.director.tools.ToolExecutionReceipt
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
        val record = JSONObject().apply {
            put("schema_version", 1)
            put("created_at_ms", now)
            put("mode", state.sierraMode.name)
            put("scene", state.sceneLabel)
            put("objective", state.objective)
            put("ai_provider", state.aiProvider)
            put("ai_latency_ms", state.aiLatencyMs)
            put("ai_fallback_reason", state.aiFallbackReason)
            put("final_review_status", state.finalReviewStatus)
            put("final_uri", savedUri)
            put("readiness_score", state.liveDirector.readinessScore)
            put("capture_ready", state.liveDirector.captureReady)
            put("entity_locked", state.liveDirector.entityLocked)
            put("camera", state.camera.toString())
            put("execution_receipts", JSONArray(state.executionReceipts.map(::receiptJson)))
            put("timeline", JSONArray(events.map(::eventJson)))
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

    companion object { private const val MAX_RECORDS = 80 }
}
