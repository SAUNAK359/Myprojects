package com.saunak.director.ai

import android.graphics.Bitmap
import com.saunak.director.camera.CameraSnapshot
import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.director.EvaluationOutput
import com.saunak.director.director.PlanOutput

data class VlmDiagnostics(
    val provider: String,
    val latencyMs: Long? = null,
    val lastError: String? = null
)

data class VlmStreamProgress(
    val phase: String,
    val detail: String,
    val receivedChars: Int = 0,
    val interactionId: String? = null
)

interface VlmRuntime {
    suspend fun load(modelPath: String, projectorPath: String): Result<Unit>
    suspend fun syncDeviceFoundation(profile: DeviceFoundationProfile): Result<Unit> = Result.success(Unit)
    suspend fun plan(image: Bitmap, capabilities: CapabilityRegistry, camera: CameraSnapshot, skillContext: String): Result<PlanOutput>
    suspend fun evaluate(image: Bitmap, objective: String, attempt: Int, camera: CameraSnapshot, previousIssues: List<String>, evaluationContext: String, capabilities: CapabilityRegistry): Result<EvaluationOutput>
    fun setMaxImageDimension(pixels: Int)
    fun setStreamProgressListener(listener: ((VlmStreamProgress) -> Unit)?) {}
    fun cancel()
    fun isReady(): Boolean
    fun waitForSemanticPlan(): Boolean = false
    fun supportsFinalReview(): Boolean = false
    fun diagnostics(): VlmDiagnostics = VlmDiagnostics(provider = "unknown")
    fun unload()
    fun close()
}
