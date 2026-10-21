package com.saunak.director.ai

import android.graphics.Bitmap
import com.saunak.director.camera.CameraSnapshot
import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.director.EvaluationOutput
import com.saunak.director.director.PlanOutput

interface VlmRuntime {
    suspend fun load(modelPath: String, projectorPath: String): Result<Unit>
    suspend fun plan(image: Bitmap, capabilities: CapabilityRegistry, camera: CameraSnapshot, skillContext: String): Result<PlanOutput>
    suspend fun evaluate(image: Bitmap, objective: String, attempt: Int, camera: CameraSnapshot, previousIssues: List<String>, evaluationContext: String): Result<EvaluationOutput>
    fun setMaxImageDimension(pixels: Int)
    fun cancel()
    fun isReady(): Boolean
    fun close()
}
