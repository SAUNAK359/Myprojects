package com.saunak.director.tools

import com.saunak.director.camera.CameraEngine
import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.camera.DirectorExtension
import com.saunak.director.camera.StillFlashMode
import com.saunak.director.director.ToolCall
import com.saunak.director.engine.SierraEngine
import com.saunak.director.engine.subject.PredictiveSubjectPolicy
import com.saunak.director.processing.Look
import com.saunak.director.processing.LookSpec
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class ToolExecutor(private val camera: CameraEngine, private val validator: ToolValidator = ToolValidator()) {
    data class Rejected(val call: ToolCall, val reason: String)
    data class Result(
        val applied: List<ToolCall>,
        val rejected: List<Rejected>,
        val look: LookSpec,
        val requiresRecapture: Boolean,
        val predictiveRetargeted: Boolean = false
    )
    private val mutationMutex = Mutex()

    suspend fun apply(calls: List<ToolCall>, capabilities: CapabilityRegistry, currentLook: LookSpec = LookSpec()): Result = mutationMutex.withLock {
        val effectiveCaps = capabilities.copy(activeExtension = camera.snapshot().extension)
        val validated = validator.validate(calls, effectiveCaps)
        val applied = mutableListOf<ToolCall>()
        val rejected = validated.rejected.mapTo(mutableListOf()) { Rejected(it.call, it.reason) }
        var look = currentLook
        val ordered = validated.accepted.sortedBy { if (it.id == ToolIds.EXTENSION) 0 else if (it.id in ToolIds.LOOK_IDS) 2 else 1 }
        for (call in ordered) {
            runCatching {
                when (call.id) {
                    ToolIds.FOCUS -> camera.aiFocusNormalized(call.x!!.toFloat(), call.y!!.toFloat())
                    ToolIds.ZOOM -> camera.aiSetZoom(call.value!!.toFloat())
                    ToolIds.EXPOSURE_EV -> camera.aiSetExposureEv(call.value!!.toFloat())
                    ToolIds.TORCH -> camera.aiSetTorch(call.value!! >= 0.5)
                    ToolIds.FLASH -> {
                        val mode = checkNotNull(StillFlashMode.fromWire(call.stringValue))
                        if (mode != StillFlashMode.OFF) camera.aiSetTorch(false)
                        camera.aiSetFlash(mode)
                    }
                    ToolIds.EXTENSION -> camera.aiSetExtension(checkNotNull(DirectorExtension.fromWire(call.stringValue)))
                    ToolIds.LOOK_NATURAL -> look = LookSpec(Look.NATURAL, call.value?.toFloat() ?: 0f)
                    ToolIds.LOOK_WARM -> look = LookSpec(Look.WARM, call.value?.toFloat() ?: .65f)
                    ToolIds.LOOK_COOL -> look = LookSpec(Look.COOL, call.value?.toFloat() ?: .65f)
                    ToolIds.LOOK_CINEMATIC_WARM -> look = LookSpec(Look.CINEMATIC_WARM, call.value?.toFloat() ?: .68f)
                    ToolIds.LOOK_CINEMATIC_COOL -> look = LookSpec(Look.CINEMATIC_COOL, call.value?.toFloat() ?: .68f)
                    ToolIds.LOOK_MONOCHROME -> look = LookSpec(Look.MONOCHROME, call.value?.toFloat() ?: 1f)
                    ToolIds.LOOK_SOFT_PORTRAIT -> look = LookSpec(Look.SOFT_PORTRAIT, call.value?.toFloat() ?: .58f)
                    ToolIds.LOOK_LANDSCAPE_VIVID -> look = LookSpec(Look.LANDSCAPE_VIVID, call.value?.toFloat() ?: .55f)
                    ToolIds.LOOK_SOCIAL_POP -> look = LookSpec(Look.SOCIAL_POP, call.value?.toFloat() ?: .52f)
                    ToolIds.LOOK_FILM_TEAL_ORANGE -> look = LookSpec(Look.FILM_TEAL_ORANGE, call.value?.toFloat() ?: .58f)
                    ToolIds.LOOK_GOLDEN_HOUR -> look = LookSpec(Look.GOLDEN_HOUR, call.value?.toFloat() ?: .56f)
                    ToolIds.LOOK_MOODY -> look = LookSpec(Look.MOODY, call.value?.toFloat() ?: .50f)
                    ToolIds.LOOK_CLEAN_PORTRAIT -> look = LookSpec(Look.CLEAN_PORTRAIT, call.value?.toFloat() ?: .46f)
                    ToolIds.LOOK_NIGHT_NEON -> look = LookSpec(Look.NIGHT_NEON, call.value?.toFloat() ?: .54f)
                    else -> error("Unknown tool ${call.id}")
                }
            }.onSuccess { applied += call }.onFailure { rejected += Rejected(call, it.message ?: "execution_failed") }
        }

        // System-owned predictive 3A runs only when the plan did not explicitly choose a focus point.
        // CameraX's existing aiFocusNormalized action meters AF+AE+AWB together, so the orchestrator's
        // normal requiresRecapture path will wait for convergence and verify the result afterward.
        val explicitFocusApplied = applied.any { it.id == ToolIds.FOCUS }
        val extensionActive = camera.snapshot().extension != DirectorExtension.NONE
        val predictiveRetargeted = if (!explicitFocusApplied && !extensionActive && effectiveCaps.focusMeteringSupported) {
            val track = SierraEngine.getOrNull()?.subjects?.latest()
            val directive = track?.let(PredictiveSubjectPolicy::evaluate)
            if (directive != null && directive.shouldRetargetThreeA) {
                runCatching {
                    camera.aiFocusNormalized(
                        directive.autofocusPoint.x.toFloat(),
                        directive.autofocusPoint.y.toFloat()
                    )
                }.isSuccess
            } else false
        } else false

        val requiresRecapture = predictiveRetargeted || applied.any { it.id in ToolIds.CAMERA_IDS }
        Result(applied, rejected, look, requiresRecapture, predictiveRetargeted)
    }

    suspend fun restoreUserState() = mutationMutex.withLock { camera.restoreUserState() }
}
