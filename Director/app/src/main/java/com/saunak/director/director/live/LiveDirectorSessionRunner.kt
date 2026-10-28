package com.saunak.director.director.live

import android.graphics.Bitmap
import com.saunak.director.camera.CameraEngine
import com.saunak.director.director.SierraPerceptionFabric
import com.saunak.director.processing.PhotoQuality
import kotlinx.coroutines.delay
import kotlinx.coroutines.withTimeoutOrNull

/**
 * Owns the live preview cadence around SierraLiveDirectorRuntime. It deliberately has no network or
 * creative authority: Gemini's validated spec is immutable for the session.
 */
class LiveDirectorSessionRunner(
    private val camera: CameraEngine,
    private val perception: SierraPerceptionFabric,
    private val runtime: SierraLiveDirectorRuntime = SierraLiveDirectorRuntime()
) {
    data class Result(
        val ready: Boolean,
        val state: LiveDirectorState,
        val reason: String? = null
    )

    fun stop(): LiveDirectorState = runtime.stop()
    fun latest(): LiveDirectorState = runtime.latest()

    suspend fun guideUntilReady(
        planningFrame: Bitmap,
        spec: SierraDirectorSpec,
        thermalPressure: Double,
        isCurrent: () -> Boolean,
        onUpdate: (LiveDirectorState) -> Unit
    ): Result {
        if (camera.snapshot().extension != com.saunak.director.camera.DirectorExtension.NONE) {
            val state = runtime.stop()
            return Result(false, state, "Live Director analysis unavailable while OEM extension owns the session")
        }
        var state = runtime.initialize(spec, planningFrame)
        onUpdate(state)
        var analysisMisses = 0
        while (isCurrent() && !state.captureReady) {
            val frame = camera.captureAnalysisFrame(LIVE_FRAME_TIMEOUT_MS)
            if (frame == null) {
                analysisMisses++
                if (analysisMisses >= MAX_ANALYSIS_MISSES) {
                    val unavailable = state.copy(
                        entityLocked = false,
                        stable = false,
                        captureReady = false,
                        status = "Live Director preview unavailable"
                    )
                    onUpdate(unavailable)
                    return Result(false, unavailable, "Camera analysis stream unavailable")
                }
                delay(cadenceMs(thermalPressure))
                continue
            }
            try {
                analysisMisses = 0
                val quality = PhotoQuality.analyze(frame)
                val caps = camera.capabilities()
                val snapshot = camera.snapshot()
                val perceived = withTimeoutOrNull(LIVE_PERCEPTION_BUDGET_MS) {
                    perception.analyze(
                        frame = frame,
                        quality = quality,
                        lensFacing = caps.lensFacing,
                        zoomRatio = snapshot.zoomRatio,
                        thermalPressure = thermalPressure,
                        cameraId = camera.activeCameraId()
                    )
                }
                state = runtime.observe(frame, perceived, quality)
                onUpdate(state)
            } finally {
                if (!frame.isRecycled) frame.recycle()
            }
            if (!state.captureReady) delay(cadenceMs(thermalPressure))
        }
        return if (state.captureReady) Result(true, state) else Result(false, state, "Director session cancelled")
    }

    /**
     * READY expires naturally because the runtime stability window is timestamp based. One fresh
     * sample only preserves READY when the prior stable evidence is still within its temporal window.
     */
    suspend fun reverify(
        thermalPressure: Double,
        isCurrent: () -> Boolean,
        onUpdate: (LiveDirectorState) -> Unit
    ): Result {
        if (!isCurrent()) return Result(false, runtime.latest(), "Director session cancelled")
        val frame = camera.captureAnalysisFrame(REVERIFY_FRAME_TIMEOUT_MS)
            ?: return Result(false, runtime.latest().copy(captureReady = false, stable = false), "Fresh preview unavailable")
        return try {
            val quality = PhotoQuality.analyze(frame)
            val caps = camera.capabilities()
            val snapshot = camera.snapshot()
            val perceived = withTimeoutOrNull(REVERIFY_PERCEPTION_BUDGET_MS) {
                perception.analyze(
                    frame, quality, caps.lensFacing, snapshot.zoomRatio,
                    thermalPressure, camera.activeCameraId()
                )
            }
            val state = runtime.observe(frame, perceived, quality)
            onUpdate(state)
            Result(state.captureReady, state, if (state.captureReady) null else "Photographic target moved outside stable tolerance")
        } finally {
            if (!frame.isRecycled) frame.recycle()
        }
    }

    private fun cadenceMs(pressure: Double): Long = when {
        pressure >= 0.85 -> 320L
        pressure >= 0.65 -> 250L
        pressure >= 0.40 -> 190L
        else -> 140L
    }

    companion object {
        private const val LIVE_FRAME_TIMEOUT_MS = 420L
        private const val LIVE_PERCEPTION_BUDGET_MS = 340L
        private const val REVERIFY_FRAME_TIMEOUT_MS = 480L
        private const val REVERIFY_PERCEPTION_BUDGET_MS = 360L
        private const val MAX_ANALYSIS_MISSES = 6
    }
}
