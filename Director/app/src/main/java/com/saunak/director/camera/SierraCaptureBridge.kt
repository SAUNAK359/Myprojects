package com.saunak.director.camera

import com.saunak.director.director.SierraSceneSignals
import com.saunak.director.engine.SierraEngine
import com.saunak.director.engine.capture.AutoExposureSnapshot
import com.saunak.director.engine.capture.CaptureBudget
import com.saunak.director.engine.capture.CaptureCompiler
import com.saunak.director.engine.capture.CaptureObservation
import com.saunak.director.engine.capture.CaptureProgram
import com.saunak.director.engine.capture.MotionClass
import com.saunak.director.platform.ThermalPolicy
import com.saunak.director.processing.TechnicalQuality
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.log2
import kotlin.math.max

/**
 * Converts only verified camera/quality state into a physics-constrained Sierra capture program.
 * Motion intentionally defaults to MEDIUM until gyro/flow experts provide trustworthy evidence.
 */
class SierraCaptureBridge(
    private val camera: CameraEngine,
    private val engine: SierraEngine = SierraEngine.get(),
    private val compiler: CaptureCompiler = CaptureCompiler()
) {
    suspend fun plan(
        quality: TechnicalQuality,
        scene: SierraSceneSignals,
        thermal: ThermalPolicy
    ): Result<CaptureProgram> = withContext(Dispatchers.Default) {
        runCatching {
            val telemetry = requireNotNull(camera.sensorTelemetry()) {
                "No synchronized Camera2 sensor telemetry is available yet"
            }
            val twin = engine.deviceTwin().getOrThrow()
            val cameraId = telemetry.cameraId ?: camera.activeCameraId()
            val module = requireNotNull(cameraId?.let(twin::camera)) {
                "Active camera is missing from Sierra device twin"
            }
            val observation = estimateObservation(quality, scene, camera.snapshot().zoomRatio)
            val budget = CaptureBudget(
                thermalPressure = thermal.thermalPressure,
                memoryPressure = 0.0,
                maxBurstFrames = thermal.maxSierraBurstFrames
            )
            compiler.compile(
                camera = module,
                autoExposure = AutoExposureSnapshot(telemetry.exposureTimeNs, telemetry.sensitivityIso),
                observation = observation,
                budget = budget,
                oemExtensionAvailable = camera.capabilities().extensions.isNotEmpty()
            )
        }
    }

    internal fun estimateObservation(
        quality: TechnicalQuality,
        scene: SierraSceneSignals,
        zoomRatio: Float
    ): CaptureObservation {
        val p05 = quality.shadowP05.coerceIn(0.0, 255.0)
        val p95 = quality.highlightP95.coerceIn(0.0, 255.0)
        // This is a scene-luminance proxy, not claimed sensor dynamic range. It is used only as a
        // relative capture-policy input and is clamped to a physically plausible decision range.
        val rangeProxyStops = log2((p95 + 1.0) / (p05 + 1.0)).coerceIn(0.0, 14.0)
        val highlightTailRisk = ((p95 - 230.0) / 25.0).coerceIn(0.0, 1.0)
        val highlightRisk = max(quality.brightClipFraction.coerceIn(0.0, 1.0), highlightTailRisk)
        val shadowTailRisk = ((34.0 - p05) / 34.0).coerceIn(0.0, 1.0)
        val shadowRisk = max(quality.darkClipFraction.coerceIn(0.0, 1.0), shadowTailRisk)

        val subjectMotion = if (scene.primaryHumanSubject) MotionClass.MEDIUM else MotionClass.MEDIUM
        val cameraMotion = MotionClass.MEDIUM
        return CaptureObservation(
            dynamicRangeStops = rangeProxyStops,
            highlightClipRisk = highlightRisk,
            shadowNoiseRisk = shadowRisk,
            cameraMotion = cameraMotion,
            subjectMotion = subjectMotion,
            facePresent = scene.faceCount > 0,
            digitalZoomRatio = zoomRatio.toDouble().coerceAtLeast(1.0),
            confidence = scene.confidence.coerceIn(0.0, 1.0)
        )
    }
}
