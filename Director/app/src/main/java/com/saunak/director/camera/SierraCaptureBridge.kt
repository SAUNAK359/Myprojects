package com.saunak.director.camera

import com.saunak.director.director.SierraSceneSignals
import com.saunak.director.engine.SierraEngine
import com.saunak.director.engine.capture.AutoExposureSnapshot
import com.saunak.director.engine.capture.CaptureBudget
import com.saunak.director.engine.capture.CaptureCompiler
import com.saunak.director.engine.capture.CaptureObservation
import com.saunak.director.engine.capture.CaptureProgram
import com.saunak.director.engine.cognition.CognitionDecision
import com.saunak.director.engine.motion.MotionExposurePolicy
import com.saunak.director.engine.state.SceneState
import com.saunak.director.engine.state.SceneStateAssembler
import com.saunak.director.platform.MemoryPressureBus
import com.saunak.director.platform.ThermalPolicy
import com.saunak.director.processing.TechnicalQuality
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.log2
import kotlin.math.max

class SierraCaptureBridge(
    private val camera: CameraEngine,
    private val engine: SierraEngine = SierraEngine.get(),
    private val compiler: CaptureCompiler = CaptureCompiler()
) {
    /** Always available, including when an OEM CameraX extension is active. */
    suspend fun evaluateCognition(quality: TechnicalQuality, scene: SierraSceneSignals): Result<CognitionDecision> =
        withContext(Dispatchers.Default) {
            runCatching { engine.cognition.evaluate(assembleState(quality, scene)) }
        }

    suspend fun plan(
        quality: TechnicalQuality,
        scene: SierraSceneSignals,
        thermal: ThermalPolicy
    ): Result<CaptureProgram> = withContext(Dispatchers.Default) {
        runCatching {
            val state = assembleState(quality, scene)
            val cognition = engine.cognition.evaluate(state)
            val telemetry = state.telemetry
            val twin = engine.deviceTwin().getOrThrow()
            val cameraId = telemetry.cameraId ?: camera.activeCameraId()
            val module = requireNotNull(cameraId?.let(twin::camera)) { "Active camera is missing from Sierra device twin" }
            val observation = estimateObservation(state).let { base ->
                val intent = cognition.intent
                base.copy(
                    highlightClipRisk = max(base.highlightClipRisk, intent.capture.highlightProtectionPriority * base.highlightClipRisk),
                    shadowNoiseRisk = max(base.shadowNoiseRisk, intent.capture.shadowRecoveryPriority * base.shadowNoiseRisk),
                    confidence = minOf(base.confidence, cognition.reliability.effectiveConfidence),
                    motionRisk = max(base.motionRisk ?: 0.0, intent.capture.motionFreezePriority * (base.motionRisk ?: state.motion.conservativeRisk)).coerceIn(0.0, 1.0)
                )
            }
            val budget = CaptureBudget(
                thermalPressure = thermal.thermalPressure,
                memoryPressure = MemoryPressureBus.normalizedPressure(),
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

    private suspend fun assembleState(quality: TechnicalQuality, scene: SierraSceneSignals): SceneState {
        val motion = engine.motionProbe.measure(camera)
        val telemetry = requireNotNull(camera.sensorTelemetry()) { "No synchronized Camera2 sensor telemetry is available yet" }
        return SceneStateAssembler.assemble(
            scene = scene,
            quality = quality,
            motion = motion,
            telemetry = telemetry,
            zoomRatio = camera.snapshot().zoomRatio,
            subject = engine.subjects.latest()
        )
    }

    internal fun estimateObservation(state: SceneState): CaptureObservation {
        val quality = state.technicalQuality
        val scene = state.sceneSignals
        val p05 = quality.shadowP05.coerceIn(0.0, 255.0)
        val p95 = quality.highlightP95.coerceIn(0.0, 255.0)
        val rangeProxyStops = log2((p95 + 1.0) / (p05 + 1.0)).coerceIn(0.0, 14.0)
        val highlightTailRisk = ((p95 - 230.0) / 25.0).coerceIn(0.0, 1.0)
        val highlightRisk = max(quality.brightClipFraction.coerceIn(0.0, 1.0), highlightTailRisk)
        val shadowTailRisk = ((34.0 - p05) / 34.0).coerceIn(0.0, 1.0)
        val shadowRisk = max(quality.darkClipFraction.coerceIn(0.0, 1.0), shadowTailRisk)
        val effectiveConfidence = (state.confidence * (1.0 - 0.35 * state.uncertainty)).coerceIn(0.0, 1.0)
        val baseExposureCap = MotionExposurePolicy.exposureCapNs(state.motion)
        val predictive = state.predictiveSubject
        val predictiveCap = predictive?.let { (baseExposureCap.toDouble() * it.exposureCapMultiplier).toLong().coerceAtLeast(MIN_PREDICTIVE_EXPOSURE_NS) }
        val exposureCap = minOf(baseExposureCap, predictiveCap ?: baseExposureCap)
        val combinedMotionRisk = max(state.motion.conservativeRisk, predictive?.kinematicRisk?.times(predictive.confidence) ?: 0.0).coerceIn(0.0, 1.0)

        return CaptureObservation(
            dynamicRangeStops = rangeProxyStops,
            highlightClipRisk = highlightRisk,
            shadowNoiseRisk = shadowRisk,
            cameraMotion = state.motion.camera.toMotionClass(),
            subjectMotion = state.motion.subject.toMotionClass(),
            facePresent = scene.faceCount > 0,
            digitalZoomRatio = state.zoomRatio.toDouble().coerceAtLeast(1.0),
            confidence = effectiveConfidence,
            motionExposureCapNs = exposureCap,
            motionRisk = combinedMotionRisk,
            subjectKinematicRisk = predictive?.kinematicRisk,
            subjectEdgeRisk = predictive?.edgeRisk,
            subjectTrackConfidence = predictive?.confidence
        )
    }

    companion object { private const val MIN_PREDICTIVE_EXPOSURE_NS = 2_000_000L }
}
