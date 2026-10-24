package com.saunak.director.engine.capture

import com.saunak.director.engine.device.CameraModuleTwin
import com.saunak.director.engine.device.CapturePath
import kotlin.math.pow
import kotlin.math.roundToInt

enum class CaptureFrameRole { REFERENCE, HIGHLIGHT, SHADOW, DETAIL }
enum class MotionClass { STATIC, LOW, MEDIUM, HIGH }

data class AutoExposureSnapshot(val exposureTimeNs: Long, val sensitivityIso: Int) {
    init { require(exposureTimeNs > 0 && sensitivityIso > 0) }
}

data class CaptureObservation(
    val dynamicRangeStops: Double,
    val highlightClipRisk: Double,
    val shadowNoiseRisk: Double,
    val cameraMotion: MotionClass,
    val subjectMotion: MotionClass,
    val facePresent: Boolean,
    val digitalZoomRatio: Double = 1.0,
    val confidence: Double = 1.0
) {
    init {
        require(dynamicRangeStops >= 0)
        require(highlightClipRisk in 0.0..1.0)
        require(shadowNoiseRisk in 0.0..1.0)
        require(digitalZoomRatio >= 1.0)
        require(confidence in 0.0..1.0)
    }
}

data class CaptureBudget(
    val thermalPressure: Double = 0.0,
    val memoryPressure: Double = 0.0,
    /** A value below three explicitly disables computational burst candidates. */
    val maxBurstFrames: Int = 8
) {
    init {
        require(thermalPressure in 0.0..1.0)
        require(memoryPressure in 0.0..1.0)
        require(maxBurstFrames in 1..16)
    }
}

data class CaptureFrameSpec(
    val exposureTimeNs: Long,
    val sensitivityIso: Int,
    val role: CaptureFrameRole,
    val evOffset: Double
)

data class CaptureProgram(
    val path: CapturePath,
    val frames: List<CaptureFrameSpec>,
    val lockThreeA: Boolean,
    val hdrFusion: Boolean,
    val burstSuperResolution: Boolean,
    val expectedCaptureDurationMs: Double,
    val expectedUtility: Double,
    val rationale: String
) {
    init {
        require(frames.isNotEmpty())
        require(expectedCaptureDurationMs > 0)
        require(expectedUtility in 0.0..1.0)
    }
}

/** Physics-constrained capture planner. Neural systems may propose intent; this class owns sensor safety. */
class CaptureCompiler {
    fun compile(
        camera: CameraModuleTwin,
        autoExposure: AutoExposureSnapshot,
        observation: CaptureObservation,
        budget: CaptureBudget,
        oemExtensionAvailable: Boolean = false
    ): CaptureProgram {
        val candidates = mutableListOf<CaptureProgram>()
        candidates += safeSingle(camera, autoExposure, observation, oemExtensionAvailable)

        // Severe thermal/memory policies may explicitly allow only one or two physical captures.
        // In that state a "burst" would add complexity without enough samples to justify custom fusion.
        if (budget.maxBurstFrames >= MIN_COMPUTATIONAL_BURST) {
            if (camera.manualSensorSupported && camera.rawSupported && CapturePath.RAW_CUSTOM in camera.preferredCapturePaths) {
                candidates += constantRawBurst(camera, autoExposure, observation, budget)
                if (observation.dynamicRangeStops >= 10.0 || observation.highlightClipRisk >= 0.35 || observation.shadowNoiseRisk >= 0.45) {
                    candidates += adaptiveRawBurst(camera, autoExposure, observation, budget)
                }
            } else if (CapturePath.YUV_REPROCESS in camera.preferredCapturePaths) {
                candidates += yuvBurst(camera, autoExposure, observation, budget)
            }
        }

        return candidates.maxWithOrNull(
            compareBy<CaptureProgram> { it.expectedUtility }.thenByDescending { -it.expectedCaptureDurationMs }
        ) ?: safeSingle(camera, autoExposure, observation, oemExtensionAvailable)
    }

    private fun safeSingle(camera: CameraModuleTwin, base: AutoExposureSnapshot, observation: CaptureObservation, oemExtensionAvailable: Boolean): CaptureProgram {
        val path = if (oemExtensionAvailable && (observation.highlightClipRisk > 0.45 || observation.shadowNoiseRisk > 0.55)) CapturePath.OEM_EXTENSION else CapturePath.CAMERA2_SAFE
        val frame = constrainedFrame(camera, base, 0.0, CaptureFrameRole.REFERENCE, maxExposureNs(observation))
        val utility = (0.58 + 0.15 * observation.confidence - 0.12 * observation.shadowNoiseRisk - 0.08 * observation.highlightClipRisk).coerceIn(0.05, 0.86)
        return CaptureProgram(path, listOf(frame), false, false, false,
            frame.exposureTimeNs / 1_000_000.0 + 35.0, utility,
            if (path == CapturePath.OEM_EXTENSION) "OEM computational path is safer than an unsupported custom burst" else "single-frame safety baseline")
    }

    private fun constantRawBurst(camera: CameraModuleTwin, base: AutoExposureSnapshot, observation: CaptureObservation, budget: CaptureBudget): CaptureProgram {
        val highMotion = maxMotion(observation) == MotionClass.HIGH
        val countTarget = when {
            highMotion -> 4
            observation.shadowNoiseRisk > 0.65 -> 8
            observation.digitalZoomRatio >= 1.8 -> 7
            else -> 6
        }
        val count = qualityPreservingFrameCount(countTarget, budget)
        val frame = constrainedFrame(camera, base, if (observation.highlightClipRisk > 0.6) -0.7 else 0.0,
            CaptureFrameRole.REFERENCE, maxExposureNs(observation))
        val frames = List(count) { index -> frame.copy(role = if (index == 0) CaptureFrameRole.REFERENCE else CaptureFrameRole.DETAIL) }
        val utility = (0.72 + 0.16 * observation.shadowNoiseRisk + 0.07 * (observation.digitalZoomRatio - 1.0).coerceAtMost(1.0) - if (highMotion) 0.08 else 0.0).coerceIn(0.0, 0.96)
        return CaptureProgram(CapturePath.RAW_CUSTOM, frames, true, observation.dynamicRangeStops >= 9.0,
            observation.digitalZoomRatio >= 1.45 && maxMotion(observation) != MotionClass.HIGH,
            estimateDuration(frames), utility, "constant-exposure RAW burst preserves highlight alignment under motion")
    }

    private fun adaptiveRawBurst(camera: CameraModuleTwin, base: AutoExposureSnapshot, observation: CaptureObservation, budget: CaptureBudget): CaptureProgram {
        val motion = maxMotion(observation)
        val offsets = when (motion) {
            MotionClass.HIGH -> listOf(-1.3, -0.7, 0.0, 0.0)
            MotionClass.MEDIUM -> listOf(-1.7, -0.8, 0.0, 0.0, 0.35)
            MotionClass.LOW -> listOf(-2.0, -1.0, 0.0, 0.0, 0.6, 0.6)
            MotionClass.STATIC -> listOf(-2.0, -1.0, 0.0, 0.0, 0.7, 1.0, 1.0)
        }
        val selectedOffsets = offsets.take(qualityPreservingFrameCount(offsets.size, budget))
        val frames = selectedOffsets.mapIndexed { index, ev ->
            val role = when {
                ev <= -0.75 -> CaptureFrameRole.HIGHLIGHT
                ev >= 0.5 -> CaptureFrameRole.SHADOW
                index == 0 || ev == 0.0 -> CaptureFrameRole.REFERENCE
                else -> CaptureFrameRole.DETAIL
            }
            constrainedFrame(camera, base, ev, role, maxExposureNs(observation))
        }
        val ghostPenalty = when (motion) {
            MotionClass.HIGH -> 0.22
            MotionClass.MEDIUM -> 0.10
            MotionClass.LOW -> 0.035
            MotionClass.STATIC -> 0.0
        }
        val utility = (0.76 + observation.highlightClipRisk * 0.11 + observation.shadowNoiseRisk * 0.12 - ghostPenalty).coerceIn(0.0, 0.98)
        return CaptureProgram(CapturePath.RAW_CUSTOM, frames, true, true,
            observation.digitalZoomRatio >= 1.55 && motion <= MotionClass.LOW,
            estimateDuration(frames), utility, "motion-bounded mixed-exposure RAW burst for high dynamic range")
    }

    private fun yuvBurst(camera: CameraModuleTwin, base: AutoExposureSnapshot, observation: CaptureObservation, budget: CaptureBudget): CaptureProgram {
        val count = qualityPreservingFrameCount(if (maxMotion(observation) >= MotionClass.MEDIUM) 3 else 5, budget)
        val frame = constrainedFrame(camera, base, 0.0, CaptureFrameRole.REFERENCE, maxExposureNs(observation))
        val frames = List(count) { i -> frame.copy(role = if (i == 0) CaptureFrameRole.REFERENCE else CaptureFrameRole.DETAIL) }
        return CaptureProgram(CapturePath.YUV_REPROCESS, frames, true, observation.dynamicRangeStops >= 10.0, false,
            estimateDuration(frames), (0.69 + 0.1 * observation.shadowNoiseRisk).coerceIn(0.0, 0.87),
            "reprocessable multi-frame path when RAW/manual capture is unavailable")
    }

    private fun constrainedFrame(camera: CameraModuleTwin, base: AutoExposureSnapshot, evOffset: Double, role: CaptureFrameRole, motionExposureCapNs: Long): CaptureFrameSpec {
        val requestedExposure = (base.exposureTimeNs.toDouble() * 2.0.pow(evOffset)).toLong().coerceAtLeast(1L)
        val minExposure = camera.exposureTimeMinNs ?: 1L
        val maxExposure = minOf(camera.exposureTimeMaxNs ?: requestedExposure, motionExposureCapNs).coerceAtLeast(minExposure)
        val exposure = requestedExposure.coerceIn(minExposure, maxExposure)
        val desiredEnergy = base.exposureTimeNs.toDouble() * base.sensitivityIso.toDouble() * 2.0.pow(evOffset)
        val requestedIso = (desiredEnergy / exposure.toDouble()).roundToInt().coerceAtLeast(1)
        val minIso = camera.sensitivityIsoMin ?: 1
        val maxIso = camera.sensitivityIsoMax ?: maxOf(requestedIso, minIso)
        return CaptureFrameSpec(exposure, requestedIso.coerceIn(minIso, maxIso), role, evOffset)
    }

    private fun qualityPreservingFrameCount(target: Int, budget: CaptureBudget): Int {
        require(budget.maxBurstFrames >= MIN_COMPUTATIONAL_BURST)
        val pressure = maxOf(budget.memoryPressure, budget.thermalPressure)
        val pressureCap = when {
            pressure >= 0.90 -> maxOf(MIN_COMPUTATIONAL_BURST, target - 2)
            pressure >= 0.75 -> maxOf(MIN_COMPUTATIONAL_BURST, target - 1)
            else -> target
        }
        return minOf(target, budget.maxBurstFrames, pressureCap).coerceAtLeast(MIN_COMPUTATIONAL_BURST)
    }

    private fun maxExposureNs(observation: CaptureObservation): Long = when (maxMotion(observation)) {
        MotionClass.HIGH -> 8_000_000L
        MotionClass.MEDIUM -> 12_500_000L
        MotionClass.LOW -> 25_000_000L
        MotionClass.STATIC -> 66_666_667L
    }

    private fun maxMotion(observation: CaptureObservation): MotionClass = maxOf(observation.cameraMotion, observation.subjectMotion)
    private fun estimateDuration(frames: List<CaptureFrameSpec>): Double = frames.sumOf { maxOf(it.exposureTimeNs / 1_000_000.0, 8.0) } + 18.0 * frames.size

    companion object { private const val MIN_COMPUTATIONAL_BURST = 3 }
}
