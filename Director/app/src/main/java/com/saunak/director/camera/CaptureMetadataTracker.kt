package com.saunak.director.camera

import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.util.Pair
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * Per-frame sensor/ISP calibration emitted by Camera2 for the exact sensor timestamp.
 * Arrays are immutable copies owned by Sierra and ordered in the camera's documented CFA order.
 */
data class FrameSensorCalibration(
    val sensorTimestampNs: Long,
    val dynamicBlackLevel: FloatArray?,
    val dynamicWhiteLevel: Int?,
    val noiseProfile: DoubleArray?,
    val neutralColorPoint: DoubleArray?,
    /** [R, G_even, G_odd, B] gains. */
    val awbGains: FloatArray?,
    /** Row-major 3x3 sensor RGB -> linear sRGB transform when reported by the HAL. */
    val sensorToLinearSrgb: DoubleArray?,
    val greenSplit: Float?
) {
    init {
        require(sensorTimestampNs > 0)
        require(dynamicBlackLevel == null || dynamicBlackLevel.size == 4)
        require(dynamicWhiteLevel == null || dynamicWhiteLevel > 0)
        require(noiseProfile == null || noiseProfile.size == 8)
        require(neutralColorPoint == null || neutralColorPoint.size == 3)
        require(awbGains == null || awbGains.size == 4)
        require(sensorToLinearSrgb == null || sensorToLinearSrgb.size == 9)
    }

    /** Returns [slope, intercept] for CFA phase 0..3 where variance = slope*signal + intercept. */
    fun noiseForPhase(phase: Int): Pair<Double, Double>? {
        val values = noiseProfile ?: return null
        val p = phase.coerceIn(0, 3) * 2
        return Pair(values[p], values[p + 1])
    }
}

/** Bounded global calibration ledger that survives the normal-session rebind after RAW capture. */
object FrameCalibrationLedger {
    private const val CAPACITY = 128
    private val values = ConcurrentHashMap<Long, FrameSensorCalibration>(CAPACITY)
    private val insertionOrder = ConcurrentLinkedQueue<Long>()

    fun put(value: FrameSensorCalibration) {
        val previous = values.put(value.sensorTimestampNs, value)
        if (previous == null) insertionOrder.add(value.sensorTimestampNs)
        while (values.size > CAPACITY) {
            val oldest = insertionOrder.poll() ?: break
            values.remove(oldest)
        }
    }

    fun find(sensorTimestampNs: Long): FrameSensorCalibration? = values[sensorTimestampNs]
}

/**
 * Bounded timestamp-indexed Camera2 metadata ledger.
 *
 * SENSOR_TIMESTAMP is the bridge between Camera2 CaptureResult and CameraX ImageInfo.timestamp for
 * buffers emitted from the same sensor capture. Keeping the ledger bounded avoids retaining an
 * unbounded stream of preview results while allowing RAW callbacks to resolve exact per-frame data.
 */
class CaptureMetadataTracker(
    private val capacity: Int = 96
) {
    init { require(capacity in 16..512) }

    private val byTimestamp = ConcurrentHashMap<Long, CameraSensorTelemetry>(capacity)
    private val insertionOrder = ConcurrentLinkedQueue<Long>()

    fun record(cameraId: String?, result: TotalCaptureResult): CameraSensorTelemetry? {
        val timestamp = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return null
        val exposure = result.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: return null
        val iso = result.get(CaptureResult.SENSOR_SENSITIVITY) ?: return null
        if (timestamp <= 0L || exposure <= 0L || iso <= 0) return null
        val frameDuration = (result.get(CaptureResult.SENSOR_FRAME_DURATION) ?: exposure).coerceAtLeast(exposure)
        val telemetry = CameraSensorTelemetry(
            cameraId = cameraId,
            sensorTimestampNs = timestamp,
            exposureTimeNs = exposure,
            sensitivityIso = iso,
            frameDurationNs = frameDuration,
            rollingShutterSkewNs = result.get(CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW),
            focalLengthMm = result.get(CaptureResult.LENS_FOCAL_LENGTH),
            aperture = result.get(CaptureResult.LENS_APERTURE),
            focusDistanceDiopters = result.get(CaptureResult.LENS_FOCUS_DISTANCE),
            afState = result.get(CaptureResult.CONTROL_AF_STATE),
            aeState = result.get(CaptureResult.CONTROL_AE_STATE),
            awbState = result.get(CaptureResult.CONTROL_AWB_STATE)
        )
        val previous = byTimestamp.put(timestamp, telemetry)
        if (previous == null) insertionOrder.add(timestamp)
        trim()
        extractCalibration(timestamp, result)?.let(FrameCalibrationLedger::put)
        return telemetry
    }

    fun find(sensorTimestampNs: Long): CameraSensorTelemetry? = byTimestamp[sensorTimestampNs]

    fun clear() {
        byTimestamp.clear()
        insertionOrder.clear()
    }

    private fun trim() {
        while (byTimestamp.size > capacity) {
            val oldest = insertionOrder.poll() ?: break
            byTimestamp.remove(oldest)
        }
    }

    private fun extractCalibration(timestamp: Long, result: TotalCaptureResult): FrameSensorCalibration? {
        val black = result.get(CaptureResult.SENSOR_DYNAMIC_BLACK_LEVEL)?.copyOf()
        val white = result.get(CaptureResult.SENSOR_DYNAMIC_WHITE_LEVEL)
        val noise = result.get(CaptureResult.SENSOR_NOISE_PROFILE)?.let { profile ->
            if (profile.size == 4) DoubleArray(8) { index ->
                val pair = profile[index / 2]
                if (index % 2 == 0) pair.first else pair.second
            } else null
        }
        val neutral = result.get(CaptureResult.SENSOR_NEUTRAL_COLOR_POINT)?.let { points ->
            if (points.size == 3) DoubleArray(3) { points[it].toDouble() } else null
        }
        val gains = result.get(CaptureResult.COLOR_CORRECTION_GAINS)?.let { value ->
            floatArrayOf(value.red, value.greenEven, value.greenOdd, value.blue)
        }
        val transform = result.get(CaptureResult.COLOR_CORRECTION_TRANSFORM)?.let { matrix ->
            DoubleArray(9) { index ->
                val row = index / 3
                val column = index % 3
                matrix.getElement(column, row).toDouble()
            }
        }
        val greenSplit = result.get(CaptureResult.SENSOR_GREEN_SPLIT)
        if (black == null && white == null && noise == null && neutral == null && gains == null && transform == null && greenSplit == null) {
            return null
        }
        return FrameSensorCalibration(
            sensorTimestampNs = timestamp,
            dynamicBlackLevel = black,
            dynamicWhiteLevel = white,
            noiseProfile = noise,
            neutralColorPoint = neutral,
            awbGains = gains,
            sensorToLinearSrgb = transform,
            greenSplit = greenSplit
        )
    }
}

fun CameraSensorTelemetry.matchesRequest(spec: ManualRawFrameSpec): Boolean {
    val exposureTolerance = maxOf(120_000L, (spec.exposureTimeNs * 0.06).toLong())
    val isoTolerance = maxOf(2, (spec.sensitivityIso * 0.10).toInt())
    return kotlin.math.abs(exposureTimeNs - spec.exposureTimeNs) <= exposureTolerance &&
        kotlin.math.abs(sensitivityIso - spec.sensitivityIso) <= isoTolerance
}
