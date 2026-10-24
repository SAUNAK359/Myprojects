package com.saunak.director.camera

import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ConcurrentLinkedQueue

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
}

fun CameraSensorTelemetry.matchesRequest(spec: ManualRawFrameSpec): Boolean {
    val exposureTolerance = maxOf(120_000L, (spec.exposureTimeNs * 0.06).toLong())
    val isoTolerance = maxOf(2, (spec.sensitivityIso * 0.10).toInt())
    return kotlin.math.abs(exposureTimeNs - spec.exposureTimeNs) <= exposureTolerance &&
        kotlin.math.abs(sensitivityIso - spec.sensitivityIso) <= isoTolerance
}
