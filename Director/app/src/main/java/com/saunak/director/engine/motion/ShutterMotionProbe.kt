package com.saunak.director.engine.motion

import android.content.Context
import android.graphics.Bitmap
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.SystemClock
import com.saunak.director.camera.CameraEngine

/**
 * Refreshes temporal motion immediately before capture planning. The camera's existing KEEP_ONLY_LATEST
 * ImageAnalysis path is sampled for a short rolling window, keeping the green v4 capture path intact.
 */
class ShutterMotionProbe(
    context: Context,
    private val fusion: MotionFusionEngine
) {
    private val cameraManager = context.applicationContext.getSystemService(CameraManager::class.java)

    suspend fun measure(camera: CameraEngine, samples: Int = DEFAULT_SAMPLES): MotionSnapshot {
        val count = samples.coerceIn(2, MAX_SAMPLES)
        val comparable = timestampComparable(camera)
        repeat(count) {
            val bitmap = camera.captureAnalysisFrame(FRAME_TIMEOUT_MS) ?: return@repeat
            try {
                val telemetry = camera.sensorTelemetry()
                val callbackNs = SystemClock.elapsedRealtimeNanos()
                // If the HAL does not declare REALTIME timestamps, do not compare its sensor clock
                // against IMU time. Use callback elapsed realtime for temporal visual tracking instead.
                val frameTimeNs = if (comparable) telemetry?.sensorTimestampNs ?: callbackNs else callbackNs
                val luma = bitmap.toMotionLuma(frameTimeNs, callbackNs)
                fusion.onFrame(
                    frame = luma,
                    timestampComparableToImu = comparable && telemetry != null,
                    exposureTimeNs = telemetry?.exposureTimeNs
                )
            } finally {
                if (!bitmap.isRecycled) bitmap.recycle()
            }
        }
        // No categorical fallback: absent evidence raises uncertainty toward the upper risk bound.
        return fusion.latest(SystemClock.elapsedRealtimeNanos())
    }

    private fun timestampComparable(camera: CameraEngine): Boolean {
        val id = camera.activeCameraId() ?: return false
        return runCatching {
            cameraManager?.getCameraCharacteristics(id)
                ?.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) ==
                CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
        }.getOrDefault(false)
    }

    private fun Bitmap.toMotionLuma(sensorTimestampNs: Long, callbackNs: Long): LumaMotionFrame {
        val targetWidth = MOTION_WIDTH
        val targetHeight = ((height.toDouble() / width.toDouble()) * targetWidth)
            .toInt().coerceIn(MIN_MOTION_HEIGHT, MAX_MOTION_HEIGHT)
        val scaled = if (width == targetWidth && height == targetHeight) this
        else Bitmap.createScaledBitmap(this, targetWidth, targetHeight, false)
        try {
            val pixels = IntArray(targetWidth * targetHeight)
            scaled.getPixels(pixels, 0, targetWidth, 0, 0, targetWidth, targetHeight)
            val luma = ByteArray(pixels.size)
            for (i in pixels.indices) {
                val p = pixels[i]
                val r = (p shr 16) and 0xff
                val g = (p shr 8) and 0xff
                val b = p and 0xff
                luma[i] = ((77 * r + 150 * g + 29 * b) shr 8).toByte()
            }
            return LumaMotionFrame(
                width = targetWidth,
                height = targetHeight,
                luma = luma,
                sensorTimestampNs = sensorTimestampNs,
                callbackElapsedRealtimeNs = callbackNs,
                rotationDegrees = 0
            )
        } finally {
            if (scaled !== this && !scaled.isRecycled) scaled.recycle()
        }
    }

    companion object {
        private const val DEFAULT_SAMPLES = 3
        private const val MAX_SAMPLES = 5
        private const val FRAME_TIMEOUT_MS = 180L
        private const val MOTION_WIDTH = 112
        private const val MIN_MOTION_HEIGHT = 56
        private const val MAX_MOTION_HEIGHT = 112
    }
}
