package com.saunak.director.engine.geometry

import kotlin.math.abs
import kotlin.math.max

data class GyroSample(
    val timestampNs: Long,
    val wx: Float,
    val wy: Float,
    val wz: Float
) {
    init { require(timestampNs > 0); require(wx.isFinite() && wy.isFinite() && wz.isFinite()) }
}

data class ImuSyncModel(
    val sensorToCameraOffsetNs: Long,
    val jitterNs: Long,
    val confidence: Float
) {
    init {
        require(jitterNs >= 0)
        require(confidence in 0f..1f)
    }
}

data class FrameReadoutModel(
    val exposureStartNs: Long,
    val exposureDurationNs: Long,
    val rollingShutterSkewNs: Long,
    val imageHeight: Int
) {
    init {
        require(exposureStartNs > 0)
        require(exposureDurationNs > 0)
        require(rollingShutterSkewNs >= 0)
        require(imageHeight > 0)
    }

    fun rowTimeNs(y: Int): Long {
        val row = y.coerceIn(0, imageHeight - 1)
        val fraction = if (imageHeight <= 1) 0.0 else row.toDouble() / (imageHeight - 1).toDouble()
        return exposureStartNs + (rollingShutterSkewNs * fraction).toLong() + exposureDurationNs / 2L
    }
}

data class RotationVector(
    val x: Double,
    val y: Double,
    val z: Double
) {
    val magnitude: Double get() = kotlin.math.sqrt(x * x + y * y + z * z)
}

data class RowMotionEstimate(
    val row: Int,
    val rotation: RotationVector,
    val confidence: Float
)

/**
 * Portable gyro integration baseline used before dense optical flow is available. It models only
 * camera rotation; image residual flow/depth later supplies translation and scene motion.
 */
object TemporalGeometry {
    fun integrateGyro(
        samples: List<GyroSample>,
        startNs: Long,
        endNs: Long,
        sync: ImuSyncModel
    ): Pair<RotationVector, Float> {
        require(endNs >= startNs)
        if (samples.size < 2 || sync.confidence <= 0f) return RotationVector(0.0, 0.0, 0.0) to 0f

        val cameraStart = startNs + sync.sensorToCameraOffsetNs
        val cameraEnd = endNs + sync.sensorToCameraOffsetNs
        val relevant = samples.filter { it.timestampNs in cameraStart..cameraEnd }
        if (relevant.size < 2) return RotationVector(0.0, 0.0, 0.0) to 0f

        var rx = 0.0
        var ry = 0.0
        var rz = 0.0
        for (i in 1 until relevant.size) {
            val a = relevant[i - 1]
            val b = relevant[i]
            val dt = (b.timestampNs - a.timestampNs).toDouble() * 1e-9
            if (dt <= 0.0 || dt > 0.05) continue
            rx += 0.5 * (a.wx + b.wx) * dt
            ry += 0.5 * (a.wy + b.wy) * dt
            rz += 0.5 * (a.wz + b.wz) * dt
        }

        val span = max(1L, cameraEnd - cameraStart)
        val coverage = ((relevant.last().timestampNs - relevant.first().timestampNs).toDouble() / span.toDouble()).coerceIn(0.0, 1.0)
        val jitterPenalty = (1.0 - (sync.jitterNs.toDouble() / 5_000_000.0)).coerceIn(0.0, 1.0)
        val confidence = (sync.confidence * coverage * jitterPenalty).toFloat().coerceIn(0f, 1f)
        return RotationVector(rx, ry, rz) to confidence
    }

    fun rowMotion(
        samples: List<GyroSample>,
        readout: FrameReadoutModel,
        sync: ImuSyncModel,
        referenceTimeNs: Long = readout.rowTimeNs(readout.imageHeight / 2)
    ): List<RowMotionEstimate> {
        val result = ArrayList<RowMotionEstimate>(readout.imageHeight)
        for (y in 0 until readout.imageHeight) {
            val rowTime = readout.rowTimeNs(y)
            val (rotation, confidence) = if (rowTime >= referenceTimeNs) {
                integrateGyro(samples, referenceTimeNs, rowTime, sync)
            } else {
                val (forward, c) = integrateGyro(samples, rowTime, referenceTimeNs, sync)
                RotationVector(-forward.x, -forward.y, -forward.z) to c
            }
            result += RowMotionEstimate(y, rotation, confidence)
        }
        return result
    }

    fun motionClass(rotation: RotationVector): CameraMotionClass = when {
        rotation.magnitude < 0.0015 -> CameraMotionClass.STATIC
        rotation.magnitude < 0.006 -> CameraMotionClass.LOW
        rotation.magnitude < 0.02 -> CameraMotionClass.MEDIUM
        else -> CameraMotionClass.HIGH
    }
}

enum class CameraMotionClass { STATIC, LOW, MEDIUM, HIGH }
