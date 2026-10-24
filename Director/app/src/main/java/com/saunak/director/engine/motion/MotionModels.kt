package com.saunak.director.engine.motion

import com.saunak.director.engine.capture.MotionClass
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

data class MotionVector(val xPxPerSecond: Double = 0.0, val yPxPerSecond: Double = 0.0) {
    val speedPxPerSecond: Double get() = hypot(xPxPerSecond, yPxPerSecond)
}

data class MotionEstimate(
    val risk: Double,
    val uncertainty: Double,
    val confidence: Double,
    val vector: MotionVector = MotionVector(),
    val predictedBlurPx: Double = 0.0,
    val observedAtNs: Long,
    val source: String
) {
    init {
        require(risk in 0.0..1.0)
        require(uncertainty in 0.0..1.0)
        require(confidence in 0.0..1.0)
        require(predictedBlurPx >= 0.0)
        require(observedAtNs > 0)
        require(source.isNotBlank())
    }
    val conservativeRisk: Double get() = (risk + uncertainty).coerceIn(0.0, 1.0)
    fun toMotionClass(): MotionClass = when {
        conservativeRisk < 0.14 -> MotionClass.STATIC
        conservativeRisk < 0.38 -> MotionClass.LOW
        conservativeRisk < 0.68 -> MotionClass.MEDIUM
        else -> MotionClass.HIGH
    }
    companion object {
        fun unobserved(nowNs: Long, source: String = "motion_unobserved") = MotionEstimate(0.0, 1.0, 0.0, observedAtNs = nowNs, source = source)
    }
}

data class MotionSnapshot(
    val camera: MotionEstimate,
    val subject: MotionEstimate,
    val frameTimestampNs: Long,
    val fusionConfidence: Double,
    val timestampComparableToImu: Boolean
) {
    init { require(frameTimestampNs > 0); require(fusionConfidence in 0.0..1.0) }
    val conservativeRisk: Double get() = max(camera.conservativeRisk, subject.conservativeRisk)
    companion object {
        fun unobserved(nowNs: Long) = MotionSnapshot(
            MotionEstimate.unobserved(nowNs, "camera_motion_unobserved"),
            MotionEstimate.unobserved(nowNs, "subject_motion_unobserved"),
            nowNs, 0.0, false
        )
    }
}

data class LumaMotionFrame(
    val width: Int,
    val height: Int,
    val luma: ByteArray,
    val sensorTimestampNs: Long,
    val callbackElapsedRealtimeNs: Long,
    val rotationDegrees: Int
) {
    init {
        require(width > 0 && height > 0)
        require(luma.size == width * height)
        require(sensorTimestampNs > 0 && callbackElapsedRealtimeNs > 0)
        require(rotationDegrees in setOf(0, 90, 180, 270))
    }
}

data class VisualMotionObservation(
    val globalVector: MotionVector,
    val residualSubjectVector: MotionVector,
    val globalRisk: Double,
    val subjectRisk: Double,
    val confidence: Double,
    val frameTimestampNs: Long,
    val trackCount: Int
)

data class ImuMotionObservation(
    val angularVelocityRadPerSec: Double,
    val linearAccelerationStdMps2: Double,
    val confidence: Double,
    val observedAtElapsedRealtimeNs: Long,
    val timestampAligned: Boolean,
    val gyroSamples: Int,
    val accelerationSamples: Int
)

object MotionExposurePolicy {
    private const val MIN_EXPOSURE_NS = 4_000_000L
    private const val MAX_EXPOSURE_NS = 66_666_667L
    private const val TARGET_BLUR_PX = 1.5
    private const val RISK_EXPONENT = 2.8134107
    fun exposureCapNs(snapshot: MotionSnapshot): Long {
        val speed = max(snapshot.camera.vector.speedPxPerSecond, snapshot.subject.vector.speedPxPerSecond)
        val blurBound = if (speed > 1e-6) ((TARGET_BLUR_PX / speed) * 1_000_000_000.0).toLong() else MAX_EXPOSURE_NS
        val riskBound = (MAX_EXPOSURE_NS.toDouble() * exp(-RISK_EXPONENT * snapshot.conservativeRisk)).toLong()
        return min(blurBound.coerceIn(MIN_EXPOSURE_NS, MAX_EXPOSURE_NS), riskBound.coerceIn(MIN_EXPOSURE_NS, MAX_EXPOSURE_NS))
    }
}
