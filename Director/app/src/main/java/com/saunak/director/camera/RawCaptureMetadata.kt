package com.saunak.director.camera

import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.ColorSpaceTransform
import android.hardware.camera2.params.RggbChannelVector
import android.util.Pair
import android.util.Rational
import com.saunak.director.engine.reconstruction.Matrix3x3
import com.saunak.director.engine.reconstruction.NoiseModel
import com.saunak.director.engine.reconstruction.WhiteBalanceGains

/** Immutable metadata paired to one sensor timestamp. */
data class RawCaptureMetadata(
    val telemetry: CameraSensorTelemetry,
    /** Per CFA channel, in the same order as the camera's CFA layout. */
    val dynamicBlackLevel: FloatArray? = null,
    /** Sensor noise model per CFA channel: S * signal + O. */
    val noiseProfile: List<NoiseModel> = emptyList(),
    val whiteBalance: WhiteBalanceGains? = null,
    val cameraToLinearSrgb: Matrix3x3? = null,
    val neutralColorPoint: FloatArray? = null
) {
    init {
        require(dynamicBlackLevel == null || dynamicBlackLevel.size == 4)
        require(noiseProfile.isEmpty() || noiseProfile.size == 4)
        require(neutralColorPoint == null || neutralColorPoint.size == 3)
    }

    fun representativeNoise(): NoiseModel = if (noiseProfile.isEmpty()) {
        NoiseModel(shotNoiseSlope = 0.002f, readNoiseVariance = 0.0001f)
    } else {
        NoiseModel(
            shotNoiseSlope = noiseProfile.map { it.shotNoiseSlope }.average().toFloat().coerceAtLeast(0f),
            readNoiseVariance = noiseProfile.map { it.readNoiseVariance }.average().toFloat().coerceAtLeast(0f)
        )
    }

    companion object {
        fun fromResult(cameraId: String?, result: TotalCaptureResult): RawCaptureMetadata? {
            val exposure = result.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: return null
            val iso = result.get(CaptureResult.SENSOR_SENSITIVITY) ?: return null
            val timestamp = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return null
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
            val black = result.get(CaptureResult.SENSOR_DYNAMIC_BLACK_LEVEL)
                ?.takeIf { it.size == 4 && it.all(Float::isFinite) }
                ?.copyOf()
            val noise = result.get(CaptureResult.SENSOR_NOISE_PROFILE)
                ?.mapNotNull(::noisePair)
                .orEmpty()
            val gains = result.get(CaptureResult.COLOR_CORRECTION_GAINS)?.let(::whiteBalance)
            val transform = result.get(CaptureResult.COLOR_CORRECTION_TRANSFORM)?.let(::matrix)
            val neutral = result.get(CaptureResult.SENSOR_NEUTRAL_COLOR_POINT)
                ?.takeIf { it.size == 3 }
                ?.map { it.toFloat() }
                ?.toFloatArray()
            return RawCaptureMetadata(telemetry, black, noise, gains, transform, neutral)
        }

        private fun noisePair(pair: Pair<Double, Double>): NoiseModel? {
            val shot = pair.first ?: return null
            val read = pair.second ?: return null
            if (!shot.isFinite() || !read.isFinite() || shot < 0.0 || read < 0.0) return null
            return NoiseModel(shot.toFloat(), read.toFloat())
        }

        private fun whiteBalance(gains: RggbChannelVector): WhiteBalanceGains {
            val green = (gains.greenEven + gains.greenOdd) * 0.5f
            return WhiteBalanceGains(gains.red, green, gains.blue)
        }

        private fun matrix(transform: ColorSpaceTransform): Matrix3x3 {
            val values = FloatArray(9)
            var index = 0
            for (row in 0 until 3) for (col in 0 until 3) {
                val rational: Rational = transform.getElement(col, row)
                values[index++] = rational.toFloat()
            }
            return Matrix3x3(values)
        }
    }
}
