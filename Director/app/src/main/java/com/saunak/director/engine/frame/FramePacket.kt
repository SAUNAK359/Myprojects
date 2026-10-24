package com.saunak.director.engine.frame

/** Immutable sensor metadata bound to exactly one camera frame. */
enum class FramePixelFormat { YUV_420_888, RAW_SENSOR, RAW_PRIVATE, JPEG, PRIVATE, RGBA_8888, UNKNOWN }

data class ImuSample(
    val timestampNs: Long,
    val x: Double,
    val y: Double,
    val z: Double
)

data class ImuWindow(
    val gyro: List<ImuSample> = emptyList(),
    val accelerometer: List<ImuSample> = emptyList(),
    /** 0 means the camera and IMU clock domains are not trusted to be comparable. */
    val syncConfidence: Double = 0.0
) {
    init { require(syncConfidence in 0.0..1.0) }
}

data class SensorExposure(
    val exposureTimeNs: Long,
    val sensitivityIso: Int,
    val frameDurationNs: Long,
    val rollingShutterSkewNs: Long? = null
) {
    init {
        require(exposureTimeNs > 0)
        require(sensitivityIso > 0)
        require(frameDurationNs >= exposureTimeNs)
        require(rollingShutterSkewNs == null || rollingShutterSkewNs >= 0)
    }
}

data class FrameGeometry(
    val width: Int,
    val height: Int,
    val rotationDegrees: Int,
    val cropLeft: Int = 0,
    val cropTop: Int = 0,
    val cropRight: Int = width,
    val cropBottom: Int = height
) {
    init {
        require(width > 0 && height > 0)
        require(rotationDegrees in setOf(0, 90, 180, 270))
        require(cropLeft in 0 until cropRight && cropRight <= width)
        require(cropTop in 0 until cropBottom && cropBottom <= height)
    }
}

data class FramePacket(
    val frameId: Long,
    /** SENSOR_TIMESTAMP for the image; never substitute wall-clock time. */
    val sensorTimestampNs: Long,
    val requestId: Long? = null,
    val logicalCameraId: String,
    val physicalCameraId: String? = null,
    val pixelFormat: FramePixelFormat,
    val geometry: FrameGeometry,
    val exposure: SensorExposure,
    val focalLengthMm: Float? = null,
    val aperture: Float? = null,
    val focusDistanceDiopters: Float? = null,
    val zoomRatio: Float = 1f,
    val imu: ImuWindow = ImuWindow(),
    /** Small numeric metadata only. Pixel buffers remain owned by the camera buffer fabric. */
    val numericMetadata: Map<String, Double> = emptyMap()
) {
    init {
        require(frameId >= 0)
        require(sensorTimestampNs > 0)
        require(logicalCameraId.isNotBlank())
        require(zoomRatio >= 1f)
        require(focalLengthMm == null || focalLengthMm > 0f)
        require(aperture == null || aperture > 0f)
        require(focusDistanceDiopters == null || focusDistanceDiopters >= 0f)
    }

    fun ageNs(nowSensorTimeNs: Long): Long = (nowSensorTimeNs - sensorTimestampNs).coerceAtLeast(0L)
}
