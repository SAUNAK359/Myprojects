package com.saunak.director.camera

import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.util.Locale

data class JpegExifTelemetry(
    val exposureTimeNs: Long? = null,
    val sensitivityIso: Int? = null,
    val focalLengthMm: Double? = null,
    val aperture: Double? = null,
    val exposureBiasEv: Double? = null,
    val whiteBalanceMode: Int? = null,
    val flashTag: Int? = null,
    val imageWidth: Int? = null,
    val imageHeight: Int? = null
) {
    fun compactDescription(): String = buildString {
        append("exifExposureNs=").append(exposureTimeNs ?: "unknown")
        append(";exifIso=").append(sensitivityIso ?: "unknown")
        append(";exifFocalMm=").append(focalLengthMm?.fmt() ?: "unknown")
        append(";exifAperture=").append(aperture?.fmt() ?: "unknown")
        append(";exifBiasEv=").append(exposureBiasEv?.fmt() ?: "unknown")
        append(";exifWb=").append(whiteBalanceMode ?: "unknown")
        append(";exifFlash=").append(flashTag ?: "unknown")
        append(";exifSize=").append(imageWidth ?: "?").append('x').append(imageHeight ?: "?")
    }
}

data class CapturedStillTelemetry(
    val captureId: String,
    val capturedAtEpochMs: Long,
    val shutterState: CameraSnapshot,
    val postCaptureState: CameraSnapshot,
    val sensor: CameraSensorTelemetry?,
    val exif: JpegExifTelemetry?,
    val bindingMethod: String,
    val bindingConfidence: Double
) {
    init {
        require(captureId.isNotBlank())
        require(capturedAtEpochMs > 0L)
        require(bindingMethod.isNotBlank())
        require(bindingConfidence in 0.0..1.0)
    }

    fun compactDescription(): String = buildString {
        append("captureId=").append(captureId)
        append(";capturedAtMs=").append(capturedAtEpochMs)
        append(";binding=").append(bindingMethod)
        append(";bindingConfidence=").append("%.3f".format(Locale.US, bindingConfidence))
        append(";shutter={").append(shutterState.compactDescription()).append('}')
        append(";post={").append(postCaptureState.compactDescription()).append('}')
        sensor?.let {
            append(";camera2={cameraId=").append(it.cameraId ?: "unknown")
            append(";sensorTsNs=").append(it.sensorTimestampNs)
            append(";exposureNs=").append(it.exposureTimeNs)
            append(";iso=").append(it.sensitivityIso)
            append(";frameDurationNs=").append(it.frameDurationNs)
            append(";rollingShutterNs=").append(it.rollingShutterSkewNs ?: "unknown")
            append(";focalMm=").append(it.focalLengthMm ?: "unknown")
            append(";aperture=").append(it.aperture ?: "unknown")
            append(";focusDiopters=").append(it.focusDistanceDiopters ?: "unknown")
            append(";af=").append(it.afState ?: "unknown")
            append(";ae=").append(it.aeState ?: "unknown")
            append(";awb=").append(it.awbState ?: "unknown").append('}')
        }
        exif?.let { append(';').append(it.compactDescription()) }
    }

    companion object {
        fun fromJpeg(
            file: File,
            captureId: String,
            shutterState: CameraSnapshot,
            postCaptureState: CameraSnapshot,
            sensor: CameraSensorTelemetry?
        ): CapturedStillTelemetry {
            val exif = readJpegExif(file)
            val exactExposurePair = exif?.exposureTimeNs != null && exif.sensitivityIso != null
            val confidence = when {
                exactExposurePair && sensor != null -> 0.97
                exactExposurePair -> 0.92
                sensor != null -> 0.78
                else -> 0.55
            }
            return CapturedStillTelemetry(
                captureId = captureId,
                capturedAtEpochMs = file.lastModified().takeIf { it > 0L } ?: System.currentTimeMillis(),
                shutterState = shutterState,
                postCaptureState = postCaptureState,
                sensor = sensor,
                exif = exif,
                bindingMethod = "jpeg_exif_exact_values+camera2_nearest_capture_state",
                bindingConfidence = confidence
            )
        }

        fun fromRaw(
            captureId: String,
            shutterState: CameraSnapshot,
            sensor: CameraSensorTelemetry
        ): CapturedStillTelemetry = CapturedStillTelemetry(
            captureId = captureId,
            capturedAtEpochMs = System.currentTimeMillis(),
            shutterState = shutterState,
            postCaptureState = shutterState,
            sensor = sensor,
            exif = null,
            bindingMethod = "raw_sensor_timestamp_exact",
            bindingConfidence = 1.0
        )

        private fun readJpegExif(file: File): JpegExifTelemetry? = runCatching {
            val exif = ExifInterface(file)
            val exposureSeconds = exif.getAttributeDouble(ExifInterface.TAG_EXPOSURE_TIME, Double.NaN)
                .takeIf { it.isFinite() && it > 0.0 }
            val iso = exif.getAttributeInt(ExifInterface.TAG_PHOTOGRAPHIC_SENSITIVITY, -1).takeIf { it > 0 }
            val focal = exif.getAttributeDouble(ExifInterface.TAG_FOCAL_LENGTH, Double.NaN).takeIf { it.isFinite() && it > 0.0 }
            val aperture = exif.getAttributeDouble(ExifInterface.TAG_F_NUMBER, Double.NaN).takeIf { it.isFinite() && it > 0.0 }
            val bias = exif.getAttributeDouble(ExifInterface.TAG_EXPOSURE_BIAS_VALUE, Double.NaN).takeIf(Double::isFinite)
            val wb = exif.getAttributeInt(ExifInterface.TAG_WHITE_BALANCE, -1).takeIf { it >= 0 }
            val flash = exif.getAttributeInt(ExifInterface.TAG_FLASH, -1).takeIf { it >= 0 }
            val width = exif.getAttributeInt(ExifInterface.TAG_IMAGE_WIDTH, -1).takeIf { it > 0 }
            val height = exif.getAttributeInt(ExifInterface.TAG_IMAGE_LENGTH, -1).takeIf { it > 0 }
            JpegExifTelemetry(
                exposureTimeNs = exposureSeconds?.let { (it * 1_000_000_000.0).toLong().coerceAtLeast(1L) },
                sensitivityIso = iso,
                focalLengthMm = focal,
                aperture = aperture,
                exposureBiasEv = bias,
                whiteBalanceMode = wb,
                flashTag = flash,
                imageWidth = width,
                imageHeight = height
            )
        }.getOrNull()
    }
}

private fun Double.fmt(): String = "%.4f".format(Locale.US, this)
