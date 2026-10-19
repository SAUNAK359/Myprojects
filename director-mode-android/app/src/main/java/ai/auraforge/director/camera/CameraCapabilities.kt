package ai.auraforge.director.camera

import android.graphics.ImageFormat
import android.hardware.camera2.CameraCharacteristics
import android.os.Build
import android.util.Range
import android.util.Rational
import android.util.Size
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraInfo
import androidx.camera.core.ImageCapture

data class CameraCapabilities(
    val hardwareLevel: Int?,
    val supportsManualSensor: Boolean,
    val supportsRaw: Boolean,
    val supportsRawJpeg: Boolean,
    val supportsUltraHdr: Boolean,
    val supportsLogicalMultiCamera: Boolean,
    val flashAvailable: Boolean,
    val sensitivityRange: Range<Int>?,
    val exposureTimeRange: Range<Long>?,
    val exposureCompensationRange: Range<Int>?,
    val exposureCompensationStep: Rational?,
    val minFocusDistance: Float?,
    val maxDigitalZoom: Float,
    val apertures: List<Float>,
    val focalLengthsMm: List<Float>,
    val awbModes: Set<Int>,
    val effectModes: Set<Int>,
    val sceneModes: Set<Int>,
    val afModes: Set<Int>,
    val aeModes: Set<Int>,
    val maxJpegSize: Size?,
    val native4K: Boolean
) {
    val supportsVariableAperture: Boolean get() = apertures.distinct().size > 1
    val supportsManualFocus: Boolean get() = (minFocusDistance ?: 0f) > 0f
}

@OptIn(ExperimentalCamera2Interop::class)
object CameraCapabilityInspector {
    fun inspect(cameraInfo: CameraInfo): CameraCapabilities {
        val info = Camera2CameraInfo.from(cameraInfo)
        val level = info.getCameraCharacteristic(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
        val caps = info.getCameraCharacteristic(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.toSet().orEmpty()
        val map = info.getCameraCharacteristic(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val outputFormats = runCatching { ImageCapture.getImageCaptureCapabilities(cameraInfo).supportedOutputFormats.toSet() }.getOrDefault(emptySet())

        val ordinary = map?.getOutputSizes(ImageFormat.JPEG)?.toList().orEmpty()
        val highResolution = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            map?.getHighResolutionOutputSizes(ImageFormat.JPEG)?.toList().orEmpty()
        } else emptyList()
        val maxJpeg = (ordinary + highResolution)
            .distinctBy { it.width to it.height }
            .maxByOrNull { it.width.toLong() * it.height.toLong() }

        val longSide = maxOf(maxJpeg?.width ?: 0, maxJpeg?.height ?: 0)
        val shortSide = minOf(maxJpeg?.width ?: 0, maxJpeg?.height ?: 0)

        return CameraCapabilities(
            hardwareLevel = level,
            supportsManualSensor = CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR in caps,
            supportsRaw = CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW in caps,
            supportsRawJpeg = ImageCapture.OUTPUT_FORMAT_RAW_JPEG in outputFormats,
            supportsUltraHdr = ImageCapture.OUTPUT_FORMAT_JPEG_ULTRA_HDR in outputFormats,
            supportsLogicalMultiCamera = CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA in caps,
            flashAvailable = info.getCameraCharacteristic(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true,
            sensitivityRange = info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE),
            exposureTimeRange = info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE),
            exposureCompensationRange = info.getCameraCharacteristic(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE),
            exposureCompensationStep = info.getCameraCharacteristic(CameraCharacteristics.CONTROL_AE_COMPENSATION_STEP),
            minFocusDistance = info.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE),
            maxDigitalZoom = info.getCameraCharacteristic(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM) ?: 1f,
            apertures = info.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES)?.toList().orEmpty().sorted(),
            focalLengthsMm = info.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.toList().orEmpty().sorted(),
            awbModes = info.getCameraCharacteristic(CameraCharacteristics.CONTROL_AWB_AVAILABLE_MODES)?.toSet().orEmpty(),
            effectModes = info.getCameraCharacteristic(CameraCharacteristics.CONTROL_AVAILABLE_EFFECTS)?.toSet().orEmpty(),
            sceneModes = info.getCameraCharacteristic(CameraCharacteristics.CONTROL_AVAILABLE_SCENE_MODES)?.toSet().orEmpty(),
            afModes = info.getCameraCharacteristic(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)?.toSet().orEmpty(),
            aeModes = info.getCameraCharacteristic(CameraCharacteristics.CONTROL_AE_AVAILABLE_MODES)?.toSet().orEmpty(),
            maxJpegSize = maxJpeg,
            native4K = longSide >= 3840 && shortSide >= 2160
        )
    }
}
