package com.saunak.director.ai

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager

class DeviceCapabilityScanner(context: Context) {
    private val manager = context.getSystemService(CameraManager::class.java)

    fun scan(): String = buildString {
        appendLine("## Phone camera inventory")
        manager.cameraIdList.forEach { id ->
            val c = runCatching { manager.getCameraCharacteristics(id) }.getOrNull() ?: return@forEach
            val facing = when (c.get(CameraCharacteristics.LENS_FACING)) {
                CameraCharacteristics.LENS_FACING_FRONT -> "front"
                CameraCharacteristics.LENS_FACING_BACK -> "back"
                CameraCharacteristics.LENS_FACING_EXTERNAL -> "external"
                else -> "unknown"
            }
            val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.toSet().orEmpty()
            val iso = c.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
            val exposure = c.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
            val ae = c.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)
            val aeStep = c.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_STEP)
            val focal = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.joinToString().orEmpty()
            val apertures = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES)?.joinToString().orEmpty()
            val minFocus = c.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE)
            val ois = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)?.joinToString().orEmpty()
            val videoStab = c.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES)?.joinToString().orEmpty()
            val afModes = c.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)?.joinToString().orEmpty()
            val awbModes = c.get(CameraCharacteristics.CONTROL_AWB_AVAILABLE_MODES)?.joinToString().orEmpty()
            val maxDigitalZoom = c.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM)
            val activeArray = c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            val pixelArray = c.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
            val flash = c.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true
            val hardware = c.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)

            appendLine("### cameraId=$id facing=$facing")
            appendLine("hardwareLevel=$hardware; flash=$flash")
            appendLine("focalLengthsMm=$focal; apertures=$apertures; minimumFocusDistance=$minFocus")
            appendLine("iso=${iso?.lower ?: "unsupported"}..${iso?.upper ?: "unsupported"}")
            appendLine("exposureTimeNs=${exposure?.lower ?: "unsupported"}..${exposure?.upper ?: "unsupported"}")
            appendLine("exposureCompensationIndex=${ae?.lower ?: 0}..${ae?.upper ?: 0}; step=${aeStep ?: "unknown"}")
            appendLine("maxDigitalZoom=${maxDigitalZoom ?: 1f}")
            appendLine("afModes=$afModes; awbModes=$awbModes")
            appendLine("opticalStabilizationModes=$ois; videoStabilizationModes=$videoStab")
            appendLine("sensorActiveArray=$activeArray; sensorPixelArray=$pixelArray")
            appendLine("manualSensor=${CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR in caps}; raw=${CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW in caps}; logicalMultiCamera=${CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA in caps}")
            appendLine()
        }
    }
}
