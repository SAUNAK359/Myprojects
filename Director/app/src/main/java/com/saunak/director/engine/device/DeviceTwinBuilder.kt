package com.saunak.director.engine.device

import android.app.ActivityManager
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.params.ColorSpaceTransform
import android.hardware.camera2.params.StreamConfigurationMap
import android.os.Build
import android.util.Size

/** Builds the immutable hardware/calibration snapshot that every Sierra decision is constrained by. */
class DeviceTwinBuilder(private val context: Context) {
    private val cameraManager = context.getSystemService(CameraManager::class.java)

    fun build(): DeviceTwin {
        val cameras = cameraManager.cameraIdList.mapNotNull { id ->
            runCatching { buildCamera(id) }.getOrNull()
        }
        require(cameras.isNotEmpty()) { "No usable Camera2 cameras reported by the device" }

        val memoryInfo = ActivityManager.MemoryInfo().also {
            context.getSystemService(ActivityManager::class.java).getMemoryInfo(it)
        }
        return DeviceTwin(
            fingerprint = buildFingerprint(),
            cameras = cameras,
            computeBackends = detectCandidateBackends(),
            totalMemoryMb = (memoryInfo.totalMem / (1024L * 1024L)).coerceAtLeast(1L),
            createdAtEpochMs = System.currentTimeMillis()
        )
    }

    private fun buildCamera(id: String): CameraModuleTwin {
        val c = cameraManager.getCameraCharacteristics(id)
        val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.toSet().orEmpty()
        val raw = CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW in caps
        val manual = CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR in caps
        val yuvReprocess = CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_YUV_REPROCESSING in caps
        val privateReprocess = CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_PRIVATE_REPROCESSING in caps
        val logical = CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA in caps
        val offline = Build.VERSION.SDK_INT >= 30 && CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_OFFLINE_PROCESSING in caps
        val ultraHighRes = Build.VERSION.SDK_INT >= 31 && CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_ULTRA_HIGH_RESOLUTION_SENSOR in caps

        val active = requireNotNull(c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE))
        val pre = c.get(CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE) ?: active
        val physicalSize = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
        val sensitivity = c.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        val exposure = c.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val timestampSource = c.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
        val normalMap = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val maximumMap = if (Build.VERSION.SDK_INT >= 31) {
            c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP_MAXIMUM_RESOLUTION)
        } else null

        val paths = buildList {
            if (raw && manual) add(CapturePath.RAW_CUSTOM)
            if (yuvReprocess || privateReprocess) add(CapturePath.YUV_REPROCESS)
            add(CapturePath.CAMERA2_SAFE)
        }

        return CameraModuleTwin(
            cameraId = id,
            physicalIds = if (logical) c.physicalCameraIds else emptySet(),
            lensFacing = c.get(CameraCharacteristics.LENS_FACING) ?: CameraCharacteristics.LENS_FACING_EXTERNAL,
            hardwareLevel = c.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
                ?: CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY,
            sensorOrientationDegrees = c.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0,
            cfaPattern = mapCfa(c.get(CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT)),
            activeArrayWidth = active.width(),
            activeArrayHeight = active.height(),
            preCorrectionArrayWidth = pre.width(),
            preCorrectionArrayHeight = pre.height(),
            physicalSensorWidthMm = physicalSize?.width?.toDouble(),
            physicalSensorHeightMm = physicalSize?.height?.toDouble(),
            rawSupported = raw,
            manualSensorSupported = manual,
            yuvReprocessingSupported = yuvReprocess,
            privateReprocessingSupported = privateReprocess,
            offlineProcessingSupported = offline,
            ultraHighResolutionSupported = ultraHighRes,
            logicalMultiCamera = logical,
            syncQuality = mapSyncQuality(logical, c.get(CameraCharacteristics.LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE)),
            focalLengthsMm = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.toList().orEmpty(),
            apertures = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES)?.toList().orEmpty(),
            sensitivityIsoMin = sensitivity?.lower,
            sensitivityIsoMax = sensitivity?.upper,
            maxAnalogSensitivityIso = c.get(CameraCharacteristics.SENSOR_MAX_ANALOG_SENSITIVITY),
            exposureTimeMinNs = exposure?.lower,
            exposureTimeMaxNs = exposure?.upper,
            whiteLevel = c.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL),
            blackLevelPattern = c.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN)?.let { p ->
                listOf(
                    p.getOffsetForIndex(0, 0), p.getOffsetForIndex(1, 0),
                    p.getOffsetForIndex(0, 1), p.getOffsetForIndex(1, 1)
                )
            }.orEmpty(),
            intrinsics = c.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)?.let(::mapIntrinsics),
            distortion = c.get(CameraCharacteristics.LENS_DISTORTION)?.let { DistortionModel(it.map(Float::toDouble)) },
            lensPose = mapLensPose(
                c.get(CameraCharacteristics.LENS_POSE_TRANSLATION),
                c.get(CameraCharacteristics.LENS_POSE_ROTATION)
            ),
            colorCalibration = mapColorCalibration(c),
            timestampSource = timestampSource,
            imuClockComparable = timestampSource == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME,
            streams = (streamProfiles(normalMap, maximumResolutionMode = false) +
                streamProfiles(maximumMap, maximumResolutionMode = true)).distinctBy {
                listOf(it.format, it.width, it.height, it.maximumResolutionMode)
            },
            preferredCapturePaths = paths
        )
    }

    private fun streamProfiles(map: StreamConfigurationMap?, maximumResolutionMode: Boolean): List<StreamProfile> {
        if (map == null) return emptyList()
        val formats = intArrayOf(ImageFormat.YUV_420_888, ImageFormat.JPEG, ImageFormat.RAW_SENSOR, ImageFormat.PRIVATE)
        return buildList {
            for (format in formats) {
                val regular = runCatching { map.getOutputSizes(format)?.toList().orEmpty() }.getOrDefault(emptyList())
                val high = runCatching { map.getHighResolutionOutputSizes(format)?.toList().orEmpty() }.getOrDefault(emptyList())
                for ((size, highResolution) in (regular.map { it to false } + high.map { it to true }).distinct()) {
                    add(streamProfile(map, format, size, highResolution, maximumResolutionMode))
                }
            }
        }
    }

    private fun streamProfile(
        map: StreamConfigurationMap,
        format: Int,
        size: Size,
        highResolution: Boolean,
        maximumResolutionMode: Boolean
    ): StreamProfile = StreamProfile(
        format = format,
        width = size.width,
        height = size.height,
        minFrameDurationNs = runCatching { map.getOutputMinFrameDuration(format, size) }.getOrNull(),
        stallDurationNs = runCatching { map.getOutputStallDuration(format, size) }.getOrNull(),
        highResolution = highResolution,
        maximumResolutionMode = maximumResolutionMode
    )

    private fun mapCfa(value: Int?): CfaPattern = when (value) {
        CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_RGGB -> CfaPattern.RGGB
        CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_BGGR -> CfaPattern.BGGR
        CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GRBG -> CfaPattern.GRBG
        CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GBRG -> CfaPattern.GBRG
        CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_MONO -> CfaPattern.MONO
        else -> CfaPattern.UNKNOWN
    }

    private fun mapSyncQuality(logical: Boolean, value: Int?): CameraSyncQuality {
        if (!logical) return CameraSyncQuality.UNSUPPORTED
        return when (value) {
            CameraCharacteristics.LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE_CALIBRATED -> CameraSyncQuality.CALIBRATED
            CameraCharacteristics.LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE_APPROXIMATE -> CameraSyncQuality.APPROXIMATE
            else -> CameraSyncQuality.UNKNOWN
        }
    }

    private fun mapIntrinsics(v: FloatArray): Intrinsics? =
        if (v.size >= 5 && v[0] > 0f && v[1] > 0f) Intrinsics(v[0].toDouble(), v[1].toDouble(), v[2].toDouble(), v[3].toDouble(), v[4].toDouble()) else null

    private fun mapLensPose(translation: FloatArray?, rotation: FloatArray?): LensPose? {
        if (translation == null && rotation == null) return null
        return LensPose(
            translationMeters = translation?.map(Float::toDouble).orEmpty(),
            quaternion = rotation?.map(Float::toDouble).orEmpty()
        )
    }

    private fun matrix(transform: ColorSpaceTransform?): List<Double> {
        if (transform == null) return emptyList()
        return buildList(9) {
            for (row in 0..2) for (column in 0..2) {
                add(transform.getElement(column, row).toDouble())
            }
        }
    }

    private fun mapColorCalibration(c: CameraCharacteristics): ColorCalibration? {
        val result = ColorCalibration(
            referenceIlluminant1 = c.get(CameraCharacteristics.SENSOR_REFERENCE_ILLUMINANT1),
            referenceIlluminant2 = c.get(CameraCharacteristics.SENSOR_REFERENCE_ILLUMINANT2)?.toInt(),
            calibrationTransform1 = matrix(c.get(CameraCharacteristics.SENSOR_CALIBRATION_TRANSFORM1)),
            calibrationTransform2 = matrix(c.get(CameraCharacteristics.SENSOR_CALIBRATION_TRANSFORM2)),
            colorTransform1 = matrix(c.get(CameraCharacteristics.SENSOR_COLOR_TRANSFORM1)),
            colorTransform2 = matrix(c.get(CameraCharacteristics.SENSOR_COLOR_TRANSFORM2)),
            forwardMatrix1 = matrix(c.get(CameraCharacteristics.SENSOR_FORWARD_MATRIX1)),
            forwardMatrix2 = matrix(c.get(CameraCharacteristics.SENSOR_FORWARD_MATRIX2))
        )
        return result.takeIf {
            it.calibrationTransform1.isNotEmpty() || it.colorTransform1.isNotEmpty() || it.forwardMatrix1.isNotEmpty()
        }
    }

    private fun detectCandidateBackends(): Set<ComputeBackend> = buildSet {
        add(ComputeBackend.CPU_XNNPACK)
        add(ComputeBackend.NNAPI)
        if (context.packageManager.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_LEVEL)) add(ComputeBackend.GPU_VULKAN)
        val soc = if (Build.VERSION.SDK_INT >= 31) {
            "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL} ${Build.HARDWARE}"
        } else Build.HARDWARE
        val normalized = soc.lowercase()
        when {
            "qualcomm" in normalized || "snapdragon" in normalized -> add(ComputeBackend.NPU_QNN)
            "mediatek" in normalized || normalized.startsWith("mt") -> add(ComputeBackend.NPU_MEDIATEK)
            "samsung" in normalized || "exynos" in normalized -> add(ComputeBackend.NPU_SAMSUNG)
        }
    }

    private fun buildFingerprint(): String = buildString {
        append(Build.FINGERPRINT)
        append('|').append(Build.BOARD)
        append('|').append(Build.HARDWARE)
        if (Build.VERSION.SDK_INT >= 31) append('|').append(Build.SOC_MANUFACTURER).append('|').append(Build.SOC_MODEL)
    }
}
