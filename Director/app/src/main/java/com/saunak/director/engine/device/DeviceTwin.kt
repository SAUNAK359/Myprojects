package com.saunak.director.engine.device

enum class ComputeBackend { CPU_XNNPACK, GPU_VULKAN, NPU_QNN, NPU_MEDIATEK, NPU_SAMSUNG, NNAPI, DSP, UNKNOWN }
enum class CameraSyncQuality { CALIBRATED, APPROXIMATE, UNSUPPORTED, UNKNOWN }
enum class CfaPattern { RGGB, BGGR, GRBG, GBRG, QUAD_BAYER, BINNED_BAYER, REMOSAICED_BAYER, MONO, UNKNOWN }
enum class CapturePath { RAW_CUSTOM, YUV_REPROCESS, OEM_EXTENSION, CAMERA2_SAFE }

data class Intrinsics(
    val fx: Double,
    val fy: Double,
    val cx: Double,
    val cy: Double,
    val skew: Double = 0.0
) {
    init { require(fx > 0 && fy > 0) }
}

data class DistortionModel(
    val coefficients: List<Double> = emptyList()
)

data class NoiseModel(
    /** Poisson-Gaussian approximation: variance = shot * signal + read. */
    val shotNoise: Double,
    val readNoise: Double
) {
    init { require(shotNoise >= 0 && readNoise >= 0) }
}

data class StreamProfile(
    val format: Int,
    val width: Int,
    val height: Int,
    val minFrameDurationNs: Long?,
    val stallDurationNs: Long?,
    val highResolution: Boolean = false
) {
    init {
        require(width > 0 && height > 0)
        require(minFrameDurationNs == null || minFrameDurationNs >= 0)
        require(stallDurationNs == null || stallDurationNs >= 0)
    }
}

data class CameraModuleTwin(
    val cameraId: String,
    val physicalIds: Set<String>,
    val lensFacing: Int,
    val hardwareLevel: Int,
    val cfaPattern: CfaPattern,
    val activeArrayWidth: Int,
    val activeArrayHeight: Int,
    val rawSupported: Boolean,
    val manualSensorSupported: Boolean,
    val yuvReprocessingSupported: Boolean,
    val privateReprocessingSupported: Boolean,
    val offlineProcessingSupported: Boolean,
    val ultraHighResolutionSupported: Boolean,
    val logicalMultiCamera: Boolean,
    val syncQuality: CameraSyncQuality,
    val focalLengthsMm: List<Float>,
    val apertures: List<Float>,
    val intrinsics: Intrinsics? = null,
    val distortion: DistortionModel? = null,
    val timestampSource: Int? = null,
    val imuClockComparable: Boolean = false,
    val streams: List<StreamProfile> = emptyList(),
    val preferredCapturePaths: List<CapturePath> = emptyList()
) {
    init {
        require(cameraId.isNotBlank())
        require(activeArrayWidth > 0 && activeArrayHeight > 0)
    }
}

data class BackendExecutionProfile(
    val modelId: String,
    val backend: ComputeBackend,
    val inputSignature: String,
    val graphCoverage: Double,
    val transferBoundaries: Int,
    val p50LatencyMs: Double,
    val p95LatencyMs: Double,
    val peakMemoryMb: Double,
    val sustainedPowerScore: Double
) {
    init {
        require(modelId.isNotBlank())
        require(inputSignature.isNotBlank())
        require(graphCoverage in 0.0..1.0)
        require(transferBoundaries >= 0)
        require(p50LatencyMs >= 0 && p95LatencyMs >= p50LatencyMs)
        require(peakMemoryMb >= 0 && sustainedPowerScore >= 0)
    }
}

data class InterferenceProfile(
    val firstWorkload: String,
    val secondWorkload: String,
    /** 0.25 means 25% slower than isolated execution. */
    val latencyPenaltyFraction: Double
) {
    init { require(firstWorkload.isNotBlank() && secondWorkload.isNotBlank()); require(latencyPenaltyFraction >= 0) }
}

data class DeviceTwin(
    val fingerprint: String,
    val cameras: List<CameraModuleTwin>,
    val computeBackends: Set<ComputeBackend>,
    val backendProfiles: List<BackendExecutionProfile> = emptyList(),
    val interference: List<InterferenceProfile> = emptyList(),
    val totalMemoryMb: Long,
    val createdAtEpochMs: Long,
    val schemaVersion: Int = CURRENT_SCHEMA
) {
    init {
        require(fingerprint.isNotBlank())
        require(cameras.isNotEmpty())
        require(totalMemoryMb > 0)
        require(schemaVersion == CURRENT_SCHEMA)
    }

    fun camera(id: String): CameraModuleTwin? = cameras.firstOrNull { it.cameraId == id }

    companion object { const val CURRENT_SCHEMA = 3 }
}
