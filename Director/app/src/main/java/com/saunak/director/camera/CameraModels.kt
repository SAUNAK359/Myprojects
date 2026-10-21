package com.saunak.director.camera

enum class DirectorExtension(val wireName: String) {
    NONE("none"), AUTO("auto"), HDR("hdr"), NIGHT("night"), BOKEH("bokeh"), FACE_RETOUCH("face_retouch");
    companion object { fun fromWire(value: String?): DirectorExtension? = entries.firstOrNull { it.wireName == value?.lowercase() } }
}

enum class StillFlashMode(val wireName: String) {
    OFF("off"), AUTO("auto"), ON("on");
    companion object { fun fromWire(value: String?): StillFlashMode? = entries.firstOrNull { it.wireName == value?.lowercase() } }
}

data class CapabilityRegistry(
    val lensFacing: Int,
    val minZoom: Float,
    val maxZoom: Float,
    val exposureMinIndex: Int,
    val exposureMaxIndex: Int,
    val exposureStepEv: Float,
    val hasFlash: Boolean,
    val torchSupported: Boolean,
    val focusMeteringSupported: Boolean,
    val manualSensorSupported: Boolean,
    val rawSupported: Boolean,
    val isoMin: Int? = null,
    val isoMax: Int? = null,
    val exposureTimeMinNs: Long? = null,
    val exposureTimeMaxNs: Long? = null,
    val minimumFocusDistance: Float? = null,
    val focalLengthsMm: List<Float> = emptyList(),
    val opticalStabilizationModes: List<Int> = emptyList(),
    val videoStabilizationModes: List<Int> = emptyList(),
    val extensions: Set<DirectorExtension> = emptySet(),
    val allowedToolIds: Set<Int>
) {
    val minExposureEv: Float get() = exposureMinIndex * exposureStepEv
    val maxExposureEv: Float get() = exposureMaxIndex * exposureStepEv
    fun compactDescription(): String = buildString {
        append("zoom=").append(minZoom).append("..").append(maxZoom)
        append(";ev=").append(minExposureEv).append("..").append(maxExposureEv)
        append(";evStep=").append(exposureStepEv)
        append(";flash=").append(hasFlash)
        append(";focus=").append(focusMeteringSupported)
        append(";manualSensor=").append(manualSensorSupported)
        append(";raw=").append(rawSupported)
        if (focalLengthsMm.isNotEmpty()) append(";focalMm=").append(focalLengthsMm.joinToString(","))
        append(";extensions=").append(extensions.joinToString(",") { it.wireName })
    }
}

data class CameraSnapshot(
    val lensFacing: Int,
    val zoomRatio: Float,
    val exposureCompensationIndex: Int,
    val torchEnabled: Boolean,
    val flashMode: StillFlashMode,
    val extension: DirectorExtension,
    val rotationDegrees: Int = 0
) {
    fun compactDescription(): String = "lens=$lensFacing;zoom=$zoomRatio;evIndex=$exposureCompensationIndex;torch=$torchEnabled;flash=${flashMode.wireName};extension=${extension.wireName};rot=$rotationDegrees"
}

data class CameraUiState(
    val minZoom: Float = 1f,
    val maxZoom: Float = 1f,
    val zoom: Float = 1f,
    val exposureSupported: Boolean = false,
    val exposureMinIndex: Int = 0,
    val exposureMaxIndex: Int = 0,
    val exposureIndex: Int = 0,
    val exposureStepEv: Float = 0f,
    val torchSupported: Boolean = false,
    val torchEnabled: Boolean = false,
    val flashMode: StillFlashMode = StillFlashMode.OFF,
    val lensFacing: Int = 1,
    val extensions: Set<DirectorExtension> = emptySet()
)
