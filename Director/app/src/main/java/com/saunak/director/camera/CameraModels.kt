package com.saunak.director.camera

data class CapabilityRegistry(
    val lensFacing: Int,
    val minZoom: Float,
    val maxZoom: Float,
    val exposureMin: Int,
    val exposureMax: Int,
    val exposureStepEv: Float,
    val hasFlash: Boolean,
    val torchSupported: Boolean,
    val manualSensorSupported: Boolean = false,
    val allowedToolIds: Set<Int>
) {
    fun compactDescription(): String = buildString {
        append("zoom=").append(minZoom).append("..").append(maxZoom)
        append(";evIndex=").append(exposureMin).append("..").append(exposureMax)
        append(";evStep=").append(exposureStepEv)
        append(";flash=").append(hasFlash)
        append(";torch=").append(torchSupported)
        append(";manualSensor=").append(manualSensorSupported)
    }
}

data class CameraSnapshot(
    val lensFacing: Int,
    val zoomRatio: Float,
    val exposureCompensationIndex: Int,
    val torchEnabled: Boolean,
    val rotationDegrees: Int = 0
) {
    fun compactDescription(): String = "lens=$lensFacing;zoom=$zoomRatio;evIndex=$exposureCompensationIndex;torch=$torchEnabled;rot=$rotationDegrees"
}
