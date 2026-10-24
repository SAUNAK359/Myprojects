package com.saunak.director.engine.capture

import com.saunak.director.camera.DirectorExtension
import com.saunak.director.engine.device.CameraModuleTwin
import com.saunak.director.engine.device.CapturePath

enum class CaptureExecutionMode {
    RAW_JPEG_BURST,
    OEM_BOUND,
    SAFE_JPEG
}

data class CaptureExecutionCapabilities(
    val supportedImageCaptureFormats: Set<Int>,
    val rawJpegFormatValue: Int,
    val canRestoreNormalSession: Boolean,
    val currentExtension: DirectorExtension
) {
    init { require(rawJpegFormatValue >= 0) }
}

data class CaptureExecutionDecision(
    val mode: CaptureExecutionMode,
    val reason: String
)

/**
 * Hard safety gate between CaptureCompiler intent and camera mutation. A planned RAW path is never
 * sufficient by itself to enable manual RAW capture on an arbitrary Android camera session.
 */
object CaptureExecutionPolicy {
    fun decide(
        program: CaptureProgram,
        camera: CameraModuleTwin,
        runtime: CaptureExecutionCapabilities
    ): CaptureExecutionDecision {
        if (runtime.currentExtension != DirectorExtension.NONE) {
            return CaptureExecutionDecision(
                CaptureExecutionMode.OEM_BOUND,
                "OEM ${runtime.currentExtension.wireName} extension is active; preserve its bound private pipeline"
            )
        }

        if (program.path != CapturePath.RAW_CUSTOM) {
            return CaptureExecutionDecision(
                CaptureExecutionMode.SAFE_JPEG,
                "capture program does not request a certified custom RAW path"
            )
        }
        if (!camera.rawSupported || !camera.manualSensorSupported) {
            return CaptureExecutionDecision(
                CaptureExecutionMode.SAFE_JPEG,
                "Camera2 does not expose both RAW and manual sensor capability"
            )
        }
        if (runtime.rawJpegFormatValue !in runtime.supportedImageCaptureFormats) {
            return CaptureExecutionDecision(
                CaptureExecutionMode.SAFE_JPEG,
                "currently bound CameraX camera does not advertise RAW+JPEG ImageCapture"
            )
        }
        if (!runtime.canRestoreNormalSession) {
            return CaptureExecutionDecision(
                CaptureExecutionMode.SAFE_JPEG,
                "normal CameraX session restoration has not been certified"
            )
        }
        if (program.frames.size < MIN_RAW_BURST_FRAMES) {
            return CaptureExecutionDecision(
                CaptureExecutionMode.SAFE_JPEG,
                "custom RAW program is too short to justify session mutation"
            )
        }
        if (program.frames.any { it.exposureTimeNs <= 0 || it.sensitivityIso <= 0 }) {
            return CaptureExecutionDecision(
                CaptureExecutionMode.SAFE_JPEG,
                "capture program contains invalid physical exposure controls"
            )
        }

        return CaptureExecutionDecision(
            CaptureExecutionMode.RAW_JPEG_BURST,
            "RAW/manual sensor capability, RAW+JPEG CameraX output and restore contract are certified"
        )
    }

    private const val MIN_RAW_BURST_FRAMES = 3
}
