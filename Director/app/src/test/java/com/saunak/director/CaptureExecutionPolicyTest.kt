package com.saunak.director

import com.saunak.director.camera.DirectorExtension
import com.saunak.director.engine.capture.CaptureExecutionCapabilities
import com.saunak.director.engine.capture.CaptureExecutionMode
import com.saunak.director.engine.capture.CaptureExecutionPolicy
import com.saunak.director.engine.capture.CaptureFrameRole
import com.saunak.director.engine.capture.CaptureFrameSpec
import com.saunak.director.engine.capture.CaptureProgram
import com.saunak.director.engine.device.CameraModuleTwin
import com.saunak.director.engine.device.CameraSyncQuality
import com.saunak.director.engine.device.CapturePath
import com.saunak.director.engine.device.CfaPattern
import org.junit.Assert.assertEquals
import org.junit.Test

class CaptureExecutionPolicyTest {
    private val camera = CameraModuleTwin(
        cameraId = "0",
        physicalIds = emptySet(),
        lensFacing = 1,
        hardwareLevel = 1,
        sensorOrientationDegrees = 90,
        cfaPattern = CfaPattern.RGGB,
        activeArrayWidth = 4000,
        activeArrayHeight = 3000,
        preCorrectionArrayWidth = 4000,
        preCorrectionArrayHeight = 3000,
        rawSupported = true,
        manualSensorSupported = true,
        yuvReprocessingSupported = false,
        privateReprocessingSupported = false,
        offlineProcessingSupported = false,
        ultraHighResolutionSupported = false,
        logicalMultiCamera = false,
        syncQuality = CameraSyncQuality.UNSUPPORTED,
        focalLengthsMm = listOf(6f),
        apertures = listOf(1.8f),
        preferredCapturePaths = listOf(CapturePath.RAW_CUSTOM, CapturePath.CAMERA2_SAFE)
    )

    private val program = CaptureProgram(
        path = CapturePath.RAW_CUSTOM,
        frames = List(4) { CaptureFrameSpec(10_000_000L, 200, CaptureFrameRole.REFERENCE, 0.0) },
        lockThreeA = true,
        hdrFusion = true,
        burstSuperResolution = false,
        expectedCaptureDurationMs = 120.0,
        expectedUtility = 0.9,
        rationale = "test"
    )

    @Test fun certifiedRawSessionMayExecuteRawJpegBurst() {
        val decision = CaptureExecutionPolicy.decide(
            program,
            camera,
            CaptureExecutionCapabilities(setOf(0, 3), rawJpegFormatValue = 3, canRestoreNormalSession = true, currentExtension = DirectorExtension.NONE)
        )
        assertEquals(CaptureExecutionMode.RAW_JPEG_BURST, decision.mode)
    }

    @Test fun activeOemExtensionAlwaysWinsOverCustomRaw() {
        val decision = CaptureExecutionPolicy.decide(
            program,
            camera,
            CaptureExecutionCapabilities(setOf(3), rawJpegFormatValue = 3, canRestoreNormalSession = true, currentExtension = DirectorExtension.NIGHT)
        )
        assertEquals(CaptureExecutionMode.OEM_BOUND, decision.mode)
    }

    @Test fun missingRawJpegCapabilityFallsBackSafely() {
        val decision = CaptureExecutionPolicy.decide(
            program,
            camera,
            CaptureExecutionCapabilities(setOf(0, 2), rawJpegFormatValue = 3, canRestoreNormalSession = true, currentExtension = DirectorExtension.NONE)
        )
        assertEquals(CaptureExecutionMode.SAFE_JPEG, decision.mode)
    }
}
