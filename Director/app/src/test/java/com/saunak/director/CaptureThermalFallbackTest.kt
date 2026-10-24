package com.saunak.director

import com.saunak.director.engine.capture.AutoExposureSnapshot
import com.saunak.director.engine.capture.CaptureBudget
import com.saunak.director.engine.capture.CaptureCompiler
import com.saunak.director.engine.capture.CaptureObservation
import com.saunak.director.engine.capture.MotionClass
import com.saunak.director.engine.device.CameraModuleTwin
import com.saunak.director.engine.device.CameraSyncQuality
import com.saunak.director.engine.device.CapturePath
import com.saunak.director.engine.device.CfaPattern
import org.junit.Assert.assertEquals
import org.junit.Test

class CaptureThermalFallbackTest {
    @Test fun severeThermalLimitSuppressesCustomBurst() {
        val camera = CameraModuleTwin(
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
            sensitivityIsoMin = 50,
            sensitivityIsoMax = 6400,
            exposureTimeMinNs = 100_000L,
            exposureTimeMaxNs = 1_000_000_000L,
            preferredCapturePaths = listOf(CapturePath.RAW_CUSTOM, CapturePath.CAMERA2_SAFE)
        )
        val program = CaptureCompiler().compile(
            camera,
            AutoExposureSnapshot(20_000_000L, 400),
            CaptureObservation(13.0, 0.8, 0.8, MotionClass.LOW, MotionClass.LOW, false),
            CaptureBudget(thermalPressure = 1.0, maxBurstFrames = 1)
        )
        assertEquals(CapturePath.CAMERA2_SAFE, program.path)
        assertEquals(1, program.frames.size)
    }
}
