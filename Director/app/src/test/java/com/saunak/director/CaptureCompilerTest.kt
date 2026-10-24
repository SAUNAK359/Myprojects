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
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CaptureCompilerTest {
    private fun camera(raw: Boolean = true) = CameraModuleTwin(
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
        rawSupported = raw,
        manualSensorSupported = raw,
        yuvReprocessingSupported = !raw,
        privateReprocessingSupported = false,
        offlineProcessingSupported = false,
        ultraHighResolutionSupported = false,
        logicalMultiCamera = false,
        syncQuality = CameraSyncQuality.UNSUPPORTED,
        focalLengthsMm = listOf(6.0f),
        apertures = listOf(1.8f),
        sensitivityIsoMin = 50,
        sensitivityIsoMax = 6400,
        exposureTimeMinNs = 100_000L,
        exposureTimeMaxNs = 1_000_000_000L,
        preferredCapturePaths = if (raw) listOf(CapturePath.RAW_CUSTOM, CapturePath.CAMERA2_SAFE)
        else listOf(CapturePath.YUV_REPROCESS, CapturePath.CAMERA2_SAFE)
    )

    @Test fun highDynamicRangeStaticSceneGetsBoundedRawProgram() {
        val program = CaptureCompiler().compile(
            camera(),
            AutoExposureSnapshot(16_000_000L, 200),
            CaptureObservation(13.0, 0.8, 0.7, MotionClass.STATIC, MotionClass.STATIC, facePresent = false),
            CaptureBudget(maxBurstFrames = 8)
        )
        assertEquals(CapturePath.RAW_CUSTOM, program.path)
        assertTrue(program.hdrFusion)
        assertTrue(program.frames.size in 3..8)
        assertTrue(program.frames.all { it.sensitivityIso in 50..6400 })
        assertTrue(program.frames.all { it.exposureTimeNs in 100_000L..1_000_000_000L })
    }

    @Test fun highMotionCapsShutterTime() {
        val program = CaptureCompiler().compile(
            camera(),
            AutoExposureSnapshot(50_000_000L, 100),
            CaptureObservation(8.0, 0.1, 0.6, MotionClass.HIGH, MotionClass.HIGH, facePresent = true),
            CaptureBudget(maxBurstFrames = 5)
        )
        assertTrue(program.frames.all { it.exposureTimeNs <= 8_000_000L })
    }

    @Test fun fallsBackToYuvWhenRawManualIsUnavailable() {
        val program = CaptureCompiler().compile(
            camera(raw = false),
            AutoExposureSnapshot(10_000_000L, 400),
            CaptureObservation(9.0, 0.2, 0.6, MotionClass.LOW, MotionClass.LOW, facePresent = false),
            CaptureBudget(maxBurstFrames = 5)
        )
        assertEquals(CapturePath.YUV_REPROCESS, program.path)
    }

    @Test fun confidentFastSubjectSuppressesMixedExposureAndSuperResolution() {
        val program = CaptureCompiler().compile(
            camera(),
            AutoExposureSnapshot(20_000_000L, 200),
            CaptureObservation(
                dynamicRangeStops = 13.0,
                highlightClipRisk = 0.8,
                shadowNoiseRisk = 0.72,
                cameraMotion = MotionClass.LOW,
                subjectMotion = MotionClass.LOW,
                facePresent = true,
                digitalZoomRatio = 2.0,
                confidence = 0.92,
                motionExposureCapNs = 7_000_000L,
                motionRisk = 0.58,
                subjectKinematicRisk = 0.86,
                subjectEdgeRisk = 0.82,
                subjectTrackConfidence = 0.91
            ),
            CaptureBudget(maxBurstFrames = 8)
        )
        assertTrue(program.frames.size <= 4)
        assertTrue(program.frames.all { it.exposureTimeNs <= 7_000_000L })
        assertFalse(program.burstSuperResolution)
    }

    @Test fun stableTrackedSubjectCanRetainSuperResolutionEligibility() {
        val program = CaptureCompiler().compile(
            camera(),
            AutoExposureSnapshot(10_000_000L, 160),
            CaptureObservation(
                dynamicRangeStops = 8.0,
                highlightClipRisk = 0.10,
                shadowNoiseRisk = 0.55,
                cameraMotion = MotionClass.LOW,
                subjectMotion = MotionClass.LOW,
                facePresent = true,
                digitalZoomRatio = 2.0,
                confidence = 0.95,
                motionExposureCapNs = 22_000_000L,
                motionRisk = 0.24,
                subjectKinematicRisk = 0.12,
                subjectEdgeRisk = 0.10,
                subjectTrackConfidence = 0.93
            ),
            CaptureBudget(maxBurstFrames = 7)
        )
        assertEquals(CapturePath.RAW_CUSTOM, program.path)
        assertTrue(program.burstSuperResolution)
    }
}
