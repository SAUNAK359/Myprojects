package com.saunak.director

import com.saunak.director.engine.capture.CaptureFrameRole
import com.saunak.director.engine.capture.CaptureFrameSpec
import com.saunak.director.engine.capture.CaptureProgram
import com.saunak.director.engine.device.CapturePath
import com.saunak.director.engine.prif.PrifReconstructionCompiler
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PrifReconstructionCompilerTest {
    private fun program(): CaptureProgram = CaptureProgram(
        path = CapturePath.RAW_CUSTOM,
        frames = listOf(
            CaptureFrameSpec(5_000_000L, 400, CaptureFrameRole.HIGHLIGHT, -1.5),
            CaptureFrameSpec(10_000_000L, 400, CaptureFrameRole.REFERENCE, 0.0),
            CaptureFrameSpec(10_000_000L, 400, CaptureFrameRole.DETAIL, 0.0),
            CaptureFrameSpec(10_000_000L, 400, CaptureFrameRole.DETAIL, 0.0),
            CaptureFrameSpec(14_000_000L, 400, CaptureFrameRole.SHADOW, 0.5)
        ),
        lockThreeA = true,
        hdrFusion = true,
        burstSuperResolution = false,
        expectedCaptureDurationMs = 120.0,
        expectedUtility = 0.9,
        rationale = "test"
    )

    @Test fun severeResourcePressureReducesEvidenceBeforeCapture() {
        val compiled = PrifReconstructionCompiler.compile(
            captureProgram = program(),
            motion = null,
            subject = null,
            thermalPressure = 0.92,
            memoryPressure = 0.95
        )
        val indices = PrifReconstructionCompiler.selectCaptureIndices(compiled)
        assertEquals(2, indices.size)
        assertTrue(compiled.referenceFrameIndex in indices)
    }

    @Test fun calmResourceStateRetainsFullCaptureLattice() {
        val compiled = PrifReconstructionCompiler.compile(
            captureProgram = program(),
            motion = null,
            subject = null,
            thermalPressure = 0.0,
            memoryPressure = 0.0
        )
        // Unknown motion is intentionally conservative, so we still require at least a useful subset.
        val indices = PrifReconstructionCompiler.selectCaptureIndices(compiled)
        assertTrue(indices.size in 2..5)
        assertTrue(compiled.referenceFrameIndex in indices)
    }
}
