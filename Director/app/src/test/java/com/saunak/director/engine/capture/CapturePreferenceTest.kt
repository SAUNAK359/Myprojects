package com.saunak.director.engine.capture

import com.saunak.director.engine.device.CapturePath
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CapturePreferenceTest {
    private val compiler = CaptureCompiler()

    @Test
    fun hdrPreferenceRewardsOnlyExistingHdrCandidate() {
        val single = program(frames = 1, hdr = false, utility = 0.70, durationMs = 70.0)
        val hdr = program(frames = 5, hdr = true, utility = 0.70, durationMs = 210.0, brackets = true)
        val preference = CapturePreference(highlightProtection = 1.0, preferHdr = 1.0)

        val singleScore = compiler.intentAwareUtility(single, preference)
        val hdrScore = compiler.intentAwareUtility(hdr, preference)

        assertTrue(hdrScore > singleScore)
        assertEquals(0.70, compiler.intentAwareUtility(single, null), 1e-9)
    }

    @Test
    fun motionPreferenceDoesNotCreateUnboundedUtilityGain() {
        val fast = program(frames = 1, hdr = false, utility = 0.62, durationMs = 50.0)
        val slow = program(frames = 7, hdr = true, utility = 0.62, durationMs = 430.0, brackets = true)
        val preference = CapturePreference(motionFreeze = 1.0)

        assertTrue(compiler.intentAwareUtility(fast, preference) > compiler.intentAwareUtility(slow, preference))
        assertTrue(compiler.intentAwareUtility(fast, preference) <= 0.78)
    }

    private fun program(
        frames: Int,
        hdr: Boolean,
        utility: Double,
        durationMs: Double,
        brackets: Boolean = false
    ): CaptureProgram {
        val list = List(frames) { index ->
            CaptureFrameSpec(
                exposureTimeNs = 8_000_000L,
                sensitivityIso = 200,
                role = when {
                    brackets && index == 0 -> CaptureFrameRole.HIGHLIGHT
                    brackets && index == frames - 1 -> CaptureFrameRole.SHADOW
                    index == 0 -> CaptureFrameRole.REFERENCE
                    else -> CaptureFrameRole.DETAIL
                },
                evOffset = 0.0
            )
        }
        return CaptureProgram(
            path = CapturePath.CAMERA2_SAFE,
            frames = list,
            lockThreeA = frames > 1,
            hdrFusion = hdr,
            burstSuperResolution = false,
            expectedCaptureDurationMs = durationMs,
            expectedUtility = utility,
            rationale = "test"
        )
    }
}
