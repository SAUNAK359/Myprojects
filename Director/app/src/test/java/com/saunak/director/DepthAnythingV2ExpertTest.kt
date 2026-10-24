package com.saunak.director

import com.saunak.director.engine.experts.DepthAnythingV2Expert
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DepthAnythingV2ExpertTest {
    @Test fun landscapeInputPreservesAspectAndPatchAlignment() {
        val (width, height) = DepthAnythingV2Expert.contractInputSize(1920, 1080)
        assertEquals(518, width)
        assertTrue(height <= 518)
        assertEquals(0, height % 14)
        val sourceRatio = 1920.0 / 1080.0
        val modelRatio = width.toDouble() / height.toDouble()
        assertTrue(kotlin.math.abs(sourceRatio - modelRatio) < 0.06)
    }

    @Test fun portraitInputPreservesAspectAndPatchAlignment() {
        val (width, height) = DepthAnythingV2Expert.contractInputSize(1080, 1920)
        assertEquals(518, height)
        assertTrue(width <= 518)
        assertEquals(0, width % 14)
    }

    @Test fun squareInputUsesFullModelResolution() {
        assertEquals(518 to 518, DepthAnythingV2Expert.contractInputSize(1024, 1024))
    }
}
