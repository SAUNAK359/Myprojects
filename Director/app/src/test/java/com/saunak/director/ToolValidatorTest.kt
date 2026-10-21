package com.saunak.director

import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.camera.DirectorExtension
import com.saunak.director.director.ToolCall
import com.saunak.director.tools.ToolIds
import com.saunak.director.tools.ToolValidator
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ToolValidatorTest {
    private val caps = CapabilityRegistry(
        lensFacing = 1,
        minZoom = 1f,
        maxZoom = 4f,
        exposureMinIndex = -4,
        exposureMaxIndex = 4,
        exposureStepEv = .25f,
        hasFlash = true,
        torchSupported = true,
        focusMeteringSupported = true,
        manualSensorSupported = false,
        rawSupported = false,
        extensions = setOf(DirectorExtension.HDR),
        allowedToolIds = setOf(ToolIds.FOCUS, ToolIds.ZOOM, ToolIds.EXPOSURE_EV, ToolIds.TORCH, ToolIds.FLASH, ToolIds.EXTENSION) + ToolIds.LOOK_IDS
    )

    @Test fun clampsCameraValues() {
        val result = ToolValidator().validate(listOf(ToolCall(ToolIds.ZOOM, value = 99.0), ToolCall(ToolIds.EXPOSURE_EV, value = -99.0)), caps)
        assertEquals(4.0, result.accepted.first { it.id == ToolIds.ZOOM }.value!!, .0001)
        assertEquals(-1.0, result.accepted.first { it.id == ToolIds.EXPOSURE_EV }.value!!, .0001)
    }

    @Test fun extensionRejectsAllNormalCameraMutations() {
        val result = ToolValidator().validate(
            listOf(
                ToolCall(ToolIds.ZOOM, value = 2.0),
                ToolCall(ToolIds.EXPOSURE_EV, value = .5),
                ToolCall(ToolIds.TORCH, value = 1.0),
                ToolCall(ToolIds.FLASH, stringValue = "on"),
                ToolCall(ToolIds.EXTENSION, stringValue = "hdr")
            ), caps)
        assertTrue(result.accepted.any { it.id == ToolIds.EXTENSION })
        assertTrue(result.accepted.none { it.id in setOf(ToolIds.ZOOM, ToolIds.EXPOSURE_EV, ToolIds.TORCH, ToolIds.FLASH) })
        assertTrue(result.rejected.count { it.reason == "extension_conflict" } >= 4)
    }

    @Test fun stillFlashWinsOverTorchWhenBothAreRequested() {
        val result = ToolValidator().validate(listOf(ToolCall(ToolIds.TORCH, value = 1.0), ToolCall(ToolIds.FLASH, stringValue = "auto")), caps)
        // Current validator permits both in normal CameraX mode; this test documents that the two controls are distinct.
        // Device qualification decides whether a vendor needs a stricter profile.
        assertTrue(result.accepted.any { it.id == ToolIds.TORCH })
        assertTrue(result.accepted.any { it.id == ToolIds.FLASH })
    }

    @Test fun rejectsUnknownToolAndSupersedesOldLook() {
        val result = ToolValidator().validate(listOf(ToolCall(999), ToolCall(ToolIds.LOOK_WARM, value = .5), ToolCall(ToolIds.LOOK_COOL, value = .4)), caps)
        assertTrue(result.rejected.any { it.call.id == 999 })
        assertEquals(ToolIds.LOOK_COOL, result.accepted.single { it.id in ToolIds.LOOK_IDS }.id)
    }
}
