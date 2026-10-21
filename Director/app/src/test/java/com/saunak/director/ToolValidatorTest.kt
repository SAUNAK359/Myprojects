package com.saunak.director

import androidx.camera.core.CameraSelector
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
        lensFacing = CameraSelector.LENS_FACING_BACK,
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

    @Test fun extensionRejectsTorchConflict() {
        val result = ToolValidator().validate(listOf(ToolCall(ToolIds.EXTENSION, stringValue = "hdr"), ToolCall(ToolIds.TORCH, value = 1.0)), caps)
        assertTrue(result.accepted.any { it.id == ToolIds.EXTENSION })
        assertTrue(result.rejected.any { it.call.id == ToolIds.TORCH && it.reason == "extension_conflict" })
    }

    @Test fun rejectsUnknownToolAndSupersedesOldLook() {
        val result = ToolValidator().validate(listOf(ToolCall(999), ToolCall(ToolIds.LOOK_WARM, value = .5), ToolCall(ToolIds.LOOK_COOL, value = .4)), caps)
        assertTrue(result.rejected.any { it.call.id == 999 })
        assertEquals(ToolIds.LOOK_COOL, result.accepted.single { it.id in ToolIds.LOOK_IDS }.id)
    }
}
