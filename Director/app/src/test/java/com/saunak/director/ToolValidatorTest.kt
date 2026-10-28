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

    @Test fun exposureIsQuantizedToRealDeviceStep() {
        val result = ToolValidator().validate(listOf(ToolCall(ToolIds.EXPOSURE_EV, value = -.62)), caps)
        assertEquals(-.5, result.accepted.single().value!!, .0001)
    }

    @Test fun extensionAndDirectControlsFailClosedAsAmbiguous() {
        val result = ToolValidator().validate(
            listOf(
                ToolCall(ToolIds.ZOOM, value = 2.0),
                ToolCall(ToolIds.EXPOSURE_EV, value = .5),
                ToolCall(ToolIds.TORCH, value = 0.0),
                ToolCall(ToolIds.FLASH, stringValue = "off"),
                ToolCall(ToolIds.EXTENSION, stringValue = "hdr")
            ), caps)
        assertTrue(result.accepted.none { it.id in ToolIds.CAMERA_IDS })
        assertTrue(result.rejected.count { it.reason == "ambiguous_plan" } == 5)
    }

    @Test fun flashAndTorchConflictFailsClosedRatherThanChoosingCreatively() {
        val result = ToolValidator().validate(listOf(ToolCall(ToolIds.TORCH, value = 1.0), ToolCall(ToolIds.FLASH, stringValue = "auto")), caps)
        assertTrue(result.accepted.none { it.id in setOf(ToolIds.TORCH, ToolIds.FLASH) })
        assertEquals(2, result.rejected.count { it.reason == "ambiguous_plan" })
    }

    @Test fun explicitExtensionReleaseAllowsDirectControlPlanToBeValidated() {
        val active = caps.copy(activeExtension = DirectorExtension.HDR)
        val result = ToolValidator().validate(listOf(ToolCall(ToolIds.ZOOM, value = 2.0), ToolCall(ToolIds.EXTENSION, stringValue = "none")), active)
        assertTrue(result.rejected.isEmpty())
        assertTrue(result.accepted.any { it.id == ToolIds.ZOOM && it.value == 2.0 })
        assertTrue(result.accepted.any { it.id == ToolIds.EXTENSION && it.stringValue == "none" })
    }

    @Test fun rejectsUnknownToolAndMultipleLegacyLooksAsAmbiguous() {
        val result = ToolValidator().validate(listOf(ToolCall(999), ToolCall(ToolIds.LOOK_WARM, value = .5), ToolCall(ToolIds.LOOK_COOL, value = .4)), caps)
        assertTrue(result.rejected.any { it.call.id == 999 && it.reason == "unsupported_tool" })
        assertEquals(2, result.rejected.count { it.call.id in ToolIds.LOOK_IDS && it.reason == "ambiguous_plan" })
        assertTrue(result.accepted.none { it.id in ToolIds.LOOK_IDS })
    }
}
