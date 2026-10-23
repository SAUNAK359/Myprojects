package com.saunak.director.ai

import android.content.Context
import android.os.Build
import com.saunak.director.camera.CameraSnapshot
import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.tools.ToolCatalog
import java.io.File

/** Generates the device-specific config.md Gemini consults before every shot plan. */
class DeviceConfigSkill(private val context: Context) {
    private val file = File(context.filesDir, "director/config.md")
    private val scanner = DeviceCapabilityScanner(context)
    @Volatile private var inventoryCache: String? = null

    fun refresh(capabilities: CapabilityRegistry, snapshot: CameraSnapshot): String {
        val inventory = inventoryCache ?: scanner.scan().also { inventoryCache = it }
        val text = buildString {
            appendLine("# Director Device Camera Skill")
            appendLine("device=${Build.MANUFACTURER} ${Build.MODEL}")
            appendLine("android=${Build.VERSION.SDK_INT}")
            appendLine()
            appendLine(inventory)
            appendLine("## Active Director camera session")
            appendLine("lensFacing=${capabilities.lensFacing}")
            appendLine("current=${snapshot.compactDescription()}")
            appendLine()
            appendLine("### Exact supported runtime ranges")
            appendLine("zoom=${capabilities.minZoom}..${capabilities.maxZoom}")
            appendLine("exposureEv=${capabilities.minExposureEv}..${capabilities.maxExposureEv}; step=${capabilities.exposureStepEv}")
            appendLine("iso=${capabilities.isoMin ?: "unsupported"}..${capabilities.isoMax ?: "unsupported"}")
            appendLine("exposureTimeNs=${capabilities.exposureTimeMinNs ?: "unsupported"}..${capabilities.exposureTimeMaxNs ?: "unsupported"}")
            appendLine("minimumFocusDistance=${capabilities.minimumFocusDistance ?: "unknown"}")
            appendLine("focalLengthsMm=${capabilities.focalLengthsMm.joinToString()}")
            appendLine("manualSensor=${capabilities.manualSensorSupported}")
            appendLine("raw=${capabilities.rawSupported}")
            appendLine("flash=${capabilities.hasFlash}; torch=${capabilities.torchSupported}; focusMetering=${capabilities.focusMeteringSupported}")
            appendLine("extensions=${capabilities.extensions.joinToString { it.wireName }}")
            appendLine("opticalStabilizationModes=${capabilities.opticalStabilizationModes.joinToString()}")
            appendLine("videoStabilizationModes=${capabilities.videoStabilizationModes.joinToString()}")
            appendLine()
            appendLine("## Allowed Director actions")
            appendLine(ToolCatalog.contract(capabilities, snapshot.extension))
            appendLine()
            appendLine("## Hard rules")
            appendLine("- Never output unsupported tool IDs.")
            appendLine("- Never exceed any min/max range above.")
            appendLine("- Only the Active Director camera session controls are directly executable in the next payload.")
            appendLine("- Camera extension mode is mutually exclusive with direct focus/zoom/exposure/torch/flash mutations in one recipe.")
            appendLine("- Prefer small changes over extreme values.")
            appendLine("- Preserve natural skin, highlight detail and subject sharpness unless the scene intent strongly requires otherwise.")
            appendLine("- Output machine JSON only; no prose.")
        }
        file.parentFile?.mkdirs()
        runCatching { file.writeText(text) }
        return text
    }

    fun readOrGenerate(capabilities: CapabilityRegistry, snapshot: CameraSnapshot): String =
        refresh(capabilities, snapshot)

    fun filePath(): String = file.absolutePath
}
