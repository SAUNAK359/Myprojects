package com.saunak.director.ai

import android.content.Context
import android.os.Build
import com.saunak.director.camera.CameraSnapshot
import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.camera.DirectorExtension
import com.saunak.director.tools.ToolCatalog
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** Generates the machine-readable capability snapshot and Gemini-readable config.md. */
class DeviceConfigSkill(private val context: Context) {
    private val directory = File(context.filesDir, "director")
    private val file = File(directory, "config.md")
    private val machineFile = File(directory, "device_capabilities.json")
    private val scanner = DeviceCapabilityScanner(context)
    @Volatile private var inventoryCache: String? = null

    fun refresh(capabilities: CapabilityRegistry, snapshot: CameraSnapshot): String {
        val inventory = inventoryCache ?: scanner.scan().also { inventoryCache = it }
        directory.mkdirs()
        writeMachineSnapshot(capabilities, snapshot, inventory)

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
            appendLine("- Camera extension mode is mutually exclusive with direct focus/zoom/exposure/torch/flash mutations in one recipe.")
            appendLine("- Prefer small changes over extreme values.")
            appendLine("- Preserve natural skin, highlight detail and subject sharpness unless the scene intent strongly requires otherwise.")
            appendLine("- Output machine JSON only; no prose.")
        }
        runCatching { file.writeText(text) }
        return text
    }

    /** Full capability/tool foundation sent separately from image planning. */
    fun foundationProfile(capabilities: CapabilityRegistry): DeviceFoundationProfile {
        val inventory = inventoryCache ?: scanner.scan().also { inventoryCache = it }
        val payload = buildString {
            appendLine("SIERRA DEVICE FOUNDATION PROFILE")
            appendLine("device=${Build.MANUFACTURER} ${Build.MODEL}")
            appendLine("androidSdk=${Build.VERSION.SDK_INT}")
            appendLine()
            appendLine(inventory)
            appendLine("## Director executable contract for active lens family")
            appendLine("lensFacing=${capabilities.lensFacing}")
            appendLine("zoom=${capabilities.minZoom}..${capabilities.maxZoom}")
            appendLine("exposureEv=${capabilities.minExposureEv}..${capabilities.maxExposureEv}; step=${capabilities.exposureStepEv}")
            appendLine("iso=${capabilities.isoMin ?: "unsupported"}..${capabilities.isoMax ?: "unsupported"}")
            appendLine("exposureTimeNs=${capabilities.exposureTimeMinNs ?: "unsupported"}..${capabilities.exposureTimeMaxNs ?: "unsupported"}")
            appendLine("minimumFocusDistance=${capabilities.minimumFocusDistance ?: "unknown"}")
            appendLine("focalLengthsMm=${capabilities.focalLengthsMm.joinToString()}")
            appendLine("manualSensor=${capabilities.manualSensorSupported}; raw=${capabilities.rawSupported}")
            appendLine("flash=${capabilities.hasFlash}; torch=${capabilities.torchSupported}; focusMetering=${capabilities.focusMeteringSupported}")
            appendLine("extensions=${capabilities.extensions.joinToString { it.wireName }}")
            appendLine("opticalStabilizationModes=${capabilities.opticalStabilizationModes.joinToString()}")
            appendLine("videoStabilizationModes=${capabilities.videoStabilizationModes.joinToString()}")
            appendLine("allowedToolIds=${capabilities.allowedToolIds.sorted().joinToString()}")
            appendLine()
            appendLine("## Complete Sierra camera/look tool schema and device-specific ranges")
            appendLine(ToolCatalog.contract(capabilities, DirectorExtension.NONE))
            appendLine()
            appendLine("AUTHORITY CONTRACT")
            appendLine("- Sierra/Gemini is the photographic authority.")
            appendLine("- Sierra beta may enforce physical/API feasibility but must not replace a valid creative decision.")
            appendLine("- If a current OEM extension conflicts with a direct Sierra command, exit the extension and execute Sierra through the normal camera session.")
            appendLine("- Never request controls outside these physical/API capabilities.")
            appendLine("- Return one unambiguous final state per camera control and one final look; never emit contradictory alternatives.")
        }
        return DeviceFoundationProfile.create(payload)
    }

    private fun writeMachineSnapshot(capabilities: CapabilityRegistry, snapshot: CameraSnapshot, inventory: String) {
        val active = JSONObject().apply {
            put("lensFacing", capabilities.lensFacing)
            put("currentState", snapshot.compactDescription())
            put("zoomMin", capabilities.minZoom)
            put("zoomMax", capabilities.maxZoom)
            put("exposureEvMin", capabilities.minExposureEv)
            put("exposureEvMax", capabilities.maxExposureEv)
            put("exposureEvStep", capabilities.exposureStepEv)
            put("isoMin", capabilities.isoMin ?: JSONObject.NULL)
            put("isoMax", capabilities.isoMax ?: JSONObject.NULL)
            put("exposureTimeMinNs", capabilities.exposureTimeMinNs ?: JSONObject.NULL)
            put("exposureTimeMaxNs", capabilities.exposureTimeMaxNs ?: JSONObject.NULL)
            put("minimumFocusDistance", capabilities.minimumFocusDistance ?: JSONObject.NULL)
            put("manualSensor", capabilities.manualSensorSupported)
            put("raw", capabilities.rawSupported)
            put("flash", capabilities.hasFlash)
            put("torch", capabilities.torchSupported)
            put("focusMetering", capabilities.focusMeteringSupported)
            put("focalLengthsMm", JSONArray(capabilities.focalLengthsMm))
            put("opticalStabilizationModes", JSONArray(capabilities.opticalStabilizationModes))
            put("videoStabilizationModes", JSONArray(capabilities.videoStabilizationModes))
            put("extensions", JSONArray(capabilities.extensions.map { it.wireName }))
            put("allowedToolIds", JSONArray(capabilities.allowedToolIds.sorted()))
            put("activeExtension", capabilities.activeExtension.wireName)
        }
        val root = JSONObject().apply {
            put("schemaVersion", 1)
            put("manufacturer", Build.MANUFACTURER)
            put("model", Build.MODEL)
            put("androidSdk", Build.VERSION.SDK_INT)
            put("generatedAtMs", System.currentTimeMillis())
            put("activeDirectorSession", active)
            put("phoneCameraInventory", inventory)
        }
        runCatching { machineFile.writeText(root.toString(2)) }
    }

    fun readOrGenerate(capabilities: CapabilityRegistry, snapshot: CameraSnapshot): String = refresh(capabilities, snapshot)
    fun filePath(): String = file.absolutePath
    fun machineFilePath(): String = machineFile.absolutePath
}
