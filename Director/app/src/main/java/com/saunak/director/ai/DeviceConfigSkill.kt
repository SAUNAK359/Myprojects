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

/** Generates the machine-readable capability snapshot and Gemini-readable device foundation. */
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
            appendLine(); appendLine(inventory)
            appendLine("## Active Director camera session")
            appendLine("lensFacing=${capabilities.lensFacing}")
            appendLine("current=${snapshot.compactDescription()}")
            appendLine(); appendLine("### Exact supported runtime ranges")
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
            appendLine(); appendLine("## Allowed Director actions")
            appendLine(ToolCatalog.contract(capabilities, snapshot.extension))
            appendLine(); appendLine(renderExecutorContract())
            appendLine(); appendLine("## Hard rules")
            appendLine("- Sierra/Gemini is photographic and final-render authority when AI mode succeeds.")
            appendLine("- Beta may clamp physical/API bounds but must not creatively replace a valid Sierra decision.")
            appendLine("- If final AI Render Spec is unavailable after capture, preserve evidence rather than inventing a replacement style.")
            appendLine("- Output machine JSON only; no prose.")
        }
        runCatching { file.writeText(text) }
        return text
    }

    fun foundationProfile(capabilities: CapabilityRegistry): DeviceFoundationProfile {
        val inventory = inventoryCache ?: scanner.scan().also { inventoryCache = it }
        val payload = buildString {
            appendLine("SIERRA DEVICE FOUNDATION PROFILE")
            appendLine("device=${Build.MANUFACTURER} ${Build.MODEL}")
            appendLine("androidSdk=${Build.VERSION.SDK_INT}")
            appendLine(); appendLine(inventory)
            appendLine("## Director executable camera contract")
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
            appendLine(); appendLine("## Complete Sierra camera/look tool schema")
            appendLine(ToolCatalog.contract(capabilities, DirectorExtension.NONE))
            appendLine(); appendLine(renderExecutorContract())
            appendLine(); appendLine("AUTHORITY CONTRACT")
            appendLine("- Sierra/Gemini is the photographic authority and AI-mode final colorist.")
            appendLine("- Sierra beta may enforce physical/API/pixel-safety feasibility but must not replace a valid creative decision.")
            appendLine("- Device limits change execution strategy, not photographic intent.")
            appendLine("- If a current OEM extension conflicts with a direct Sierra command, exit the extension and execute Sierra through normal CameraX/Camera2.")
            appendLine("- Never request controls outside physical/API capabilities.")
            appendLine("- Return one unambiguous final state per camera control and one final photographic direction.")
            appendLine("- Post-capture final rendering uses one complete Sierra Render Spec; beta executes it deterministically and does not rank alternate recipes.")
            appendLine("- If post-capture AI Render Spec fails or is unavailable, preserve captured/reconstructed evidence rather than substituting local style.")
        }
        return DeviceFoundationProfile.create(payload)
    }

    private fun renderExecutorContract(): String = """
        ## Sierra Render Executor v1
        precision=linear_srgb_argb8888
        workingLongEdge=1024..4096, additionally bounded by runtime heap safety
        deterministicOperations=tone_curve,color_temperature_tint,saturation_vibrance,subject_mask_local_light,relative_depth_subject_separation,highlight_rolloff,shadow_anchor,texture_microcontrast,atmosphere_dehaze_glow_haze,optical_vignette,film_grain,final_output_contrast
        subjectMask=tracked_subject_mask_if_available; never fabricate
        depth=relative_subject_depth_if_available; never fabricate metric depth
        unsupportedOperations=denoise,chroma_denoise,deblur,optical_diffusion,edge_softening,selective_shadow_highlight_hue_shift
        unsupportedSemanticMasks=face,skin,sky unless future device/runtime foundation explicitly adds them
        invariants=preserve identity,geometry,text,material truth,luminous-source hierarchy,real scene evidence
        releaseRule=valid AI Render Spec is executed directly; unsupported requests are rejected rather than silently approximated
    """.trimIndent()

    private fun writeMachineSnapshot(capabilities: CapabilityRegistry, snapshot: CameraSnapshot, inventory: String) {
        val active = JSONObject().apply {
            put("lensFacing", capabilities.lensFacing); put("currentState", snapshot.compactDescription())
            put("zoomMin", capabilities.minZoom); put("zoomMax", capabilities.maxZoom)
            put("exposureEvMin", capabilities.minExposureEv); put("exposureEvMax", capabilities.maxExposureEv); put("exposureEvStep", capabilities.exposureStepEv)
            put("isoMin", capabilities.isoMin ?: JSONObject.NULL); put("isoMax", capabilities.isoMax ?: JSONObject.NULL)
            put("exposureTimeMinNs", capabilities.exposureTimeMinNs ?: JSONObject.NULL); put("exposureTimeMaxNs", capabilities.exposureTimeMaxNs ?: JSONObject.NULL)
            put("minimumFocusDistance", capabilities.minimumFocusDistance ?: JSONObject.NULL)
            put("manualSensor", capabilities.manualSensorSupported); put("raw", capabilities.rawSupported)
            put("flash", capabilities.hasFlash); put("torch", capabilities.torchSupported); put("focusMetering", capabilities.focusMeteringSupported)
            put("focalLengthsMm", JSONArray(capabilities.focalLengthsMm)); put("opticalStabilizationModes", JSONArray(capabilities.opticalStabilizationModes))
            put("videoStabilizationModes", JSONArray(capabilities.videoStabilizationModes)); put("extensions", JSONArray(capabilities.extensions.map { it.wireName }))
            put("allowedToolIds", JSONArray(capabilities.allowedToolIds.sorted())); put("activeExtension", capabilities.activeExtension.wireName)
        }
        val root = JSONObject().apply {
            put("schemaVersion", 1); put("manufacturer", Build.MANUFACTURER); put("model", Build.MODEL); put("androidSdk", Build.VERSION.SDK_INT)
            put("generatedAtMs", System.currentTimeMillis()); put("activeDirectorSession", active); put("phoneCameraInventory", inventory)
            put("renderExecutor", JSONObject().apply {
                put("schemaVersion", 1); put("precision", "linear_srgb_argb8888"); put("workingLongEdgeMin", 1024); put("workingLongEdgeMax", 4096)
                put("subjectMask", "tracked_if_available"); put("relativeDepth", "tracked_if_available")
                put("semanticMasksUnavailable", JSONArray(listOf("face","skin","sky")))
            })
        }
        runCatching { machineFile.writeText(root.toString(2)) }
    }

    fun readOrGenerate(capabilities: CapabilityRegistry, snapshot: CameraSnapshot): String = refresh(capabilities, snapshot)
    fun filePath(): String = file.absolutePath
    fun machineFilePath(): String = machineFile.absolutePath
}
