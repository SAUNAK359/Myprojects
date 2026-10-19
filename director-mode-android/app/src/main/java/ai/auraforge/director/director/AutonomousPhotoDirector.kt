package ai.auraforge.director.director

import ai.auraforge.director.camera.CameraCapabilities
import ai.auraforge.director.core.CaptureProgram
import ai.auraforge.director.core.ColorStyle
import ai.auraforge.director.core.CompositionGuide
import ai.auraforge.director.core.ExtensionPreference
import ai.auraforge.director.core.FlashIntent
import ai.auraforge.director.core.FocusIntent
import ai.auraforge.director.core.HardwareEffect
import ai.auraforge.director.core.OutputIntent
import ai.auraforge.director.core.PhotoPlan
import ai.auraforge.director.core.PlanReadiness
import ai.auraforge.director.core.SceneState
import ai.auraforge.director.core.WhiteBalanceIntent
import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import org.json.JSONObject
import java.io.Closeable
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs

/**
 * Complete on-device photographic planner. The VLM selects creative intent; the camera executor is
 * the authority over actual hardware and only applies capabilities the active lens exposes.
 */
class AutonomousPhotoDirector(private val context: Context) : Closeable {
    @Volatile var ready: Boolean = false
        private set
    @Volatile var error: String? = null
        private set

    private var engine: Engine? = null
    private val initialized = AtomicBoolean(false)

    fun initialize() {
        if (!initialized.compareAndSet(false, true)) return
        runCatching {
            val model = materializeModelAsset()
            engine = Engine(
                EngineConfig(
                    modelPath = model.absolutePath,
                    backend = Backend.GPU(),
                    visionBackend = Backend.GPU(),
                    maxNumTokens = MAX_CONTEXT_TOKENS,
                    cacheDir = context.cacheDir.absolutePath
                )
            ).also { it.initialize() }
            ready = true
            error = null
        }.onFailure {
            error = it.message ?: it.javaClass.simpleName
            ready = false
            runCatching { engine?.close() }
            engine = null
            Log.w(TAG, "Autonomous VLM unavailable; camera remains operational", it)
        }
    }

    fun propose(scene: SceneState, caps: CameraCapabilities, extensionNames: Set<String> = emptySet()): PhotoPlan? {
        if (!ready) return null
        val jpeg = scene.analysisFrameJpeg ?: return null
        val e = engine ?: return null
        val prompt = buildPrompt(scene, caps, extensionNames)
        return runCatching {
            e.createConversation().use { conversation ->
                val response = conversation.sendMessage(
                    Contents.of(listOf(Content.Text(prompt), Content.ImageBytes(jpeg))),
                    maxOutputToken = MAX_OUTPUT_TOKENS
                )
                parse(response.toString(), scene.timestampMs)
            }
        }.onFailure {
            error = it.message ?: it.javaClass.simpleName
            Log.w(TAG, "Autonomous photography inference failed", it)
        }.getOrNull()
    }

    private fun buildPrompt(scene: SceneState, caps: CameraCapabilities, extensions: Set<String>): String {
        val pose = scene.poses.firstOrNull()
        val profile = scene.primaryBodyProfile()
        val objects = scene.entities
            .filter { it.score >= 0.35f }
            .sortedByDescending { it.score }
            .take(12)
            .joinToString(",") { "${it.label}:${fmt(it.score)}" }
        val faceBox = pose?.bbox
        val apertureText = if (caps.apertures.isEmpty()) "none" else caps.apertures.joinToString(",") { fmt(it) }
        val focalText = if (caps.focalLengthsMm.isEmpty()) "unknown" else caps.focalLengthsMm.joinToString(",") { fmt(it) }
        val extensionText = if (extensions.isEmpty()) "none-known" else extensions.sorted().joinToString(",")

        return """
You are DIRECTOR, an autonomous professional cinematography and still-photography camera brain running fully on-device. Your job is not to suggest a generic pose. You must DESIGN THE ENTIRE SHOT and choose a coherent photographic treatment from what is actually visible.

Think like an expert photographer before capture. Evaluate subject priority, story/theme, facial presentation, background separation, depth layers, perspective distortion, negative space, leading lines, symmetry, horizon, crop, highlights, shadow detail, mixed light, key-light direction, dynamic range, motion, likely shutter risk, color palette, skin rendition, focal-length look, whether portrait/bokeh/HDR/night is justified, whether a special color treatment improves the story, and whether burst capture is useful.

Measured scene evidence (treat these as facts):
people=${scene.personCount}
crop=${profile?.crop?.name ?: "UNKNOWN"}
subjectBox=${faceBox?.let { "${fmt(it.left)},${fmt(it.top)},${fmt(it.right)},${fmt(it.bottom)}" } ?: "none"}
objects=[$objects]
meanLuma=${fmt(scene.quality.meanLuma)}
contrast=${fmt(scene.quality.contrast)}
highlightClip=${fmt(scene.quality.highlightClipping)}
shadowClip=${fmt(scene.quality.shadowClipping)}
sharpness=${fmt(scene.quality.sharpness)}
motion=${fmt(scene.quality.cameraMotion)}
rollDeg=${fmt(scene.quality.deviceRollDeg)}

Actual camera capability evidence:
manualSensor=${caps.supportsManualSensor}
raw=${caps.supportsRaw}
flash=${caps.flashAvailable}
variableAperture=${caps.supportsVariableAperture}
apertures=[$apertureText]
focalLengthsMm=[$focalText]
maxDigitalZoom=${fmt(caps.maxDigitalZoom)}
manualFocus=${caps.supportsManualFocus}
oemExtensions=[$extensionText]

Return ONLY a compact JSON object with these exact keys. Use enum strings exactly as shown:
{
 "theme":"2-5 word photographic theme",
 "confidence":0.0,
 "program":"STANDARD|PORTRAIT|NIGHT|HDR|ACTION|LANDSCAPE|MACRO",
 "grid":"NONE|THIRDS|GOLDEN|CENTER|SYMMETRY|DIAGONAL",
 "extension":"NONE|AUTO|BOKEH|HDR|NIGHT|FACE_RETOUCH",
 "focus":"AUTO|FACE|SUBJECT|CENTER|MACRO|INFINITY",
 "flash":"OFF|AUTO|ON|SCREEN_FILL",
 "whiteBalance":"AUTO|DAYLIGHT|CLOUDY|SHADE|TUNGSTEN|FLUORESCENT|WARM|COOL",
 "colorStyle":"NATURAL|CINEMATIC_WARM|CINEMATIC_COOL|FILM|VIVID|MUTED|MONO|HIGH_CONTRAST",
 "hardwareEffect":"OFF|MONO|SEPIA|NEGATIVE|AQUA|POSTERIZE",
 "output":"JPEG|ULTRA_HDR|RAW_JPEG",
 "zoomRatio":1.0,
 "exposureEv":0.0,
 "apertureFNumber":null,
 "focusX":0.5,
 "focusY":0.5,
 "contrast":0.0,
 "saturation":0.0,
 "temperature":0.0,
 "vignette":0.0,
 "retouchStrength":0.0,
 "backgroundBlurStrength":0.0,
 "facePriority":false,
 "burstCount":1,
 "subjectGuidance":"one short instruction or empty",
 "cameraGuidance":"one short instruction or empty",
 "rationale":"one concise photographic reason"
}

Rules:
- Do not invent objects or lights.
- Choose a coherent LOOK, not maximum values everywhere.
- zoomRatio must be 0.5..5.0; executor clamps it to hardware.
- exposureEv must be -2.0..2.0.
- contrast/saturation/temperature/vignette/retouchStrength/backgroundBlurStrength are normalized -1..1 except retouch/blur 0..1.
- apertureFNumber must be null unless variable aperture is exposed, then choose one of the available values.
- RAW_JPEG only if raw=true.
- SCREEN_FILL only makes sense on a front-facing portrait; otherwise prefer OFF/AUTO.
- FACE_RETOUCH should be subtle and only for a visible human face.
- BOKEH requires a clear foreground subject and should not be used for landscapes/groups where separation would damage context.
- NIGHT requires truly low light and low enough subject motion.
- HDR is for high dynamic range, not every shot.
- Avoid hardware novelty effects unless they materially improve the artistic concept.
- Prefer physical/lens perspective decisions over excessive digital zoom.
- Do not return PANORAMA or LONG_EXPOSURE; those require dedicated capture pipelines and are not yet autonomous-ready.
""".trimIndent()
    }

    private fun parse(raw: String, timestampMs: Long): PhotoPlan? {
        val start = raw.indexOf('{')
        val end = raw.lastIndexOf('}')
        if (start < 0 || end <= start) return null
        val o = JSONObject(raw.substring(start, end + 1))
        val theme = o.optString("theme").trim().ifBlank { "Autonomous Photo" }
        return PhotoPlan(
            theme = theme.take(64),
            confidence = o.optDouble("confidence", 0.5).toFloat(),
            program = enum(o.optString("program"), CaptureProgram.STANDARD),
            compositionGuide = enum(o.optString("grid"), CompositionGuide.THIRDS),
            extensionPreference = enum(o.optString("extension"), ExtensionPreference.NONE),
            focusIntent = enum(o.optString("focus"), FocusIntent.AUTO),
            flashIntent = enum(o.optString("flash"), FlashIntent.OFF),
            whiteBalanceIntent = enum(o.optString("whiteBalance"), WhiteBalanceIntent.AUTO),
            colorStyle = enum(o.optString("colorStyle"), ColorStyle.NATURAL),
            hardwareEffect = enum(o.optString("hardwareEffect"), HardwareEffect.OFF),
            outputIntent = enum(o.optString("output"), OutputIntent.JPEG),
            zoomRatio = o.optDouble("zoomRatio", 1.0).toFloat(),
            exposureEv = o.optDouble("exposureEv", 0.0).toFloat(),
            apertureFNumber = o.optNullableFloat("apertureFNumber"),
            focusX = o.optNullableFloat("focusX"),
            focusY = o.optNullableFloat("focusY"),
            contrast = o.optDouble("contrast", 0.0).toFloat(),
            saturation = o.optDouble("saturation", 0.0).toFloat(),
            temperature = o.optDouble("temperature", 0.0).toFloat(),
            vignette = o.optDouble("vignette", 0.0).toFloat(),
            retouchStrength = o.optDouble("retouchStrength", 0.0).toFloat(),
            backgroundBlurStrength = o.optDouble("backgroundBlurStrength", 0.0).toFloat(),
            facePriority = o.optBoolean("facePriority", false),
            burstCount = o.optInt("burstCount", 1),
            subjectGuidance = o.optString("subjectGuidance").clean(),
            cameraGuidance = o.optString("cameraGuidance").clean(),
            rationale = o.optString("rationale").clean(180),
            createdAtMs = timestampMs
        )
    }

    private inline fun <reified T : Enum<T>> enum(value: String, fallback: T): T =
        enumValues<T>().firstOrNull { it.name.equals(value.trim(), true) } ?: fallback

    private fun materializeModelAsset(): File {
        val target = File(context.filesDir, MODEL_ASSET)
        if (target.exists() && target.length() >= MIN_MODEL_BYTES) return target
        val tmp = File(context.filesDir, "$MODEL_ASSET.tmp")
        if (tmp.exists()) tmp.delete()
        context.assets.open(MODEL_ASSET).use { input ->
            FileOutputStream(tmp).use { output -> input.copyTo(output, 1024 * 1024) }
        }
        require(tmp.length() >= MIN_MODEL_BYTES) { "Bundled photography VLM is incomplete" }
        if (target.exists()) target.delete()
        require(tmp.renameTo(target)) { "Unable to materialize bundled photography VLM" }
        return target
    }

    private fun fmt(value: Float): String = "%.3f".format(Locale.US, value)

    override fun close() {
        ready = false
        runCatching { engine?.close() }
        engine = null
    }

    companion object {
        private const val TAG = "AutonomousPhotoDirector"
        private const val MODEL_ASSET = "SmolVLM2-500M.litertlm"
        private const val MIN_MODEL_BYTES = 300_000_000L
        private const val MAX_CONTEXT_TOKENS = 2048
        private const val MAX_OUTPUT_TOKENS = 360
    }
}

class PhotoPlanValidator {
    fun validate(plan: PhotoPlan?, scene: SceneState, caps: CameraCapabilities, availableExtensions: Set<String>): PhotoPlan? {
        if (plan == null || !plan.confidence.isFinite()) return null
        val zoomRangeMax = caps.maxDigitalZoom.coerceAtLeast(1f)
        val faceVisible = scene.poses.isNotEmpty() || scene.face != null
        val safeOutput = if (plan.outputIntent == OutputIntent.RAW_JPEG && !caps.supportsRaw) OutputIntent.JPEG else plan.outputIntent
        val safeExtension = if (plan.extensionPreference == ExtensionPreference.NONE || plan.extensionPreference.name in availableExtensions) {
            plan.extensionPreference
        } else ExtensionPreference.NONE
        val aperture = plan.apertureFNumber?.let { requested ->
            caps.apertures.minByOrNull { abs(it - requested) }
        }
        return plan.copy(
            confidence = plan.confidence.coerceIn(0f, 1f),
            extensionPreference = safeExtension,
            outputIntent = safeOutput,
            zoomRatio = plan.zoomRatio.takeIf { it.isFinite() }?.coerceIn(0.5f, minOf(5f, zoomRangeMax)) ?: 1f,
            exposureEv = plan.exposureEv.takeIf { it.isFinite() }?.coerceIn(-2f, 2f) ?: 0f,
            apertureFNumber = aperture,
            focusX = plan.focusX?.takeIf { it.isFinite() }?.coerceIn(0.08f, 0.92f),
            focusY = plan.focusY?.takeIf { it.isFinite() }?.coerceIn(0.08f, 0.92f),
            contrast = plan.contrast.coerceIn(-1f, 1f),
            saturation = plan.saturation.coerceIn(-1f, 1f),
            temperature = plan.temperature.coerceIn(-1f, 1f),
            vignette = plan.vignette.coerceIn(0f, 1f),
            retouchStrength = if (faceVisible) plan.retouchStrength.coerceIn(0f, 0.45f) else 0f,
            backgroundBlurStrength = plan.backgroundBlurStrength.coerceIn(0f, 1f),
            facePriority = plan.facePriority && faceVisible,
            burstCount = plan.burstCount.coerceIn(1, 3),
            subjectGuidance = plan.subjectGuidance?.take(96),
            cameraGuidance = plan.cameraGuidance?.take(96),
            rationale = plan.rationale?.take(180)
        )
    }
}

/** Deterministic final gate: the AI may design the shot, but it cannot claim READY on poor input. */
class PhotoReadinessEvaluator {
    fun evaluate(scene: SceneState, plan: PhotoPlan?): PlanReadiness {
        if (plan == null) return PlanReadiness(false, 0f, "AI designing shot")
        val q = scene.quality
        val subjectScore = if (plan.facePriority || plan.focusIntent == FocusIntent.FACE) {
            if (scene.poses.isNotEmpty() || scene.face != null) 1f else 0.25f
        } else 0.85f
        val exposure = q.exposureScore
        val sharpness = q.sharpness
        val stability = q.stabilityScore
        val clipping = (1f - (q.highlightClipping + q.shadowClipping) * 2.2f).coerceIn(0f, 1f)
        val planConfidence = plan.confidence.coerceIn(0f, 1f)
        val score = (0.23f * planConfidence + 0.22f * exposure + 0.22f * sharpness + 0.18f * stability + 0.10f * clipping + 0.05f * subjectScore)
            .coerceIn(0f, 1f)

        val reason = when {
            planConfidence < 0.55f -> "AI refining treatment"
            q.cameraMotion > 0.45f -> "Hold camera steady"
            q.highlightClipping > 0.16f -> "Protecting highlights"
            q.shadowClipping > 0.28f -> "Recovering shadow detail"
            q.sharpness < 0.42f -> "Waiting for focus"
            subjectScore < 0.5f -> "Acquiring subject"
            else -> "Treatment locked"
        }
        val ready = score >= 0.70f && planConfidence >= 0.58f && q.cameraMotion <= 0.48f && q.sharpness >= 0.40f
        return PlanReadiness(ready, score, reason)
    }
}

private fun JSONObject.optNullableFloat(key: String): Float? =
    if (!has(key) || isNull(key)) null else optDouble(key, Double.NaN).takeIf { it.isFinite() }?.toFloat()

private fun String.clean(max: Int = 96): String? = trim().replace(Regex("\\s+"), " ").takeIf { it.isNotBlank() }?.take(max)
