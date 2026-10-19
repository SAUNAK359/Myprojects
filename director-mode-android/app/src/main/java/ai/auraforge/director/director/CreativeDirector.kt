package ai.auraforge.director.director

import ai.auraforge.director.core.SceneState
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
import java.util.concurrent.atomic.AtomicBoolean

/**
 * A creative VLM proposal. It deliberately contains only bounded, high-level controls.
 * Exact skeleton geometry and camera commands remain deterministic.
 */
data class CreativeBlueprint(
    val recipeId: String,
    val confidence: Float,
    val subjectAnchorX: Float? = null,
    val subjectAnchorY: Float? = null,
    val mirrorPreferred: Boolean? = null,
    val subjectGuidance: String? = null,
    val cameraGuidance: String? = null,
    val lightingGuidance: String? = null,
    val backgroundGuidance: String? = null,
    val rationale: String? = null,
    val createdAtMs: Long = 0L
)

data class BlueprintValidation(
    val accepted: Boolean,
    val reason: String? = null,
    val blueprint: CreativeBlueprint? = null
)

class BlueprintValidator {
    fun validate(
        blueprint: CreativeBlueprint?,
        scene: SceneState,
        candidates: List<ShotTemplate>
    ): BlueprintValidation {
        if (blueprint == null) return BlueprintValidation(false, "missing blueprint")
        val recipe = candidates.firstOrNull { it.id == blueprint.recipeId }
            ?: return BlueprintValidation(false, "recipe not in feasible candidate set")
        if (blueprint.confidence !in 0f..1f) return BlueprintValidation(false, "invalid confidence")

        val profile = scene.primaryBodyProfile()
        if (profile != null && profile.crop !in recipe.supportedCrops) {
            return BlueprintValidation(false, "recipe incompatible with observed crop")
        }

        recipe.relation?.let { relation ->
            if (scene.primary(relation.objectLabel) == null) {
                return BlueprintValidation(false, "required relation object is not grounded")
            }
        }

        val safe = blueprint.copy(
            confidence = blueprint.confidence.coerceIn(0f, 1f),
            subjectAnchorX = blueprint.subjectAnchorX?.takeIf { it.isFinite() }?.coerceIn(0.14f, 0.86f),
            subjectAnchorY = blueprint.subjectAnchorY?.takeIf { it.isFinite() }?.coerceIn(0.14f, 0.86f),
            subjectGuidance = blueprint.subjectGuidance.cleanInstruction(),
            cameraGuidance = blueprint.cameraGuidance.cleanInstruction(),
            lightingGuidance = blueprint.lightingGuidance.cleanInstruction(),
            backgroundGuidance = blueprint.backgroundGuidance.cleanInstruction(),
            rationale = blueprint.rationale.cleanInstruction(180)
        )
        return BlueprintValidation(true, blueprint = safe)
    }

    private fun String?.cleanInstruction(max: Int = 96): String? {
        val s = this?.trim()?.replace(Regex("\\s+"), " ") ?: return null
        if (s.isBlank()) return null
        return s.take(max)
    }
}

/**
 * Real fully-offline LiteRT-LM creative director. It never downloads at runtime: the model must be
 * present in APK assets. GPU initialization failure simply marks this component unavailable.
 */
class OfflineVlmDirector(private val context: Context) : Closeable {
    @Volatile var ready: Boolean = false
        private set
    @Volatile var error: String? = null
        private set

    private var engine: Engine? = null
    private val initialized = AtomicBoolean(false)

    fun initialize() {
        if (!initialized.compareAndSet(false, true)) return
        runCatching {
            val modelFile = materializeModelAsset()
            val config = EngineConfig(
                modelPath = modelFile.absolutePath,
                backend = Backend.GPU(),
                visionBackend = Backend.GPU(),
                cacheDir = context.cacheDir.absolutePath
            )
            engine = Engine(config).also { it.initialize() }
            ready = true
            error = null
        }.onFailure {
            error = it.message ?: it.javaClass.simpleName
            ready = false
            runCatching { engine?.close() }
            engine = null
            Log.w(TAG, "Offline VLM unavailable; deterministic Director remains active", it)
        }
    }

    /** Blocking by design; MainActivity calls this only on the dedicated VLM executor. */
    fun propose(scene: SceneState, candidates: List<ShotTemplate>): CreativeBlueprint? {
        if (!ready || candidates.isEmpty()) return null
        val jpeg = scene.analysisFrameJpeg ?: return null
        val e = engine ?: return null
        val prompt = buildPrompt(scene, candidates)

        return runCatching {
            e.createConversation().use { conversation ->
                val response = conversation.sendMessage(
                    Contents.of(
                        listOf(
                            Content.ImageBytes(jpeg),
                            Content.Text(prompt)
                        )
                    )
                )
                parseBlueprint(response.text, scene.timestampMs)
            }
        }.onFailure {
            error = it.message ?: it.javaClass.simpleName
            Log.w(TAG, "VLM planning turn failed", it)
        }.getOrNull()
    }

    private fun buildPrompt(scene: SceneState, candidates: List<ShotTemplate>): String {
        val pose = scene.poses.firstOrNull()
        val profile = scene.primaryBodyProfile()
        val candidateJson = candidates.joinToString(separator = ",") { t ->
            "{\"id\":\"${escape(t.id)}\",\"name\":\"${escape(t.name)}\",\"style\":\"${escape(t.style)}\",\"difficulty\":${"%.2f".format(java.util.Locale.US, t.difficulty)}}"
        }
        val objects = scene.entities
            .filter { it.score >= 0.35f }
            .sortedByDescending { it.score }
            .take(10)
            .joinToString(",") { "${it.label}:${"%.2f".format(java.util.Locale.US, it.score)}" }

        return """
You are Director, an expert professional photographer running fully on-device.
Choose exactly ONE recipe from the supplied feasible candidates. The deterministic system has already removed physically impossible recipes. Analyze the actual image for photographic meaning: background, light, negative space, subject placement, mood and distracting objects. Do not invent an object that is not visible.

Measured evidence:
- people=${scene.personCount}
- crop=${profile?.crop?.name ?: "UNKNOWN"}
- bodyCalibration=${"%.2f".format(java.util.Locale.US, profile?.confidence ?: 0f)}
- subjectBox=${pose?.bbox?.let { "${fmt(it.left)},${fmt(it.top)},${fmt(it.right)},${fmt(it.bottom)}" } ?: "unknown"}
- exposure=${fmt(scene.quality.exposureScore)} contrast=${fmt(scene.quality.contrast)} sharpness=${fmt(scene.quality.sharpness)} motion=${fmt(scene.quality.cameraMotion)}
- highlightClip=${fmt(scene.quality.highlightClipping)} shadowClip=${fmt(scene.quality.shadowClipping)}
- detectedObjects=[$objects]
- candidates=[$candidateJson]

Return ONLY one compact JSON object with exactly these keys:
{
  "recipeId":"candidate id",
  "confidence":0.0,
  "subjectAnchorX":0.50,
  "subjectAnchorY":0.50,
  "mirrorPreferred":false,
  "subjectGuidance":"short instruction or empty",
  "cameraGuidance":"short instruction or empty",
  "lightingGuidance":"short instruction or empty",
  "backgroundGuidance":"short instruction or empty",
  "rationale":"very short photographic reason"
}
Anchor values must stay between 0.14 and 0.86. Guidance must describe achievable changes, never exact ISO/shutter values. Do not ask for body parts outside the visible crop.
""".trimIndent()
    }

    private fun parseBlueprint(raw: String, timestampMs: Long): CreativeBlueprint? {
        val start = raw.indexOf('{')
        val end = raw.lastIndexOf('}')
        if (start < 0 || end <= start) return null
        val o = JSONObject(raw.substring(start, end + 1))
        val id = o.optString("recipeId").trim()
        if (id.isBlank()) return null
        return CreativeBlueprint(
            recipeId = id,
            confidence = o.optDouble("confidence", 0.5).toFloat(),
            subjectAnchorX = o.optDoubleOrNull("subjectAnchorX"),
            subjectAnchorY = o.optDoubleOrNull("subjectAnchorY"),
            mirrorPreferred = if (o.has("mirrorPreferred")) o.optBoolean("mirrorPreferred") else null,
            subjectGuidance = o.optString("subjectGuidance").nullIfBlank(),
            cameraGuidance = o.optString("cameraGuidance").nullIfBlank(),
            lightingGuidance = o.optString("lightingGuidance").nullIfBlank(),
            backgroundGuidance = o.optString("backgroundGuidance").nullIfBlank(),
            rationale = o.optString("rationale").nullIfBlank(),
            createdAtMs = timestampMs
        )
    }

    private fun materializeModelAsset(): File {
        val target = File(context.filesDir, MODEL_ASSET)
        if (target.exists() && target.length() >= MIN_MODEL_BYTES) return target
        val tmp = File(context.filesDir, "$MODEL_ASSET.tmp")
        context.assets.open(MODEL_ASSET).use { input ->
            FileOutputStream(tmp).use { output -> input.copyTo(output, 1024 * 1024) }
        }
        require(tmp.length() >= MIN_MODEL_BYTES) { "Bundled VLM is incomplete" }
        if (target.exists()) target.delete()
        require(tmp.renameTo(target)) { "Unable to materialize bundled VLM" }
        return target
    }

    private fun fmt(v: Float): String = "%.3f".format(java.util.Locale.US, v)
    private fun escape(s: String): String = s.replace("\\", "\\\\").replace("\"", "\\\"")

    override fun close() {
        ready = false
        runCatching { engine?.close() }
        engine = null
    }

    companion object {
        private const val TAG = "DirectorVLM"
        private const val MODEL_ASSET = "SmolVLM2-500M.litertlm"
        private const val MIN_MODEL_BYTES = 300_000_000L
    }
}

private fun JSONObject.optDoubleOrNull(key: String): Float? =
    if (!has(key) || isNull(key)) null else optDouble(key, Double.NaN).takeIf { it.isFinite() }?.toFloat()

private fun String.nullIfBlank(): String? = trim().takeIf { it.isNotEmpty() }
