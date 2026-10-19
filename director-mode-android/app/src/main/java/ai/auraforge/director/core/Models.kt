package ai.auraforge.director.core

import ai.auraforge.director.body.BodyProfile
import android.graphics.RectF
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class Point2(val x: Float, val y: Float, val confidence: Float = 1f) {
    fun distance(other: Point2): Float {
        val dx = x - other.x
        val dy = y - other.y
        return sqrt(dx * dx + dy * dy)
    }
}

data class PoseState(
    val id: Int,
    val landmarks: List<Point2>,
    val bbox: RectF,
    val confidence: Float
)

data class EntityState(
    val id: Long,
    val label: String,
    val score: Float,
    val bbox: RectF
) {
    val cx: Float get() = bbox.centerX()
    val cy: Float get() = bbox.centerY()
}

data class FaceState(
    val blinkLeft: Float = 0f,
    val blinkRight: Float = 0f,
    val smile: Float = 0f,
    val confidence: Float = 0f
) {
    val eyesOpenScore: Float get() = (1f - max(blinkLeft, blinkRight)).coerceIn(0f, 1f)
}

/** Lightweight low-resolution luminance field used for lighting/background decisions. */
data class LuminanceGrid(
    val columns: Int = 0,
    val rows: Int = 0,
    val values: List<Float> = emptyList()
) {
    fun mean(rect: RectF): Float? {
        if (columns <= 0 || rows <= 0 || values.size != columns * rows) return null
        val l = (rect.left.coerceIn(0f, 1f) * columns).toInt().coerceIn(0, columns - 1)
        val r = (rect.right.coerceIn(0f, 1f) * columns).toInt().coerceIn(l, columns - 1)
        val t = (rect.top.coerceIn(0f, 1f) * rows).toInt().coerceIn(0, rows - 1)
        val b = (rect.bottom.coerceIn(0f, 1f) * rows).toInt().coerceIn(t, rows - 1)
        var sum = 0f
        var n = 0
        for (y in t..b) for (x in l..r) {
            sum += values[y * columns + x]
            n++
        }
        return if (n == 0) null else sum / n
    }

    fun meanAll(): Float? = if (values.isEmpty()) null else values.average().toFloat()
}

data class ImageQuality(
    val meanLuma: Float = 0.5f,
    val contrast: Float = 0.5f,
    val highlightClipping: Float = 0f,
    val shadowClipping: Float = 0f,
    val sharpness: Float = 0.5f,
    val cameraMotion: Float = 0f,
    val deviceRollDeg: Float = 0f,
    val lumaGrid: LuminanceGrid = LuminanceGrid()
) {
    val exposureScore: Float
        get() {
            val midpoint = 1f - kotlin.math.abs(meanLuma - 0.5f) * 1.45f
            val clippingPenalty = (highlightClipping + shadowClipping) * 1.5f
            return (midpoint - clippingPenalty).coerceIn(0f, 1f)
        }

    val stabilityScore: Float get() = (1f - cameraMotion).coerceIn(0f, 1f)
    val technicalScore: Float
        get() = (0.36f * sharpness + 0.30f * exposureScore + 0.20f * contrast + 0.14f * stabilityScore)
            .coerceIn(0f, 1f)
}

data class SceneState(
    val timestampMs: Long,
    val poses: List<PoseState>,
    val entities: List<EntityState>,
    val face: FaceState?,
    val quality: ImageQuality,
    val imageWidth: Int,
    val imageHeight: Int,
    val bodyProfiles: Map<Int, BodyProfile> = emptyMap(),
    /** Small JPEG snapshot used only for occasional on-device VLM planning. */
    val analysisFrameJpeg: ByteArray? = null
) {
    val personCount: Int get() = max(poses.size, entities.count { it.label.equals("person", true) })
    fun has(label: String): Boolean = entities.any { it.label.equals(label, true) }
    fun primary(label: String): EntityState? = entities.filter { it.label.equals(label, true) }.maxByOrNull { it.score * it.bbox.width() * it.bbox.height() }
    fun primaryBodyProfile(): BodyProfile? = poses.firstOrNull()?.let { bodyProfiles[it.id] }
}

enum class DirectorLockState {
    SEARCHING,
    SHOT_SELECTED,
    ALIGNING,
    NEAR_LOCK,
    LOCK_CANDIDATE,
    LOCKED,
    CAPTURING
}

data class TargetPose(
    val personIndex: Int,
    val joints: Map<Int, Point2>,
    val label: String
)

data class TargetRelation(
    val objectLabel: String,
    val desiredSide: Int,
    val desiredDistanceNorm: Float,
    val weight: Float
)

data class DirectorPlan(
    val templateId: String,
    val displayName: String,
    val targetPoses: List<TargetPose>,
    val relation: TargetRelation?,
    val confidence: Float,
    val expectedGain: Float,
    val style: String
)

data class AlignmentState(
    val poseScore: Float = 0f,
    val framingScore: Float = 0f,
    val relationScore: Float = 1f,
    val headScore: Float = 1f,
    val compositionScore: Float = 0f,
    val totalScore: Float = 0f,
    val technicalScore: Float = 0f,
    val momentScore: Float = 0f,
    val state: DirectorLockState = DirectorLockState.SEARCHING,
    val stableMs: Long = 0L
)

data class DirectorFrameState(
    val scene: SceneState? = null,
    val plan: DirectorPlan? = null,
    val alignment: AlignmentState = AlignmentState(),
    val gridEnabled: Boolean = false,
    val message: String = "SCANNING SCENE",
    val hint: String = "Move naturally — the shot will adapt"
)

fun RectF.clamp01(): RectF = RectF(
    min(1f, max(0f, left)),
    min(1f, max(0f, top)),
    min(1f, max(0f, right)),
    min(1f, max(0f, bottom))
)
