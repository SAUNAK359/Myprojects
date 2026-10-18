package ai.auraforge.director.ml

import ai.auraforge.director.core.EntityState
import ai.auraforge.director.core.FaceState
import ai.auraforge.director.core.Geometry
import ai.auraforge.director.core.Point2
import ai.auraforge.director.core.PoseState
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.io.Closeable
import kotlin.math.max

class VisionPipeline(
    private val context: Context,
    private val preferGpu: Boolean = true
) : Closeable {

    data class Snapshot(
        val poses: List<PoseState>,
        val entities: List<EntityState>,
        val face: FaceState?
    )

    private var pose: PoseLandmarker? = null
    private var detector: ObjectDetector? = null
    private var face: FaceLandmarker? = null

    private var lastObjects: List<EntityState> = emptyList()
    private var lastFace: FaceState? = null
    private var lastObjectAt = 0L
    private var lastFaceAt = 0L

    @Volatile var ready: Boolean = false
        private set
    @Volatile var error: String? = null
        private set

    fun initialize() {
        try {
            pose = createPose(preferGpu) ?: createPose(false)
            detector = createDetector(preferGpu) ?: createDetector(false)
            face = createFace(preferGpu) ?: createFace(false)
            ready = pose != null
            if (!ready) error = "Pose model unavailable"
        } catch (t: Throwable) {
            error = t.message ?: t.javaClass.simpleName
            Log.e(TAG, "Vision initialization failed", t)
        }
    }

    fun analyze(bitmap: Bitmap, timestampMs: Long): Snapshot {
        if (!ready) return Snapshot(emptyList(), lastObjects, lastFace)
        val mpImage = BitmapImageBuilder(bitmap).build()

        val poseStates = try {
            pose?.detect(mpImage)?.landmarks()?.mapIndexed { index, landmarks ->
                val points = landmarks.map { lm ->
                    Point2(
                        x = lm.x(),
                        y = lm.y(),
                        confidence = max(lm.visibility().orElse(0.75f), lm.presence().orElse(0.75f))
                    )
                }
                PoseState(index, points, Geometry.bbox(points), points.map { it.confidence }.average().toFloat())
            } ?: emptyList()
        } catch (t: Throwable) {
            Log.w(TAG, "Pose inference failed", t)
            emptyList()
        }

        if (timestampMs - lastObjectAt >= OBJECT_PERIOD_MS) {
            lastObjectAt = timestampMs
            lastObjects = try {
                detector?.detect(mpImage)?.detections()?.mapIndexedNotNull { index, d ->
                    val c = d.categories().maxByOrNull { it.score() } ?: return@mapIndexedNotNull null
                    val b = d.boundingBox()
                    EntityState(
                        id = stableObjectId(c.categoryName(), b.centerX(), b.centerY(), index),
                        label = c.categoryName().lowercase(),
                        score = c.score(),
                        bbox = RectF(
                            (b.left / bitmap.width).coerceIn(0f, 1f),
                            (b.top / bitmap.height).coerceIn(0f, 1f),
                            (b.right / bitmap.width).coerceIn(0f, 1f),
                            (b.bottom / bitmap.height).coerceIn(0f, 1f)
                        )
                    )
                } ?: emptyList()
            } catch (t: Throwable) {
                Log.w(TAG, "Object inference failed", t)
                lastObjects
            }
        }

        if (timestampMs - lastFaceAt >= FACE_PERIOD_MS) {
            lastFaceAt = timestampMs
            lastFace = try {
                val result = face?.detect(mpImage)
                val shapes = result?.faceBlendshapes()?.orElse(null)?.firstOrNull()
                if (shapes != null) {
                    fun score(name: String): Float = shapes.firstOrNull { it.categoryName() == name }?.score() ?: 0f
                    FaceState(
                        blinkLeft = score("eyeBlinkLeft"),
                        blinkRight = score("eyeBlinkRight"),
                        smile = max(score("mouthSmileLeft"), score("mouthSmileRight")),
                        confidence = 1f
                    )
                } else null
            } catch (t: Throwable) {
                Log.w(TAG, "Face inference failed", t)
                lastFace
            }
        }

        mpImage.close()
        return Snapshot(poseStates, lastObjects, lastFace)
    }

    private fun createPose(gpu: Boolean): PoseLandmarker? = runCatching {
        val base = baseOptions("pose_landmarker_lite.task", gpu)
        val options = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.IMAGE)
            .setNumPoses(4)
            .setMinPoseDetectionConfidence(0.45f)
            .setMinPosePresenceConfidence(0.45f)
            .setMinTrackingConfidence(0.45f)
            .build()
        PoseLandmarker.createFromOptions(context, options)
    }.onFailure { Log.w(TAG, "Pose ${if (gpu) "GPU" else "CPU"} init failed", it) }.getOrNull()

    private fun createDetector(gpu: Boolean): ObjectDetector? = runCatching {
        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions("efficientdet_lite0.tflite", gpu))
            .setRunningMode(RunningMode.IMAGE)
            .setScoreThreshold(0.32f)
            .setMaxResults(16)
            .build()
        ObjectDetector.createFromOptions(context, options)
    }.onFailure { Log.w(TAG, "Detector ${if (gpu) "GPU" else "CPU"} init failed", it) }.getOrNull()

    private fun createFace(gpu: Boolean): FaceLandmarker? = runCatching {
        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(baseOptions("face_landmarker.task", gpu))
            .setRunningMode(RunningMode.IMAGE)
            .setNumFaces(4)
            .setOutputFaceBlendshapes(true)
            .setMinFaceDetectionConfidence(0.45f)
            .setMinFacePresenceConfidence(0.45f)
            .setMinTrackingConfidence(0.45f)
            .build()
        FaceLandmarker.createFromOptions(context, options)
    }.onFailure { Log.w(TAG, "Face ${if (gpu) "GPU" else "CPU"} init failed", it) }.getOrNull()

    private fun baseOptions(asset: String, gpu: Boolean): BaseOptions {
        val b = BaseOptions.builder().setModelAssetPath(asset)
        if (gpu) b.setDelegate(Delegate.GPU) else b.setDelegate(Delegate.CPU)
        return b.build()
    }

    private fun stableObjectId(label: String, x: Float, y: Float, ordinal: Int): Long {
        val qx = (x / 32f).toInt()
        val qy = (y / 32f).toInt()
        return (label.hashCode().toLong() shl 32) xor (qx.toLong() shl 18) xor (qy.toLong() shl 4) xor ordinal.toLong()
    }

    override fun close() {
        runCatching { pose?.close() }
        runCatching { detector?.close() }
        runCatching { face?.close() }
        pose = null
        detector = null
        face = null
        ready = false
    }

    companion object {
        private const val TAG = "DirectorVision"
        private const val OBJECT_PERIOD_MS = 220L
        private const val FACE_PERIOD_MS = 100L
    }
}
