package ai.auraforge.director.ml

import ai.auraforge.director.body.BodyCalibrator
import ai.auraforge.director.core.SceneState
import ai.auraforge.director.quality.ImageQualityAnalyzer
import ai.auraforge.director.sensors.SensorFusion
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max

class FrameAnalyzer(
    private val vision: VisionPipeline,
    private val qualityAnalyzer: ImageQualityAnalyzer,
    private val sensors: SensorFusion,
    private val onScene: (SceneState) -> Unit
) : ImageAnalysis.Analyzer {

    private val processing = AtomicBoolean(false)
    private val bodyCalibrator = BodyCalibrator()
    private var lastAnalyzedAt = 0L
    private var lastSnapshotAt = 0L
    private var cachedSnapshot: ByteArray? = null

    @Volatile var targetIntervalMs: Long = 55L

    override fun analyze(image: ImageProxy) {
        val now = System.nanoTime() / 1_000_000L
        if (!processing.compareAndSet(false, true) || now - lastAnalyzedAt < targetIntervalMs) {
            image.close()
            return
        }
        lastAnalyzedAt = now

        try {
            val raw = image.toBitmap()
            val rotated = rotate(raw, image.imageInfo.rotationDegrees)
            if (rotated !== raw) raw.recycle()

            val quality = qualityAnalyzer.analyze(rotated, sensors.motionScore, sensors.rollDeg)
            val snapshot = vision.analyze(rotated, now)
            val profiles = snapshot.poses.associate { pose -> pose.id to bodyCalibrator.update(pose) }

            if (now - lastSnapshotAt >= SCENE_SNAPSHOT_PERIOD_MS) {
                lastSnapshotAt = now
                cachedSnapshot = encodePlanningSnapshot(rotated)
            }

            onScene(
                SceneState(
                    timestampMs = now,
                    poses = snapshot.poses,
                    entities = snapshot.entities,
                    face = snapshot.face,
                    quality = quality,
                    imageWidth = rotated.width,
                    imageHeight = rotated.height,
                    bodyProfiles = profiles,
                    analysisFrameJpeg = cachedSnapshot
                )
            )
            rotated.recycle()
        } catch (_: Throwable) {
            // Camera preview must never fail because analysis failed.
        } finally {
            image.close()
            processing.set(false)
        }
    }

    fun resetSubjectCalibration() {
        bodyCalibrator.reset()
    }

    private fun encodePlanningSnapshot(bitmap: Bitmap): ByteArray? = runCatching {
        val maxSide = max(bitmap.width, bitmap.height)
        val scaled = if (maxSide > PLANNING_MAX_SIDE) {
            val scale = PLANNING_MAX_SIDE.toFloat() / maxSide
            Bitmap.createScaledBitmap(
                bitmap,
                (bitmap.width * scale).toInt().coerceAtLeast(1),
                (bitmap.height * scale).toInt().coerceAtLeast(1),
                true
            )
        } else bitmap

        val out = ByteArrayOutputStream()
        scaled.compress(Bitmap.CompressFormat.JPEG, 76, out)
        if (scaled !== bitmap) scaled.recycle()
        out.toByteArray()
    }.getOrNull()

    private fun rotate(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees % 360 == 0) return bitmap
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    companion object {
        private const val SCENE_SNAPSHOT_PERIOD_MS = 900L
        private const val PLANNING_MAX_SIDE = 512
    }
}
