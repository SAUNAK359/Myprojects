package ai.auraforge.director.ml

import ai.auraforge.director.core.SceneState
import ai.auraforge.director.quality.ImageQualityAnalyzer
import ai.auraforge.director.sensors.SensorFusion
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.util.concurrent.atomic.AtomicBoolean

class FrameAnalyzer(
    private val vision: VisionPipeline,
    private val qualityAnalyzer: ImageQualityAnalyzer,
    private val sensors: SensorFusion,
    private val onScene: (SceneState) -> Unit
) : ImageAnalysis.Analyzer {

    private val processing = AtomicBoolean(false)
    private var lastAnalyzedAt = 0L
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

            onScene(
                SceneState(
                    timestampMs = now,
                    poses = snapshot.poses,
                    entities = snapshot.entities,
                    face = snapshot.face,
                    quality = quality,
                    imageWidth = rotated.width,
                    imageHeight = rotated.height
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

    private fun rotate(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees % 360 == 0) return bitmap
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

}
