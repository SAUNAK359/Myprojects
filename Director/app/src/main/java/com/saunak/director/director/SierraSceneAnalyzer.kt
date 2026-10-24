package com.saunak.director.director

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.media.FaceDetector
import androidx.camera.core.CameraSelector
import com.saunak.director.processing.TechnicalQuality

/** Lightweight semantic guard used before either Sierra AI or Sierra beta plans the shot. */
data class SierraSceneSignals(
    val branch: String,
    val confidence: Double,
    val faceCount: Int = 0,
    val faceCoverage: Double = 0.0,
    val primaryHumanSubject: Boolean = false,
    val brightSky: Boolean = false,
    val urbanStructure: Boolean = false
)

object SierraSceneAnalyzer {
    fun analyze(source: Bitmap, quality: TechnicalQuality, lensFacing: Int): SierraSceneSignals {
        val faces = runCatching { detectFaces(source) }.getOrDefault(FaceSummary(0, 0.0))
        val frontCamera = lensFacing == CameraSelector.LENS_FACING_FRONT
        val primaryHuman = faces.coverage >= 0.035 ||
            (frontCamera && faces.count > 0 && faces.coverage >= 0.012)

        val brightSky = quality.highlightP95 >= 210.0 &&
            quality.topMeanLuma - quality.bottomMeanLuma >= 28.0
        val structured = quality.strongEdgeFraction >= 0.16 &&
            quality.horizontalEdgeEnergy >= 4.5 && quality.verticalEdgeEnergy >= 4.5
        val lowLight = quality.meanLuma < 70.0 && quality.medianLuma < 65.0
        val wideFrame = source.width > source.height * 1.12

        val branch = when {
            lowLight -> "low_light"
            primaryHuman && frontCamera -> "selfie"
            primaryHuman && faces.count > 1 -> "group"
            primaryHuman -> "portrait"
            structured && (brightSky || wideFrame || quality.strongEdgeFraction >= 0.22) -> "architecture"
            brightSky || wideFrame -> "landscape"
            else -> "general"
        }

        val confidence = when (branch) {
            "portrait", "selfie", "group" -> 0.92
            "low_light" -> 0.86
            "architecture" -> if (brightSky) 0.82 else 0.74
            "landscape" -> if (brightSky) 0.78 else 0.66
            else -> 0.48
        }

        return SierraSceneSignals(
            branch = branch,
            confidence = confidence,
            faceCount = faces.count,
            faceCoverage = faces.coverage,
            primaryHumanSubject = primaryHuman,
            brightSky = brightSky,
            urbanStructure = structured
        )
    }

    private data class FaceSummary(val count: Int, val coverage: Double)

    @Suppress("DEPRECATION")
    private fun detectFaces(source: Bitmap): FaceSummary {
        val maxDim = 320
        val scale = minOf(1f, maxDim.toFloat() / maxOf(source.width, source.height).toFloat())
        var width = maxOf(2, (source.width * scale).toInt())
        if (width % 2 != 0) width -= 1
        val height = maxOf(2, (source.height * scale).toInt())
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565)
        val canvas = Canvas(bitmap)
        canvas.drawBitmap(
            source,
            null,
            Rect(0, 0, width, height),
            Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
        )
        return try {
            val faces = arrayOfNulls<FaceDetector.Face>(MAX_FACES)
            val count = FaceDetector(width, height, MAX_FACES).findFaces(bitmap, faces)
            var coverage = 0.0
            for (i in 0 until count) {
                val eyeDistance = faces[i]?.eyesDistance()?.toDouble() ?: continue
                if (eyeDistance <= 0.0) continue
                val faceWidth = eyeDistance * 2.5
                val faceHeight = eyeDistance * 3.2
                coverage += (faceWidth * faceHeight) / (width.toDouble() * height.toDouble())
            }
            FaceSummary(count, coverage.coerceIn(0.0, 1.0))
        } finally {
            bitmap.recycle()
        }
    }

    private const val MAX_FACES = 6
}
