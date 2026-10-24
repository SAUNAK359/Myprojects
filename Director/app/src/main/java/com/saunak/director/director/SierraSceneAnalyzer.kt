package com.saunak.director.director

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.media.FaceDetector
import androidx.camera.core.CameraSelector
import com.saunak.director.engine.experts.FaceDetection
import com.saunak.director.engine.experts.SubjectDepthEvidence
import com.saunak.director.processing.TechnicalQuality

/** Lightweight semantic guard used before either Sierra AI or Sierra beta plans the shot. */
data class SierraSceneSignals(
    val branch: String,
    val confidence: Double,
    val faceCount: Int = 0,
    val faceCoverage: Double = 0.0,
    val primaryHumanSubject: Boolean = false,
    val brightSky: Boolean = false,
    val urbanStructure: Boolean = false,
    val faceSource: String = "legacy",
    val relativeDepthConfidence: Double = 0.0,
    val subjectSeparation: Double = 0.0,
    val depthSource: String? = null
) {
    init {
        require(confidence in 0.0..1.0)
        require(relativeDepthConfidence in 0.0..1.0)
        require(subjectSeparation in 0.0..1.0)
    }
}

object SierraSceneAnalyzer {
    fun analyze(
        source: Bitmap,
        quality: TechnicalQuality,
        lensFacing: Int,
        expertFaces: List<FaceDetection>? = null,
        depthEvidence: SubjectDepthEvidence? = null
    ): SierraSceneSignals {
        val faces = if (expertFaces != null) {
            FaceSummary(
                count = expertFaces.size,
                coverage = expertFaces.sumOf { face ->
                    (face.width.toDouble() * face.height.toDouble()) /
                        (source.width.toDouble() * source.height.toDouble())
                }.coerceIn(0.0, 1.0),
                source = "yunet"
            )
        } else {
            runCatching { detectFaces(source) }.getOrDefault(FaceSummary(0, 0.0, "legacy"))
        }

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

        val baseConfidence = when (branch) {
            "portrait", "selfie", "group" -> 0.92
            "low_light" -> 0.86
            "architecture" -> if (brightSky) 0.82 else 0.74
            "landscape" -> if (brightSky) 0.78 else 0.66
            else -> 0.48
        }
        var confidence = if (faces.source == "yunet" && branch in HUMAN_BRANCHES) {
            (baseConfidence + 0.035).coerceAtMost(0.97)
        } else baseConfidence

        // Relative depth is allowed to strengthen a human/portrait hypothesis only when it agrees
        // with independently detected faces. It can never manufacture a person scene by itself.
        if (branch in HUMAN_BRANCHES && depthEvidence != null) {
            confidence = (confidence + 0.025 * depthEvidence.confidence + 0.015 * depthEvidence.separation)
                .coerceAtMost(0.985)
        }

        return SierraSceneSignals(
            branch = branch,
            confidence = confidence,
            faceCount = faces.count,
            faceCoverage = faces.coverage,
            primaryHumanSubject = primaryHuman,
            brightSky = brightSky,
            urbanStructure = structured,
            faceSource = faces.source,
            relativeDepthConfidence = depthEvidence?.confidence ?: 0.0,
            subjectSeparation = depthEvidence?.separation ?: 0.0,
            depthSource = depthEvidence?.source
        )
    }

    private data class FaceSummary(val count: Int, val coverage: Double, val source: String)

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
            FaceSummary(count, coverage.coerceIn(0.0, 1.0), "legacy")
        } finally {
            bitmap.recycle()
        }
    }

    private val HUMAN_BRANCHES = setOf("portrait", "selfie", "group")
    private const val MAX_FACES = 6
}
