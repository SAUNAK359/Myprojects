package com.saunak.director.engine.quality

import android.graphics.Bitmap
import com.saunak.director.engine.reconstruction.EvidenceRecord
import com.saunak.director.processing.PhotoQuality
import kotlin.math.abs
import kotlin.math.max

object ReconstructionQualityGate {
    fun evaluate(
        bitmap: Bitmap,
        evidence: List<EvidenceRecord>,
        council: QualityCouncil
    ): QualityVerdict {
        require(!bitmap.isRecycled && bitmap.width > 0 && bitmap.height > 0)
        require(evidence.isNotEmpty())
        val technical = PhotoQuality.analyze(bitmap)
        val evidenceSupport = evidence.map { it.evidenceSupport }.average().coerceIn(0.0, 1.0)
        val geometry = evidence.map { minOf(it.alignmentConfidence, it.geometryConfidence) }.average().coerceIn(0.0, 1.0)
        val motion = evidence.map { it.motionConfidence }.average().coerceIn(0.0, 1.0)
        val saturation = evidence.map { it.saturationConfidence }.average().coerceIn(0.0, 1.0)
        val color = colorNeutralityScore(bitmap)

        val signals = mutableListOf(
            QualitySignal(
                axis = QualityAxis.TECHNICAL,
                score = if (technical.catastrophic) 0.05 else technical.heuristicScore.coerceIn(0.0, 1.0),
                confidence = 0.94,
                source = "deterministic_technical_gate",
                issue = when {
                    technical.catastrophic -> "catastrophic luminance/detail failure"
                    technical.brightClipFraction > 0.08 -> "highlight clipping"
                    technical.darkClipFraction > 0.20 -> "shadow clipping"
                    technical.edgeEnergy < 1.0 -> "low detail or motion blur"
                    else -> null
                }
            ),
            QualitySignal(
                axis = QualityAxis.EVIDENCE,
                score = evidenceSupport,
                confidence = 0.98,
                source = "raw_evidence_ledger",
                issue = if (evidenceSupport < 0.45) "weak multi-frame support" else null
            ),
            QualitySignal(
                axis = QualityAxis.GEOMETRY,
                score = (0.72 * geometry + 0.28 * motion).coerceIn(0.0, 1.0),
                confidence = 0.92,
                source = "cfa_alignment",
                issue = if (geometry < 0.48 || motion < 0.45) "alignment or motion confidence is weak" else null
            ),
            QualitySignal(
                axis = QualityAxis.COLOR,
                score = color,
                confidence = 0.72,
                source = "linear_isp_color_sanity",
                issue = if (color < 0.42) "severe global color imbalance" else null
            )
        )
        if (saturation < 0.45) {
            signals += QualitySignal(
                axis = QualityAxis.TECHNICAL,
                score = saturation,
                confidence = 0.90,
                source = "raw_saturation_support",
                issue = "insufficient unsaturated RAW support"
            )
        }
        return council.evaluate(signals, requireEvidenceForPass = true)
    }

    private fun colorNeutralityScore(bitmap: Bitmap): Double {
        val stepX = max(1, bitmap.width / 48)
        val stepY = max(1, bitmap.height / 48)
        var r = 0.0
        var g = 0.0
        var b = 0.0
        var count = 0
        var y = 0
        while (y < bitmap.height) {
            var x = 0
            while (x < bitmap.width) {
                val p = bitmap.getPixel(x, y)
                r += (p ushr 16) and 0xff
                g += (p ushr 8) and 0xff
                b += p and 0xff
                count++
                x += stepX
            }
            y += stepY
        }
        if (count == 0) return 0.5
        r /= count
        g /= count
        b /= count
        val mean = (r + g + b) / 3.0
        if (mean < 1.0) return 0.5
        val imbalance = (abs(r - mean) + abs(g - mean) + abs(b - mean)) / (3.0 * mean)
        return (1.0 - imbalance / 0.75).coerceIn(0.0, 1.0)
    }
}
