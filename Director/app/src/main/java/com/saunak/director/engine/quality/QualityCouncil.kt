package com.saunak.director.engine.quality

import com.saunak.director.engine.reconstruction.NormalizedRect

enum class QualityAxis { TECHNICAL, HUMAN, COLOR, GEOMETRY, EVIDENCE, PERCEPTUAL }
enum class QualityDisposition { PASS, LOCAL_REPAIR, REJECT }
enum class RepairStage { RECONSTRUCTION, MATTING, DENOISE, DETAIL, WHITE_BALANCE, COLOR, TONE, GEOMETRY, NONE }

data class QualitySignal(
    val axis: QualityAxis,
    val score: Double,
    val confidence: Double,
    val source: String,
    val region: NormalizedRect? = null,
    val issue: String? = null
) {
    init {
        require(score in 0.0..1.0)
        require(confidence in 0.0..1.0)
        require(source.isNotBlank())
    }
}

data class RepairAction(
    val stage: RepairStage,
    val region: NormalizedRect?,
    val reason: String,
    /** Conservative relative strength adjustment; exact renderer maps it to its own controls. */
    val magnitude: Double
) {
    init { require(stage != RepairStage.NONE); require(reason.isNotBlank()); require(magnitude in -1.0..1.0) }
}

data class QualityVerdict(
    val disposition: QualityDisposition,
    /** Missing axes are intentionally absent; absence must never be interpreted as a perfect score. */
    val axisScores: Map<QualityAxis, Double>,
    val unknownAxes: Set<QualityAxis>,
    val repairs: List<RepairAction>,
    val criticalReasons: List<String>
)

/** Independent judges vote by axis. Low-confidence model opinions cannot overrule strong physical checks. */
class QualityCouncil {
    fun evaluate(
        signals: List<QualitySignal>,
        requireEvidenceForPass: Boolean = true
    ): QualityVerdict {
        require(signals.isNotEmpty())
        val axisScores = buildMap {
            QualityAxis.entries.forEach { axis ->
                val votes = signals.filter { it.axis == axis }
                if (votes.isNotEmpty()) put(axis, weightedRobustScore(votes))
            }
        }
        val unknown = QualityAxis.entries.filterNot(axisScores::containsKey).toSet()
        val critical = mutableListOf<String>()
        val repairs = mutableListOf<RepairAction>()

        signals.filter { it.confidence >= 0.75 && it.score < 0.35 }.forEach { signal ->
            val issue = signal.issue ?: "${signal.axis.name.lowercase()} quality failure"
            when (signal.axis) {
                QualityAxis.EVIDENCE -> critical += "unsupported reconstruction: $issue"
                QualityAxis.HUMAN -> repairs += RepairAction(RepairStage.DETAIL, signal.region, issue, -0.35)
                QualityAxis.COLOR -> repairs += RepairAction(RepairStage.WHITE_BALANCE, signal.region, issue, -0.25)
                QualityAxis.GEOMETRY -> repairs += RepairAction(RepairStage.GEOMETRY, signal.region, issue, -0.4)
                QualityAxis.TECHNICAL -> repairs += inferTechnicalRepair(signal)
                QualityAxis.PERCEPTUAL -> repairs += RepairAction(RepairStage.TONE, signal.region, issue, -0.18)
            }
        }

        axisScores[QualityAxis.EVIDENCE]?.let { if (it < 0.30) critical += "captured evidence does not support the rendered result" }
        axisScores[QualityAxis.HUMAN]?.let { if (it < 0.25) critical += "human fidelity is below the release threshold" }
        axisScores[QualityAxis.GEOMETRY]?.let { if (it < 0.25) critical += "geometric reconstruction is unstable" }
        axisScores[QualityAxis.TECHNICAL]?.let { if (it < 0.20) critical += "technical image quality is catastrophically low" }

        // Absence of evidence is not itself proof that a deterministic image is bad, but it prevents
        // a neural/rewritten result from being declared fully validated.
        if (requireEvidenceForPass && QualityAxis.EVIDENCE in unknown) {
            repairs += RepairAction(
                RepairStage.RECONSTRUCTION,
                region = null,
                reason = "evidence judge unavailable; retain or compare against deterministic reconstruction",
                magnitude = -0.20
            )
        }

        val disposition = when {
            critical.isNotEmpty() -> QualityDisposition.REJECT
            repairs.isNotEmpty() || axisScores.values.any { it < 0.62 } -> QualityDisposition.LOCAL_REPAIR
            requireEvidenceForPass && QualityAxis.EVIDENCE in unknown -> QualityDisposition.LOCAL_REPAIR
            else -> QualityDisposition.PASS
        }
        return QualityVerdict(disposition, axisScores, unknown, repairs.distinct(), critical.distinct())
    }

    private fun weightedRobustScore(votes: List<QualitySignal>): Double {
        val credible = votes.filter { it.confidence > 0.0 }
        if (credible.isEmpty()) return 0.5 // unknown/neutral, never perfect
        val sorted = credible.sortedBy { it.score }
        val totalWeight = sorted.sumOf { it.confidence }.coerceAtLeast(1e-9)
        val mean = sorted.sumOf { it.score * it.confidence } / totalWeight
        val lowerQuartile = sorted[((sorted.size - 1) * 0.25).toInt()].score
        return (mean * 0.72 + lowerQuartile * 0.28).coerceIn(0.0, 1.0)
    }

    private fun inferTechnicalRepair(signal: QualitySignal): RepairAction {
        val issue = signal.issue.orEmpty().lowercase()
        return when {
            "noise" in issue -> RepairAction(RepairStage.DENOISE, signal.region, signal.issue ?: "noise", 0.25)
            "sharp" in issue || "ring" in issue || "halo" in issue ->
                RepairAction(RepairStage.DETAIL, signal.region, signal.issue ?: "detail artifact", -0.30)
            "ghost" in issue || "motion" in issue ->
                RepairAction(RepairStage.RECONSTRUCTION, signal.region, signal.issue ?: "fusion artifact", -0.35)
            "clip" in issue || "highlight" in issue ->
                RepairAction(RepairStage.TONE, signal.region, signal.issue ?: "tonal clipping", -0.25)
            else -> RepairAction(RepairStage.RECONSTRUCTION, signal.region, signal.issue ?: "technical failure", -0.15)
        }
    }
}
