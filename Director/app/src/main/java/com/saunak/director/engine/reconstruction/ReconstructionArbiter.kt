package com.saunak.director.engine.reconstruction

enum class ReconstructionPath { DETERMINISTIC, NEURAL }

data class ReconstructionCandidate(
    val regionId: RegionId,
    val path: ReconstructionPath,
    val detailScore: Double,
    val noiseScore: Double,
    val fidelityScore: Double,
    val artifactRisk: Double,
    val perceptualScore: Double
) {
    init {
        listOf(detailScore, noiseScore, fidelityScore, artifactRisk, perceptualScore).forEach { require(it in 0.0..1.0) }
    }
}

data class ReconstructionChoice(
    val regionId: RegionId,
    val chosen: ReconstructionCandidate,
    val alternate: ReconstructionCandidate?,
    val evidenceSupport: Double,
    val reason: String
)

/**
 * Chooses reconstruction per region. Protected semantics strongly weight captured evidence and
 * fidelity, so a visually attractive neural result cannot overwrite unsupported face/text detail.
 */
class ReconstructionArbiter {
    fun choose(
        evidence: EvidenceRecord,
        candidates: List<ReconstructionCandidate>
    ): ReconstructionChoice {
        require(candidates.isNotEmpty())
        require(candidates.all { it.regionId == evidence.regionId })

        val ranked = candidates.sortedByDescending { score(evidence, it) }
        val chosen = ranked.first()
        return ReconstructionChoice(
            regionId = evidence.regionId,
            chosen = chosen,
            alternate = ranked.getOrNull(1),
            evidenceSupport = evidence.evidenceSupport,
            reason = buildReason(evidence, chosen)
        )
    }

    private fun score(evidence: EvidenceRecord, candidate: ReconstructionCandidate): Double {
        val protected = evidence.semantic in PROTECTED
        val evidenceWeight = if (protected) 0.34 else 0.20
        val fidelityWeight = if (protected) 0.31 else 0.20
        val detailWeight = if (protected) 0.10 else 0.22
        val noiseWeight = if (protected) 0.08 else 0.14
        val perceptualWeight = if (protected) 0.07 else 0.14
        val riskWeight = if (protected) 0.34 else 0.22

        var value = evidence.evidenceSupport * evidenceWeight +
            candidate.fidelityScore * fidelityWeight +
            candidate.detailScore * detailWeight +
            candidate.noiseScore * noiseWeight +
            candidate.perceptualScore * perceptualWeight -
            candidate.artifactRisk * riskWeight

        if (candidate.path == ReconstructionPath.NEURAL) {
            value -= evidence.hallucinationRisk * if (protected) 0.42 else 0.19
            if (evidence.evidenceSupport < if (protected) 0.78 else 0.55) {
                value -= if (protected) 0.22 else 0.08
            }
        } else if (protected) {
            // Small tie-break in favour of the evidence-preserving path for identity/text regions.
            value += 0.025
        }
        return value
    }

    private fun buildReason(evidence: EvidenceRecord, chosen: ReconstructionCandidate): String = when {
        chosen.path == ReconstructionPath.DETERMINISTIC && evidence.semantic in PROTECTED ->
            "protected ${evidence.semantic.name.lowercase()} region prefers evidence-preserving reconstruction"
        chosen.path == ReconstructionPath.DETERMINISTIC && evidence.hallucinationRisk >= 0.35 ->
            "neural residual/evidence mismatch exceeds safe threshold"
        chosen.path == ReconstructionPath.NEURAL && evidence.evidenceSupport >= 0.8 ->
            "neural reconstruction is supported by high-confidence multi-frame evidence"
        chosen.path == ReconstructionPath.NEURAL ->
            "neural candidate wins regional quality score within fidelity limits"
        else -> "deterministic candidate wins regional quality score"
    }

    companion object {
        private val PROTECTED = setOf(
            RegionSemantic.HUMAN_FACE,
            RegionSemantic.HUMAN_SKIN,
            RegionSemantic.EYES,
            RegionSemantic.TEXT,
            RegionSemantic.LOGO
        )
    }
}
