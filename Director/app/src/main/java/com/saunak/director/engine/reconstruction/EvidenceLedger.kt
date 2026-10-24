package com.saunak.director.engine.reconstruction

import java.util.concurrent.ConcurrentHashMap

@JvmInline
value class RegionId(val value: String) { init { require(value.isNotBlank()) } }

data class NormalizedRect(val left: Double, val top: Double, val right: Double, val bottom: Double) {
    init {
        require(left in 0.0..1.0 && top in 0.0..1.0 && right in 0.0..1.0 && bottom in 0.0..1.0)
        require(right > left && bottom > top)
    }
}

enum class RegionSemantic { HUMAN_FACE, HUMAN_SKIN, HUMAN_HAIR, EYES, TEXT, LOGO, ARCHITECTURE, FOLIAGE, SKY, BACKGROUND, GENERAL }

data class EvidenceRecord(
    val regionId: RegionId,
    val bounds: NormalizedRect,
    val semantic: RegionSemantic,
    val sourceFrameIds: Set<Long>,
    val fusionWeights: Map<Long, Double>,
    val alignmentConfidence: Double,
    val motionConfidence: Double,
    val exposureValidity: Double,
    val signalToNoiseConfidence: Double,
    val saturationConfidence: Double,
    val geometryConfidence: Double,
    /** Normalized difference between neural output and evidence-preserving reconstruction. */
    val neuralResidual: Double,
    val timestampNs: Long
) {
    init {
        require(sourceFrameIds.isNotEmpty())
        require(fusionWeights.keys.all { it in sourceFrameIds })
        require(fusionWeights.values.all { it >= 0.0 })
        require(fusionWeights.values.sum() > 0.0)
        listOf(alignmentConfidence, motionConfidence, exposureValidity, signalToNoiseConfidence,
            saturationConfidence, geometryConfidence).forEach { require(it in 0.0..1.0) }
        require(neuralResidual >= 0.0)
        require(timestampNs > 0)
    }

    val normalizedFusionWeights: Map<Long, Double>
        get() {
            val sum = fusionWeights.values.sum()
            return fusionWeights.mapValues { it.value / sum }
        }

    /** Conservative evidence support: a single weak physical dimension should lower trust. */
    val evidenceSupport: Double
        get() = listOf(alignmentConfidence, motionConfidence, exposureValidity,
            signalToNoiseConfidence, saturationConfidence, geometryConfidence).sorted().let {
            // Blend the weakest dimension with the average to avoid both brittle min() and optimistic mean().
            (it.first() * 0.6 + it.average() * 0.4).coerceIn(0.0, 1.0)
        }

    val hallucinationRisk: Double
        get() {
            val semanticMultiplier = when (semantic) {
                RegionSemantic.HUMAN_FACE, RegionSemantic.EYES, RegionSemantic.TEXT, RegionSemantic.LOGO -> 1.35
                RegionSemantic.ARCHITECTURE, RegionSemantic.HUMAN_SKIN, RegionSemantic.HUMAN_HAIR -> 1.15
                else -> 1.0
            }
            val residualRisk = neuralResidual / (1.0 + neuralResidual)
            return ((1.0 - evidenceSupport) * 0.65 + residualRisk * 0.35 * semanticMultiplier).coerceIn(0.0, 1.0)
        }
}

/** Region-level provenance ledger. Each visible repair/arbitration decision can be traced to captured evidence. */
class EvidenceLedger {
    private val records = ConcurrentHashMap<RegionId, EvidenceRecord>()

    fun put(record: EvidenceRecord) {
        records.compute(record.regionId) { _, existing ->
            if (existing == null || record.timestampNs >= existing.timestampNs) record else existing
        }
    }

    fun get(regionId: RegionId): EvidenceRecord? = records[regionId]
    fun snapshot(): List<EvidenceRecord> = records.values.sortedBy { it.regionId.value }
    fun remove(regionId: RegionId) { records.remove(regionId) }
    fun clear() = records.clear()

    fun risky(threshold: Double = 0.35): List<EvidenceRecord> {
        require(threshold in 0.0..1.0)
        return snapshot().filter { it.hallucinationRisk >= threshold }
    }
}
