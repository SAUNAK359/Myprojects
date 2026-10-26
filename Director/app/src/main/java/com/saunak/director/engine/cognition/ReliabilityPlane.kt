package com.saunak.director.engine.cognition

import kotlin.math.max

class ReliabilityPlane(private val policy: CognitionPolicy) {
    fun worldReliability(
        createdAtNs: Long,
        confidence: Double,
        uncertainty: Double,
        nowNs: Long,
        oodScore: Double = 0.0,
        extraFlags: Set<ReliabilityFlag> = emptySet()
    ): ReliabilityEnvelope {
        require(nowNs >= createdAtNs)
        val ageNs = nowNs - createdAtNs
        val staleNs = policy.staleAfterMs * 1_000_000L
        val freshness = (1.0 - ageNs.toDouble() / staleNs.toDouble()).coerceIn(0.0, 1.0)
        val flags = buildSet {
            addAll(extraFlags)
            if (freshness <= 0.0) add(ReliabilityFlag.STALE)
            if (confidence < policy.minStyleReliability) add(ReliabilityFlag.LOW_CONFIDENCE)
            if (uncertainty > 1.0 - policy.minStyleReliability) add(ReliabilityFlag.HIGH_UNCERTAINTY)
            if (oodScore >= 0.5) add(ReliabilityFlag.OUT_OF_DISTRIBUTION)
        }
        return ReliabilityEnvelope(
            confidence = confidence.coerceIn(0.0, 1.0),
            uncertainty = uncertainty.coerceIn(0.0, 1.0),
            freshness = freshness,
            oodScore = oodScore.coerceIn(0.0, 1.0),
            flags = flags
        )
    }

    fun <T> validate(observation: ModelObservation<T>, currentStateId: Long, nowNs: Long): ReliabilityEnvelope {
        val stale = !observation.isFresh(nowNs) || observation.stateId != currentStateId
        val uncertainty = max(observation.aleatoricUncertainty, observation.epistemicUncertainty)
        return worldReliability(
            createdAtNs = observation.timestampNs,
            confidence = observation.confidence,
            uncertainty = uncertainty,
            nowNs = max(nowNs, observation.timestampNs),
            oodScore = observation.oodScore,
            extraFlags = observation.flags + if (stale) setOf(ReliabilityFlag.STALE) else emptySet()
        ).let { envelope ->
            if (!stale) envelope else envelope.copy(freshness = 0.0, flags = envelope.flags + ReliabilityFlag.STALE)
        }
    }

    fun aggressiveness(reliability: ReliabilityEnvelope, requested: Double): Double =
        (requested.coerceIn(0.0, 1.0) * reliability.effectiveConfidence).coerceIn(0.0, 1.0)
}
