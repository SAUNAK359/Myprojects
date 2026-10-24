package com.saunak.director.engine.belief

import java.util.concurrent.ConcurrentHashMap
import kotlin.math.exp

@JvmInline
value class BeliefKey(val value: String) {
    init { require(value.isNotBlank()) }
}

data class Belief<T : Any>(
    val value: T,
    val confidence: Double,
    /** Domain-specific uncertainty. Zero is certain; larger values are less certain. */
    val uncertainty: Double = 0.0,
    val timestampNs: Long,
    val source: String,
    /** Expected half-life of this observation when propagated without a fresh measurement. */
    val halfLifeNs: Long
) {
    init {
        require(confidence in 0.0..1.0)
        require(uncertainty >= 0.0)
        require(timestampNs > 0)
        require(source.isNotBlank())
        require(halfLifeNs > 0)
    }

    fun effectiveConfidence(nowNs: Long): Double {
        val age = (nowNs - timestampNs).coerceAtLeast(0L).toDouble()
        if (age == 0.0) return confidence
        val decay = exp(-LN_2 * age / halfLifeNs.toDouble())
        return (confidence * decay).coerceIn(0.0, 1.0)
    }

    companion object { private const val LN_2 = 0.6931471805599453 }
}

/** Thread-safe temporal world state. Model outputs are evidence, never unqualified truth. */
class SceneBeliefGraph {
    private val state = ConcurrentHashMap<BeliefKey, Belief<Any>>()

    @Suppress("UNCHECKED_CAST")
    fun <T : Any> get(key: BeliefKey): Belief<T>? = state[key] as? Belief<T>

    fun <T : Any> put(key: BeliefKey, belief: Belief<T>) {
        @Suppress("UNCHECKED_CAST")
        val typed = belief as Belief<Any>
        state.compute(key) { _, existing ->
            if (existing == null || belief.timestampNs >= existing.timestampNs) typed else existing
        }
    }

    fun remove(key: BeliefKey) { state.remove(key) }

    fun invalidateBySource(source: String) {
        state.entries.removeIf { it.value.source == source }
    }

    fun prune(nowNs: Long, minimumEffectiveConfidence: Double = 0.02) {
        require(minimumEffectiveConfidence in 0.0..1.0)
        state.entries.removeIf { it.value.effectiveConfidence(nowNs) < minimumEffectiveConfidence }
    }

    fun snapshot(nowNs: Long): Map<BeliefKey, Belief<Any>> =
        state.filterValues { it.effectiveConfidence(nowNs) > 0.0 }

    companion object Keys {
        val SCENE = BeliefKey("scene.class")
        val DYNAMIC_RANGE = BeliefKey("scene.dynamic_range")
        val PERSON_COUNT = BeliefKey("human.person_count")
        val PRIMARY_FACE = BeliefKey("human.primary_face")
        val DEPTH = BeliefKey("geometry.depth")
        val CAMERA_MOTION = BeliefKey("motion.camera")
        val SUBJECT_MOTION = BeliefKey("motion.subject")
        val NOISE = BeliefKey("quality.noise")
        val BLUR = BeliefKey("quality.blur")
        val HIGHLIGHT_RISK = BeliefKey("quality.highlight_risk")
    }
}
