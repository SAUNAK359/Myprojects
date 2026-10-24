package com.saunak.director

import com.saunak.director.engine.belief.Belief
import com.saunak.director.engine.belief.BeliefKey
import com.saunak.director.engine.belief.SceneBeliefGraph
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SceneBeliefGraphTest {
    @Test fun newerEvidenceWinsAndConfidenceDecays() {
        val graph = SceneBeliefGraph()
        val key = BeliefKey("test.depth")
        graph.put(key, Belief(2.0, confidence = 0.9, uncertainty = 0.1, timestampNs = 1_000L, source = "depth", halfLifeNs = 1_000L))
        graph.put(key, Belief(3.0, confidence = 0.95, uncertainty = 0.1, timestampNs = 900L, source = "stale", halfLifeNs = 1_000L))

        val belief = requireNotNull(graph.get<Double>(key))
        assertEquals(2.0, belief.value, 0.0)
        assertEquals(0.45, belief.effectiveConfidence(2_000L), 1e-6)

        graph.prune(nowNs = 20_000L, minimumEffectiveConfidence = 0.02)
        assertTrue(graph.snapshot(20_000L).isEmpty())
    }
}
