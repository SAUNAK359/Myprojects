package com.saunak.director

import com.saunak.director.engine.belief.Belief
import com.saunak.director.engine.belief.BeliefKey
import com.saunak.director.engine.device.BackendExecutionProfile
import com.saunak.director.engine.device.ComputeBackend
import com.saunak.director.engine.experts.ExpertDescriptor
import com.saunak.director.engine.experts.ExpertScheduler
import com.saunak.director.engine.experts.InformationNeed
import com.saunak.director.engine.models.ModelTier
import com.saunak.director.engine.runtime.RuntimePressure
import com.saunak.director.engine.runtime.WorkPriority
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ExpertSchedulerTest {
    private val key = BeliefKey("geometry.depth")
    private val expert = ExpertDescriptor(
        id = "depth-oracle",
        modelId = "metric-depth",
        tier = ModelTier.CAPTURE,
        produces = setOf(key),
        inputSignature = "1x3x384x384-fp16",
        supportedBackends = setOf(ComputeBackend.NNAPI, ComputeBackend.CPU_XNNPACK),
        reliability = 0.96
    )
    private val profiles = listOf(
        BackendExecutionProfile("metric-depth", ComputeBackend.NNAPI, "1x3x384x384-fp16", 0.995, 0, 25.0, 31.0, 180.0, 4.0),
        BackendExecutionProfile("metric-depth", ComputeBackend.CPU_XNNPACK, "1x3x384x384-fp16", 1.0, 0, 90.0, 110.0, 250.0, 7.0)
    )

    @Test fun freshConfidentBeliefSuppressesRedundantOracle() {
        val now = 10_000L
        val current = Belief(2.0, 0.96, 0.02, now - 100L, "depth-monitor", 10_000L)
        val need = InformationNeed(key, 0.9, 0.9, 2_000L, 60.0, WorkPriority.CAPTURE)
        assertNull(ExpertScheduler().choose(need, current, listOf(expert), profiles, RuntimePressure(), now))
    }

    @Test fun staleBeliefDispatchesBestMeasuredOracle() {
        val now = 20_000L
        val current = Belief(2.0, 0.7, 0.3, 5_000L, "depth-monitor", 2_000L)
        val need = InformationNeed(key, 1.0, 0.9, 2_000L, 80.0, WorkPriority.CAPTURE)
        val dispatch = requireNotNull(ExpertScheduler().choose(need, current, listOf(expert), profiles, RuntimePressure(), now))
        assertEquals(ComputeBackend.NNAPI, dispatch.execution.profile.backend)
        assertTrue(dispatch.valueOfInformation > 0.0)
    }
}
