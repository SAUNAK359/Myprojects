package com.saunak.director

import com.saunak.director.engine.device.BackendExecutionProfile
import com.saunak.director.engine.device.ComputeBackend
import com.saunak.director.engine.runtime.ComputeHypervisor
import com.saunak.director.engine.runtime.InferenceRequest
import com.saunak.director.engine.runtime.RuntimePressure
import com.saunak.director.engine.runtime.WorkPriority
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ComputeHypervisorTest {
    private val qnn = BackendExecutionProfile(
        modelId = "depth",
        backend = ComputeBackend.NPU_QNN,
        inputSignature = "1x3x384x384-fp16",
        graphCoverage = 0.995,
        transferBoundaries = 0,
        p50LatencyMs = 18.0,
        p95LatencyMs = 23.0,
        peakMemoryMb = 160.0,
        sustainedPowerScore = 4.0
    )
    private val gpu = qnn.copy(
        backend = ComputeBackend.GPU_VULKAN,
        graphCoverage = 1.0,
        p50LatencyMs = 27.0,
        p95LatencyMs = 32.0,
        peakMemoryMb = 220.0,
        sustainedPowerScore = 6.0
    )

    @Test fun choosesMeasuredBestEligibleGraph() {
        val request = InferenceRequest(
            requestId = "r1",
            modelId = "depth",
            inputSignature = "1x3x384x384-fp16",
            valueOfInformation = 0.9,
            priority = WorkPriority.CAPTURE,
            deadlineMs = 60.0
        )
        val decision = requireNotNull(ComputeHypervisor().choose(request, listOf(gpu, qnn), RuntimePressure()))
        assertEquals(ComputeBackend.NPU_QNN, decision.profile.backend)
    }

    @Test fun refusesGraphsThatWouldExceedFallbackPolicy() {
        val request = InferenceRequest(
            requestId = "r2",
            modelId = "depth",
            inputSignature = "1x3x384x384-fp16",
            valueOfInformation = 0.5,
            priority = WorkPriority.PERCEPTION,
            deadlineMs = 100.0,
            minimumGraphCoverage = 1.0,
            maximumTransferBoundaries = 0
        )
        val partial = qnn.copy(graphCoverage = 0.96, transferBoundaries = 2)
        assertNull(ComputeHypervisor().choose(request, listOf(partial), RuntimePressure()))
    }
}
