package com.saunak.director.engine.runtime

import com.saunak.director.engine.device.BackendExecutionProfile
import com.saunak.director.engine.device.ComputeBackend
import kotlin.math.max

enum class WorkPriority { SENSOR_DEADLINE, SHUTTER, CAPTURE, HUMAN, PERCEPTION, RECONSTRUCTION, QUALITY, BACKGROUND }

data class InferenceRequest(
    val requestId: String,
    val modelId: String,
    val inputSignature: String,
    /** Estimated final-photo utility gained by obtaining a fresh result, 0..1. */
    val valueOfInformation: Double,
    val priority: WorkPriority,
    val deadlineMs: Double,
    val minimumGraphCoverage: Double = 0.98,
    val maximumTransferBoundaries: Int = 1,
    val requiredBackends: Set<ComputeBackend> = emptySet()
) {
    init {
        require(requestId.isNotBlank() && modelId.isNotBlank() && inputSignature.isNotBlank())
        require(valueOfInformation in 0.0..1.0)
        require(deadlineMs > 0)
        require(minimumGraphCoverage in 0.0..1.0)
        require(maximumTransferBoundaries >= 0)
    }
}

data class RuntimePressure(
    val memoryPressure: Double = 0.0,
    val bandwidthPressure: Double = 0.0,
    val thermalPressure: Double = 0.0,
    val cameraBandwidthPressure: Double = 0.0,
    /** workload id -> measured fractional slowdown when overlapped with the candidate. */
    val interferencePenalty: Map<String, Double> = emptyMap()
) {
    init {
        require(memoryPressure in 0.0..1.0)
        require(bandwidthPressure in 0.0..1.0)
        require(thermalPressure in 0.0..1.0)
        require(cameraBandwidthPressure in 0.0..1.0)
        require(interferencePenalty.values.all { it >= 0.0 })
    }
}

data class ExecutionDecision(
    val profile: BackendExecutionProfile,
    val predictedLatencyMs: Double,
    val utilityScore: Double,
    val deadlineRisk: Double,
    val reason: String
)

/**
 * Sierra Compute Hypervisor selection core. It chooses measured compiled graph variants rather than
 * assuming a model belongs on a GPU/NPU. The executor remains separate so this class is deterministic
 * and directly unit-testable.
 */
class ComputeHypervisor {
    fun choose(
        request: InferenceRequest,
        candidates: List<BackendExecutionProfile>,
        pressure: RuntimePressure
    ): ExecutionDecision? {
        val eligible = candidates.asSequence()
            .filter { it.modelId == request.modelId && it.inputSignature == request.inputSignature }
            .filter { request.requiredBackends.isEmpty() || it.backend in request.requiredBackends }
            .filter { it.graphCoverage >= request.minimumGraphCoverage }
            .filter { it.transferBoundaries <= request.maximumTransferBoundaries }
            .toList()
        if (eligible.isEmpty()) return null

        return eligible.map { profile -> evaluate(request, profile, pressure) }
            .maxWithOrNull(
                compareBy<ExecutionDecision> { it.utilityScore }
                    .thenByDescending { -it.predictedLatencyMs }
            )
    }

    private fun evaluate(
        request: InferenceRequest,
        profile: BackendExecutionProfile,
        pressure: RuntimePressure
    ): ExecutionDecision {
        val priorityWeight = when (request.priority) {
            WorkPriority.SENSOR_DEADLINE -> 4.0
            WorkPriority.SHUTTER -> 3.5
            WorkPriority.CAPTURE -> 3.0
            WorkPriority.HUMAN -> 2.5
            WorkPriority.RECONSTRUCTION -> 2.2
            WorkPriority.PERCEPTION -> 1.8
            WorkPriority.QUALITY -> 1.2
            WorkPriority.BACKGROUND -> 0.6
        }
        val interference = pressure.interferencePenalty[profile.modelId] ?: 0.0
        val memoryAmplifier = 1.0 + pressure.memoryPressure * (profile.peakMemoryMb / 512.0).coerceAtMost(3.0)
        val bandwidthAmplifier = 1.0 + pressure.bandwidthPressure * 0.7 + pressure.cameraBandwidthPressure * 0.8
        val thermalAmplifier = 1.0 + pressure.thermalPressure * (profile.sustainedPowerScore / 10.0).coerceAtMost(1.5)
        val fallbackAmplifier = 1.0 + profile.transferBoundaries * 0.18
        val predictedLatency = profile.p95LatencyMs * (1.0 + interference) * memoryAmplifier * bandwidthAmplifier * thermalAmplifier * fallbackAmplifier
        val deadlineRisk = (predictedLatency / request.deadlineMs).coerceAtLeast(0.0)
        val deadlinePenalty = max(0.0, deadlineRisk - 1.0) * 5.0
        val cost = 1.0 + predictedLatency / request.deadlineMs + profile.peakMemoryMb / 2048.0 + deadlinePenalty
        val utility = request.valueOfInformation * priorityWeight * profile.graphCoverage / cost
        return ExecutionDecision(
            profile = profile,
            predictedLatencyMs = predictedLatency,
            utilityScore = utility,
            deadlineRisk = deadlineRisk,
            reason = "voi=${request.valueOfInformation};coverage=${profile.graphCoverage};transfers=${profile.transferBoundaries};pressure=${"%.2f".format(pressure.bandwidthPressure)}"
        )
    }
}
