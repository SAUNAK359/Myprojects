package com.saunak.director.engine.experts

import com.saunak.director.engine.belief.Belief
import com.saunak.director.engine.belief.BeliefKey
import com.saunak.director.engine.device.BackendExecutionProfile
import com.saunak.director.engine.device.ComputeBackend
import com.saunak.director.engine.models.ModelTier
import com.saunak.director.engine.runtime.ComputeHypervisor
import com.saunak.director.engine.runtime.ExecutionDecision
import com.saunak.director.engine.runtime.InferenceRequest
import com.saunak.director.engine.runtime.RuntimePressure
import com.saunak.director.engine.runtime.WorkPriority

/** A question Sierra needs answered, not an instruction to run a particular model. */
data class InformationNeed(
    val key: BeliefKey,
    /** How much a better answer can affect the final photograph. */
    val finalPhotoImpact: Double,
    val requiredConfidence: Double,
    val maxAgeNs: Long,
    val deadlineMs: Double,
    val priority: WorkPriority
) {
    init {
        require(finalPhotoImpact in 0.0..1.0)
        require(requiredConfidence in 0.0..1.0)
        require(maxAgeNs > 0)
        require(deadlineMs > 0)
    }
}

data class ExpertDescriptor(
    val id: String,
    val modelId: String,
    val tier: ModelTier,
    val produces: Set<BeliefKey>,
    val inputSignature: String,
    val supportedBackends: Set<ComputeBackend>,
    /** 0..1 empirical reliability on Sierra's camera-domain validation set. */
    val reliability: Double = 1.0
) {
    init {
        require(id.isNotBlank() && modelId.isNotBlank() && inputSignature.isNotBlank())
        require(produces.isNotEmpty())
        require(supportedBackends.isNotEmpty())
        require(reliability in 0.0..1.0)
    }
}

data class ExpertDispatch(
    val expert: ExpertDescriptor,
    val execution: ExecutionDecision,
    val valueOfInformation: Double
)

/**
 * Selects the expert that maximizes expected information value per measured device cost. A current,
 * high-confidence belief suppresses redundant inference even when a heavyweight oracle is available.
 */
class ExpertScheduler(private val hypervisor: ComputeHypervisor = ComputeHypervisor()) {
    fun choose(
        need: InformationNeed,
        current: Belief<*>?,
        experts: List<ExpertDescriptor>,
        profiles: List<BackendExecutionProfile>,
        pressure: RuntimePressure,
        nowNs: Long
    ): ExpertDispatch? {
        val voi = valueOfInformation(need, current, nowNs)
        if (voi <= 0.0) return null

        return experts.asSequence()
            .filter { need.key in it.produces }
            .mapNotNull { expert ->
                val request = InferenceRequest(
                    requestId = "${need.key.value}:${expert.id}:$nowNs",
                    modelId = expert.modelId,
                    inputSignature = expert.inputSignature,
                    valueOfInformation = (voi * expert.reliability).coerceIn(0.0, 1.0),
                    priority = need.priority,
                    deadlineMs = need.deadlineMs,
                    requiredBackends = expert.supportedBackends
                )
                hypervisor.choose(request, profiles, pressure)?.let { decision ->
                    ExpertDispatch(expert, decision, request.valueOfInformation)
                }
            }
            .maxWithOrNull(
                compareBy<ExpertDispatch> { it.execution.utilityScore }
                    .thenByDescending { -it.execution.predictedLatencyMs }
            )
    }

    fun valueOfInformation(need: InformationNeed, current: Belief<*>?, nowNs: Long): Double {
        if (current == null) return need.finalPhotoImpact
        val effective = current.effectiveConfidence(nowNs)
        val age = (nowNs - current.timestampNs).coerceAtLeast(0L)
        if (effective >= need.requiredConfidence && age <= need.maxAgeNs) return 0.0

        val confidenceGap = ((need.requiredConfidence - effective) / need.requiredConfidence.coerceAtLeast(0.01))
            .coerceIn(0.0, 1.0)
        val staleness = (age.toDouble() / need.maxAgeNs.toDouble()).coerceIn(0.0, 2.0) / 2.0
        val uncertainty = (current.uncertainty / (1.0 + current.uncertainty)).coerceIn(0.0, 1.0)
        return (need.finalPhotoImpact * maxOf(confidenceGap, staleness, uncertainty * 0.75)).coerceIn(0.0, 1.0)
    }
}
