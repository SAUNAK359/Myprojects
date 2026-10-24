package com.saunak.director.director

import com.saunak.director.tools.ToolIds

/**
 * Deterministic semantic safety layer between a cloud plan and camera execution.
 * It prevents portrait-only behavior from leaking into clearly environmental scenes.
 */
object SierraPlanReconciler {
    private val portraitBranches = setOf("portrait", "selfie", "group")
    private val environmentalBranches = setOf("landscape", "architecture")
    private val portraitLooks = setOf(ToolIds.LOOK_SOFT_PORTRAIT, ToolIds.LOOK_CLEAN_PORTRAIT)

    fun reconcile(cloud: PlanOutput, local: PlanOutput, signals: SierraSceneSignals): PlanOutput {
        val cloudBranch = cloud.skillBranch.lowercase()
        val localBranch = local.skillBranch.lowercase()
        val forceEnvironmentalSemantics = !signals.primaryHumanSubject &&
            signals.confidence >= 0.72 &&
            localBranch in environmentalBranches &&
            (cloudBranch in portraitBranches || cloudBranch == "general")

        val resolvedBranch = if (forceEnvironmentalSemantics) local.skillBranch else cloud.skillBranch
        val environmental = resolvedBranch.lowercase() in environmentalBranches
        val calls = cloud.toolCalls.filterNot { call ->
            val extension = call.stringValue?.lowercase()
            when {
                !signals.primaryHumanSubject && call.id in portraitLooks -> true
                !signals.primaryHumanSubject && call.id == ToolIds.EXTENSION && extension in setOf("bokeh", "face_retouch") -> true
                signals.brightSky && environmental && call.id in setOf(ToolIds.TORCH, ToolIds.FLASH) -> true
                signals.brightSky && environmental && call.id == ToolIds.EXPOSURE_EV && (call.value ?: 0.0) > 0.0 -> true
                else -> false
            }
        }.toMutableList()

        if (calls.none { it.id in ToolIds.LOOK_IDS }) {
            local.toolCalls.lastOrNull { it.id in ToolIds.LOOK_IDS }?.let(calls::add)
        }

        val hasDynamicRangeCorrection = calls.any {
            it.id == ToolIds.EXPOSURE_EV ||
                (it.id == ToolIds.EXTENSION && it.stringValue?.lowercase() == "hdr")
        }
        if (signals.brightSky && environmental && !hasDynamicRangeCorrection) {
            local.toolCalls.firstOrNull {
                it.id == ToolIds.EXPOSURE_EV ||
                    (it.id == ToolIds.EXTENSION && it.stringValue?.lowercase() == "hdr")
            }?.let(calls::add)
        }

        return cloud.copy(
            scene = if (forceEnvironmentalSemantics) local.scene else cloud.scene,
            objective = if (forceEnvironmentalSemantics) local.objective else cloud.objective,
            skillBranch = resolvedBranch,
            toolCalls = calls.take(8)
        )
    }
}
