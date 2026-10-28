package com.saunak.director.director

/**
 * Sierra/Gemini is the photographic authority. Sierra beta may validate physical/API feasibility
 * during execution, but it must not rewrite a valid cloud plan because local semantic heuristics
 * prefer another look, exposure, flash choice, branch, or scene interpretation.
 *
 * Kept as an explicit pass-through boundary so future local evidence can be attached as telemetry
 * without silently reintroducing creative overrides.
 */
object SierraPlanReconciler {
    fun reconcile(cloud: PlanOutput, local: PlanOutput, signals: SierraSceneSignals): PlanOutput {
        @Suppress("UNUSED_VARIABLE") val ignoredLocal = local
        @Suppress("UNUSED_VARIABLE") val ignoredSignals = signals
        return cloud
    }
}
