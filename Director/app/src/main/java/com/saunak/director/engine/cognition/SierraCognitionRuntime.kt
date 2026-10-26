package com.saunak.director.engine.cognition

import android.content.Context
import com.saunak.director.engine.state.SceneState
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

class SierraCognitionRuntime(context: Context) {
    private val manifest: CognitionManifest = CognitionManifestLoader.load(context.applicationContext)
    private val reliabilityPlane = ReliabilityPlane(manifest.policy)
    private val worldAssembler = PhotographicWorldAssembler(manifest.policy, reliabilityPlane)
    private val failureForecaster = FailureForecaster(manifest.policy)
    private val themes = ThemeCortex(manifest)
    private val cortex = PhotographicCortex(manifest, reliabilityPlane, themes)
    private val sequence = AtomicLong(0L)
    private val latestRef = AtomicReference<CognitionDecision?>(null)

    fun evaluate(state: SceneState, nowNs: Long = System.nanoTime()): CognitionDecision {
        val stateId = sequence.incrementAndGet()
        val world = worldAssembler.assemble(stateId, state, nowNs)
        val reliability = world.inputReliability
        val failures = failureForecaster.forecast(world, reliability)
        val theme = themes.select(world, failures, reliability)
        val intent = cortex.compose(world, failures, reliability, theme, nowNs)
        return CognitionDecision(world, reliability, failures, intent).also(latestRef::set)
    }

    fun latest(nowNs: Long = System.nanoTime()): CognitionDecision? {
        val decision = latestRef.get() ?: return null
        return decision.takeIf { nowNs <= it.intent.expiresAtNs && it.reliability.freshness > 0.0 }
    }

    fun manifestVersion(): Int = manifest.policy.schemaVersion
    fun themeCount(): Int = manifest.themes.size
}
