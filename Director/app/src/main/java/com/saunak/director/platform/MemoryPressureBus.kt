package com.saunak.director.platform

import android.content.ComponentCallbacks2
import java.util.concurrent.CopyOnWriteArraySet
import java.util.concurrent.atomic.AtomicInteger

object MemoryPressureBus {
    private val listeners = CopyOnWriteArraySet<(Int) -> Unit>()
    private val currentLevel = AtomicInteger(0)

    fun add(listener: (Int) -> Unit) { listeners += listener }
    fun remove(listener: (Int) -> Unit) { listeners -= listener }

    fun dispatch(level: Int) {
        currentLevel.set(level.coerceAtLeast(0))
        listeners.forEach { runCatching { it(level) } }
    }

    /** Capture-facing pressure in [0,1]. Android trim constants are not a simple severity ordinal. */
    @Suppress("DEPRECATION")
    fun normalizedPressure(): Double = when (currentLevel.get()) {
        ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL,
        ComponentCallbacks2.TRIM_MEMORY_COMPLETE -> 1.0
        ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW,
        ComponentCallbacks2.TRIM_MEMORY_MODERATE -> 0.78
        ComponentCallbacks2.TRIM_MEMORY_BACKGROUND -> 0.62
        ComponentCallbacks2.TRIM_MEMORY_UI_HIDDEN -> 0.38
        ComponentCallbacks2.TRIM_MEMORY_RUNNING_MODERATE -> 0.32
        else -> 0.0
    }
}
