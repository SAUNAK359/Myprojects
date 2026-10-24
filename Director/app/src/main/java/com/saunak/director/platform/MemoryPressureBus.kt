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

    /** Capture-facing pressure in [0,1], intentionally coarse and monotonic by Android severity. */
    fun normalizedPressure(): Double = when (val level = currentLevel.get()) {
        in ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL..Int.MAX_VALUE -> 1.0
        in ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW until ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL -> 0.78
        in ComponentCallbacks2.TRIM_MEMORY_BACKGROUND until ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW -> 0.55
        in ComponentCallbacks2.TRIM_MEMORY_UI_HIDDEN until ComponentCallbacks2.TRIM_MEMORY_BACKGROUND -> 0.35
        else -> if (level > 0) 0.18 else 0.0
    }
}
