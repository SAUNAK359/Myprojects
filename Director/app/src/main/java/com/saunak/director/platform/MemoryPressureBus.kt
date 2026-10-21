package com.saunak.director.platform

import java.util.concurrent.CopyOnWriteArraySet

object MemoryPressureBus {
    private val listeners = CopyOnWriteArraySet<(Int) -> Unit>()
    fun add(listener: (Int) -> Unit) { listeners += listener }
    fun remove(listener: (Int) -> Unit) { listeners -= listener }
    fun dispatch(level: Int) { listeners.forEach { runCatching { it(level) } } }
}
