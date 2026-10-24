package com.saunak.director

import android.app.Application
import android.content.ComponentCallbacks2
import com.saunak.director.engine.SierraEngine
import com.saunak.director.platform.MemoryPressureBus

class DirectorApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        // Creates only lightweight orchestration objects. Device probing and neural sessions remain lazy.
        SierraEngine.initialize(this)
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        MemoryPressureBus.dispatch(level)
        SierraEngine.getOrNull()?.onTrimMemory(level)
    }

    @Deprecated("Deprecated in Android; retained as a last-resort process pressure signal")
    override fun onLowMemory() {
        super.onLowMemory()
        val level = ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL
        MemoryPressureBus.dispatch(level)
        SierraEngine.getOrNull()?.onTrimMemory(level)
    }
}
