package com.saunak.director

import android.app.Application
import android.content.ComponentCallbacks2
import com.saunak.director.platform.MemoryPressureBus

class DirectorApplication : Application() {
    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        MemoryPressureBus.dispatch(level)
    }

    @Deprecated("Deprecated in Android; retained as a last-resort process pressure signal")
    override fun onLowMemory() {
        super.onLowMemory()
        MemoryPressureBus.dispatch(ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL)
    }
}
