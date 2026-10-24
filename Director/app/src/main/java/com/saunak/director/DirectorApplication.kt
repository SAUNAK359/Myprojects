package com.saunak.director

import android.app.Application
import android.content.ComponentCallbacks2
import com.saunak.director.engine.SierraEngine
import com.saunak.director.engine.experts.YuNetFaceExpert
import com.saunak.director.engine.models.SierraModelProvisioning
import com.saunak.director.platform.MemoryPressureBus
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

class DirectorApplication : Application() {
    private val appScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onCreate() {
        super.onCreate()
        val engine = SierraEngine.initialize(this)

        // Hot expert preparation is failure-isolated and never blocks app/camera startup. The
        // artifact store verifies HTTPS, size, free space, licence and SHA-256 before it becomes usable.
        appScope.launch {
            runCatching { engine.deviceTwin() }
            runCatching { YuNetFaceExpert(engine).prepare() }
        }

        // Large WARM/CAPTURE packs are persistent deferrable work: unmetered network, battery-not-low
        // and storage-not-low. They can never introduce first-shutter network latency.
        runCatching { SierraModelProvisioning.scheduleProductionPacks(this, engine.models) }
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

    override fun onTerminate() {
        appScope.cancel()
        SierraEngine.getOrNull()?.close()
        super.onTerminate()
    }
}
