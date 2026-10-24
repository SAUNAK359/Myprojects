package com.saunak.director.engine

import android.content.ComponentCallbacks2
import android.content.Context
import com.saunak.director.engine.belief.SceneBeliefGraph
import com.saunak.director.engine.device.DeviceTwin
import com.saunak.director.engine.device.DeviceTwinBuilder
import com.saunak.director.engine.experts.ExpertScheduler
import com.saunak.director.engine.geometry.CoordinateGraph
import com.saunak.director.engine.models.ModelArtifactStore
import com.saunak.director.engine.models.ModelRegistry
import com.saunak.director.engine.models.SierraShippingManifests
import com.saunak.director.engine.quality.QualityCouncil
import com.saunak.director.engine.reconstruction.EvidenceLedger
import com.saunak.director.engine.runtime.ComputeHypervisor
import com.saunak.director.engine.runtime.OnnxRuntimeEngine
import java.util.concurrent.atomic.AtomicReference

/**
 * Application-scoped ownership boundary for Sierra's camera intelligence. Heavy model sessions are
 * lazy; constructing Director never allocates neural weights. The current CameraX capture path can
 * therefore coexist safely while RAW/reconstruction backends are introduced incrementally.
 */
class SierraEngine private constructor(context: Context) : AutoCloseable {
    private val appContext = context.applicationContext
    private val twinRef = AtomicReference<DeviceTwin?>(null)

    val beliefs = SceneBeliefGraph()
    val coordinates = CoordinateGraph()
    val models = ModelRegistry()
    val modelArtifacts = ModelArtifactStore(appContext)
    val hypervisor = ComputeHypervisor()
    val experts = ExpertScheduler(hypervisor)
    val evidence = EvidenceLedger()
    val quality = QualityCouncil()

    private val onnxLazy = lazy(LazyThreadSafetyMode.SYNCHRONIZED) { OnnxRuntimeEngine() }
    val onnx: OnnxRuntimeEngine get() = onnxLazy.value

    init {
        // Registration is lightweight metadata only. Artifact download/session creation remain lazy.
        SierraShippingManifests.registerAll(models)
    }

    fun deviceTwin(refresh: Boolean = false): Result<DeviceTwin> {
        if (!refresh) twinRef.get()?.let { return Result.success(it) }
        return runCatching { DeviceTwinBuilder(appContext).build() }
            .onSuccess(twinRef::set)
    }

    fun currentDeviceTwin(): DeviceTwin? = twinRef.get()

    fun health(): SierraEngineHealth {
        val twin = twinRef.get()
        return SierraEngineHealth(
            deviceTwinReady = twin != null,
            cameraCount = twin?.cameras?.size ?: 0,
            registeredModels = models.all().size,
            provisionedModels = models.all().count { modelArtifacts.findVerified(it) != null },
            beliefCount = beliefs.snapshot(System.nanoTime()).size,
            evidenceRegions = evidence.snapshot().size,
            onnxInitialized = onnxLazy.isInitialized()
        )
    }

    fun onTrimMemory(level: Int) {
        if (!onnxLazy.isInitialized()) return
        when {
            level >= ComponentCallbacks2.TRIM_MEMORY_COMPLETE -> onnx.trimTo(0)
            level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL -> onnx.trimTo(0)
            level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW -> onnx.trimTo(1)
            level >= ComponentCallbacks2.TRIM_MEMORY_BACKGROUND -> onnx.trimTo(1)
            level >= ComponentCallbacks2.TRIM_MEMORY_UI_HIDDEN -> onnx.trimTo(1)
        }
    }

    override fun close() {
        if (onnxLazy.isInitialized()) onnx.close()
        evidence.clear()
        coordinates.clear()
        twinRef.set(null)
    }

    companion object {
        @Volatile private var instance: SierraEngine? = null

        fun initialize(context: Context): SierraEngine = instance ?: synchronized(this) {
            instance ?: SierraEngine(context).also { instance = it }
        }

        fun get(): SierraEngine = checkNotNull(instance) { "SierraEngine has not been initialized" }
        fun getOrNull(): SierraEngine? = instance
    }
}

data class SierraEngineHealth(
    val deviceTwinReady: Boolean,
    val cameraCount: Int,
    val registeredModels: Int,
    val provisionedModels: Int,
    val beliefCount: Int,
    val evidenceRegions: Int,
    val onnxInitialized: Boolean
)
