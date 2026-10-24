package com.saunak.director.engine

import android.content.ComponentCallbacks2
import android.content.Context
import com.saunak.director.engine.belief.SceneBeliefGraph
import com.saunak.director.engine.device.BackendExecutionProfile
import com.saunak.director.engine.device.DeviceTwin
import com.saunak.director.engine.device.DeviceTwinBuilder
import com.saunak.director.engine.experts.ExpertScheduler
import com.saunak.director.engine.geometry.CoordinateGraph
import com.saunak.director.engine.models.ModelArtifactStore
import com.saunak.director.engine.models.ModelRegistry
import com.saunak.director.engine.models.SierraShippingManifests
import com.saunak.director.engine.motion.MotionFusionEngine
import com.saunak.director.engine.motion.ShutterMotionProbe
import com.saunak.director.engine.quality.QualityCouncil
import com.saunak.director.engine.reconstruction.DeterministicBurstPipeline
import com.saunak.director.engine.reconstruction.EvidenceLedger
import com.saunak.director.engine.runtime.BackendProfiler
import com.saunak.director.engine.runtime.ComputeHypervisor
import com.saunak.director.engine.runtime.OnnxRuntimeEngine
import com.saunak.director.engine.runtime.RuntimeProfileStore
import com.saunak.director.engine.subject.SubjectTracker
import java.util.concurrent.atomic.AtomicReference

/** Application-scoped ownership boundary for Sierra camera intelligence and image formation. */
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
    val deterministicBurst = DeterministicBurstPipeline(evidence)
    val runtimeProfiles = RuntimeProfileStore(appContext)
    val backendProfiler = BackendProfiler(appContext, runtimeProfiles, RUNTIME_VERSION)

    // Temporal state is application-scoped so history survives individual shutter requests.
    val motion = MotionFusionEngine(appContext, beliefs)
    val motionProbe = ShutterMotionProbe(appContext, motion)
    val subjects = SubjectTracker()

    private val onnxLazy = lazy(LazyThreadSafetyMode.SYNCHRONIZED) { OnnxRuntimeEngine() }
    val onnx: OnnxRuntimeEngine get() = onnxLazy.value

    init { SierraShippingManifests.registerAll(models) }

    fun deviceTwin(refresh: Boolean = false): Result<DeviceTwin> {
        if (!refresh) twinRef.get()?.let { return Result.success(it) }
        return runCatching { DeviceTwinBuilder(appContext).build() }
            .onSuccess(twinRef::set)
    }

    fun currentDeviceTwin(): DeviceTwin? = twinRef.get()

    fun measuredProfiles(modelId: String, inputSignature: String): List<BackendExecutionProfile> {
        val fingerprint = currentDeviceTwin()?.fingerprint ?: android.os.Build.FINGERPRINT.orEmpty().ifBlank { "unknown" }
        return runtimeProfiles.find(fingerprint, modelId, inputSignature).map { it.profile }
    }

    fun profileFingerprint(): String = currentDeviceTwin()?.fingerprint
        ?: android.os.Build.FINGERPRINT.orEmpty().ifBlank { "unknown" }

    fun health(): SierraEngineHealth {
        val twin = twinRef.get()
        return SierraEngineHealth(
            deviceTwinReady = twin != null,
            cameraCount = twin?.cameras?.size ?: 0,
            registeredModels = models.all().size,
            provisionedModels = models.all().count { modelArtifacts.findVerified(it) != null },
            beliefCount = beliefs.snapshot(System.nanoTime()).size,
            evidenceRegions = evidence.snapshot().size,
            measuredBackendProfiles = runtimeProfiles.snapshot().count { it.valid },
            onnxInitialized = onnxLazy.isInitialized(),
            subjectTrackReady = subjects.latest() != null
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
        motion.close()
        subjects.clear()
        if (onnxLazy.isInitialized()) onnx.close()
        evidence.clear()
        coordinates.clear()
        twinRef.set(null)
    }

    companion object {
        private const val RUNTIME_VERSION = "onnxruntime-1.31.0+sierra-v6-subject"
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
    val measuredBackendProfiles: Int,
    val onnxInitialized: Boolean,
    val subjectTrackReady: Boolean
)
