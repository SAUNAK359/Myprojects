package com.saunak.director.engine.runtime

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import com.saunak.director.engine.device.ComputeBackend
import com.saunak.director.engine.models.InstalledModelArtifact
import java.util.EnumSet
import java.util.LinkedHashMap
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/** Backends actually compiled into this application runtime. Candidate hardware in DeviceTwin is
 * intentionally separate: hardware presence does not imply the shipped ORT binary supports its EP. */
data class OnnxRuntimeCapabilities(
    val xnnpack: Boolean = true,
    val nnapi: Boolean = true,
    val qnn: Boolean = false
)

data class OnnxSessionKey(
    val modelId: String,
    val version: String,
    val backend: ComputeBackend
)

class OnnxSessionHandle internal constructor(
    val key: OnnxSessionKey,
    private val options: OrtSession.SessionOptions,
    private val session: OrtSession
) : AutoCloseable {
    val inputNames: Set<String> get() = session.inputNames
    val outputNames: Set<String> get() = session.outputNames
    val inputInfo get() = session.inputInfo
    val outputInfo get() = session.outputInfo

    /** Result values are valid only inside [block]; ORT resources are always released afterward. */
    fun <T> run(inputs: Map<String, OnnxTensor>, block: (OrtSession.Result) -> T): T {
        require(inputs.isNotEmpty()) { "At least one input tensor is required" }
        session.run(inputs).use { result -> return block(result) }
    }

    override fun close() {
        session.close()
        options.close()
    }
}

/**
 * Small LRU of warm ORT sessions. It intentionally supports only portable EPs in the stock AAR;
 * vendor runtimes are separate adapters so a missing Qualcomm/MediaTek/Samsung library cannot make
 * the application fail to load.
 */
class OnnxRuntimeEngine(
    private val capabilities: OnnxRuntimeCapabilities = OnnxRuntimeCapabilities(),
    private val maxWarmSessions: Int = 3,
    private val intraOpThreads: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
) : AutoCloseable {
    init { require(maxWarmSessions in 1..8); require(intraOpThreads in 1..16) }

    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment("sierra")
    private val lock = ReentrantLock()
    private val sessions = object : LinkedHashMap<OnnxSessionKey, OnnxSessionHandle>(8, 0.75f, true) {}

    fun getOrCreate(artifact: InstalledModelArtifact, backend: ComputeBackend): OnnxSessionHandle = lock.withLock {
        val manifest = artifact.manifest
        require(backend in manifest.supportedBackends) { "$backend is not declared for ${manifest.id}" }
        val key = OnnxSessionKey(manifest.id, manifest.version, backend)
        sessions[key]?.let { return it }

        val options = createOptions(backend)
        try {
            val session = environment.createSession(artifact.file.absolutePath, options)
            val handle = OnnxSessionHandle(key, options, session)
            sessions[key] = handle
            evictIfNeeded(except = key)
            handle
        } catch (t: Throwable) {
            options.close()
            throw t
        }
    }

    fun unload(modelId: String) = lock.withLock {
        val keys = sessions.keys.filter { it.modelId == modelId }
        keys.forEach { sessions.remove(it)?.close() }
    }

    fun trimTo(maxSessions: Int) = lock.withLock {
        require(maxSessions >= 0)
        while (sessions.size > maxSessions) {
            val eldest = sessions.entries.firstOrNull() ?: break
            sessions.remove(eldest.key)?.close()
        }
    }

    private fun createOptions(backend: ComputeBackend): OrtSession.SessionOptions {
        val options = OrtSession.SessionOptions()
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        options.setIntraOpNumThreads(intraOpThreads)
        options.setInterOpNumThreads(1)
        options.setMemoryPatternOptimization(true)
        options.setCPUArenaAllocator(true)
        when (backend) {
            ComputeBackend.CPU_XNNPACK -> {
                require(capabilities.xnnpack) { "XNNPACK is disabled for this runtime" }
                options.addXnnpack(emptyMap())
                // CPU remains ORT's final correctness fallback for operators XNNPACK does not claim.
            }
            ComputeBackend.NNAPI -> {
                require(capabilities.nnapi) { "NNAPI is disabled for this runtime" }
                // Disabling NNAPI's reference CPU avoids mistaking slow NNAPI-CPU execution for NPU/GPU acceleration.
                options.addNnapi(EnumSet.of(NNAPIFlags.CPU_DISABLED))
            }
            ComputeBackend.NPU_QNN -> {
                require(capabilities.qnn) { "QNN requires a Sierra ORT build compiled with Qualcomm QNN support" }
                throw UnsupportedOperationException("QNN provider options must be supplied by the vendor runtime adapter")
            }
            else -> throw UnsupportedOperationException("$backend is not provided by the portable ONNX runtime adapter")
        }
        return options
    }

    private fun evictIfNeeded(except: OnnxSessionKey) {
        while (sessions.size > maxWarmSessions) {
            val eldest = sessions.entries.firstOrNull { it.key != except } ?: break
            sessions.remove(eldest.key)?.close()
        }
    }

    override fun close() = lock.withLock {
        sessions.values.toList().forEach { it.close() }
        sessions.clear()
        // OrtEnvironment is process-global; do not close it here while other components may use it.
    }
}
