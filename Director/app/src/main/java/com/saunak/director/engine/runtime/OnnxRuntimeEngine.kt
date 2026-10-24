package com.saunak.director.engine.runtime

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import com.saunak.director.engine.device.ComputeBackend
import com.saunak.director.engine.models.InstalledModelArtifact
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.EnumSet
import java.util.LinkedHashMap
import java.util.concurrent.locks.ReentrantLock
import java.util.concurrent.locks.ReentrantReadWriteLock
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

/**
 * A session owns native resources. Runs acquire a read lease; close acquires the exclusive write
 * lease. This prevents Android memory-pressure trimming or backend invalidation from closing ORT
 * native state while inference is executing.
 */
class OnnxSessionHandle internal constructor(
    val key: OnnxSessionKey,
    private val options: OrtSession.SessionOptions,
    private val session: OrtSession
) : AutoCloseable {
    private val lifecycle = ReentrantReadWriteLock(true)
    @Volatile private var closed = false

    val inputNames: Set<String> get() = withReadLease { session.inputNames }
    val outputNames: Set<String> get() = withReadLease { session.outputNames }
    val inputInfo get() = withReadLease { session.inputInfo }
    val outputInfo get() = withReadLease { session.outputInfo }

    /** Result values are valid only inside [block]; ORT resources are always released afterward. */
    fun <T> run(inputs: Map<String, OnnxTensor>, block: (OrtSession.Result) -> T): T = withReadLease {
        require(inputs.isNotEmpty()) { "At least one input tensor is required" }
        session.run(inputs).use { result -> block(result) }
    }

    private inline fun <T> withReadLease(block: () -> T): T {
        val lock = lifecycle.readLock()
        lock.lock()
        try {
            check(!closed) { "ONNX session ${key.modelId}/${key.backend} is closed" }
            return block()
        } finally {
            lock.unlock()
        }
    }

    override fun close() {
        val lock = lifecycle.writeLock()
        lock.lock()
        try {
            if (closed) return
            closed = true
            session.close()
            options.close()
        } finally {
            lock.unlock()
        }
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

    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
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

    /**
     * Runs one dense float input and returns owned copies of requested float outputs. This boundary
     * prevents ORT Result/OnnxTensor lifetimes from escaping into expert code and leaking native memory.
     */
    fun runFloatTensor(
        artifact: InstalledModelArtifact,
        backend: ComputeBackend,
        inputName: String,
        values: FloatArray,
        shape: LongArray,
        requestedOutputs: Set<String>
    ): Map<String, FloatArray> {
        require(values.isNotEmpty())
        require(shape.isNotEmpty() && shape.all { it > 0L })
        val expected = shape.fold(1L) { acc, dimension -> Math.multiplyExact(acc, dimension) }
        require(expected == values.size.toLong()) { "Input contains ${values.size} values but shape requires $expected" }
        require(requestedOutputs.isNotEmpty())

        val direct = ByteBuffer.allocateDirect(values.size * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        direct.put(values)
        direct.rewind()

        OnnxTensor.createTensor(environment, direct, shape).use { tensor ->
            val handle = getOrCreate(artifact, backend)
            require(inputName in handle.inputNames) { "Model ${artifact.manifest.id} has no input '$inputName'" }
            require(requestedOutputs.all { it in handle.outputNames }) {
                "Model ${artifact.manifest.id} is missing requested outputs: ${requestedOutputs - handle.outputNames}"
            }
            return handle.run(mapOf(inputName to tensor)) { result ->
                requestedOutputs.associateWith { name ->
                    val value = result.get(name).orElseThrow { IllegalStateException("Missing ORT output '$name'") }
                    val output = value as? OnnxTensor
                        ?: error("Output '$name' is ${value.javaClass.simpleName}, expected OnnxTensor")
                    val buffer: FloatBuffer = output.floatBuffer
                        ?: error("Output '$name' cannot be represented as float32")
                    val copy = FloatArray(buffer.remaining())
                    buffer.get(copy)
                    copy
                }
            }
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
            }
            ComputeBackend.NNAPI -> {
                require(capabilities.nnapi) { "NNAPI is disabled for this runtime" }
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
    }
}
