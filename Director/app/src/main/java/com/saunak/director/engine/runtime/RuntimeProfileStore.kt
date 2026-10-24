package com.saunak.director.engine.runtime

import android.content.Context
import com.saunak.director.engine.device.BackendExecutionProfile
import com.saunak.director.engine.device.ComputeBackend
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write

data class ProfileKey(
    val deviceFingerprint: String,
    val modelId: String,
    val inputSignature: String,
    val backend: ComputeBackend
)

data class MeasuredExecutionProfile(
    val key: ProfileKey,
    val profile: BackendExecutionProfile,
    val samples: Int,
    val measuredAtEpochMs: Long,
    val runtimeVersion: String,
    val valid: Boolean = true
) {
    init {
        require(samples > 0)
        require(measuredAtEpochMs > 0)
        require(runtimeVersion.isNotBlank())
        require(profile.modelId == key.modelId)
        require(profile.inputSignature == key.inputSignature)
        require(profile.backend == key.backend)
    }
}

/** Atomic app-private persistence for backend measurements. */
class RuntimeProfileStore(context: Context) {
    private val file = File(context.filesDir, "sierra/runtime_profiles_v1.json")
    private val lock = ReentrantReadWriteLock()
    private val profiles = LinkedHashMap<ProfileKey, MeasuredExecutionProfile>()

    init { load() }

    fun upsert(profile: MeasuredExecutionProfile) = lock.write {
        profiles[profile.key] = profile
        persistLocked()
    }

    fun find(
        deviceFingerprint: String,
        modelId: String,
        inputSignature: String
    ): List<MeasuredExecutionProfile> = lock.read {
        profiles.values.filter {
            it.valid &&
                it.key.deviceFingerprint == deviceFingerprint &&
                it.key.modelId == modelId &&
                it.key.inputSignature == inputSignature
        }
    }

    fun invalidate(key: ProfileKey) = lock.write {
        val current = profiles[key] ?: return@write
        profiles[key] = current.copy(valid = false)
        persistLocked()
    }

    fun snapshot(): List<MeasuredExecutionProfile> = lock.read { profiles.values.toList() }

    private fun load() = lock.write {
        profiles.clear()
        if (!file.isFile) return@write
        runCatching {
            val root = JSONObject(file.readText())
            val array = root.optJSONArray("profiles") ?: JSONArray()
            for (i in 0 until array.length()) {
                val o = array.getJSONObject(i)
                val backend = runCatching { ComputeBackend.valueOf(o.getString("backend")) }.getOrNull() ?: continue
                val key = ProfileKey(
                    deviceFingerprint = o.getString("device_fingerprint"),
                    modelId = o.getString("model_id"),
                    inputSignature = o.getString("input_signature"),
                    backend = backend
                )
                val profile = BackendExecutionProfile(
                    modelId = key.modelId,
                    backend = backend,
                    inputSignature = key.inputSignature,
                    graphCoverage = o.getDouble("graph_coverage"),
                    transferBoundaries = o.getInt("transfer_boundaries"),
                    p50LatencyMs = o.getDouble("p50_ms"),
                    p95LatencyMs = o.getDouble("p95_ms"),
                    peakMemoryMb = o.getDouble("peak_memory_mb"),
                    sustainedPowerScore = o.getDouble("thermal_power_proxy")
                )
                val measured = MeasuredExecutionProfile(
                    key = key,
                    profile = profile,
                    samples = o.getInt("samples"),
                    measuredAtEpochMs = o.getLong("measured_at_ms"),
                    runtimeVersion = o.getString("runtime_version"),
                    valid = o.optBoolean("valid", true)
                )
                profiles[key] = measured
            }
        }.onFailure {
            // Corrupt/stale profiling data must never block camera startup.
            profiles.clear()
        }
    }

    private fun persistLocked() {
        file.parentFile?.mkdirs()
        val array = JSONArray()
        profiles.values.forEach { measured ->
            val p = measured.profile
            array.put(JSONObject().apply {
                put("device_fingerprint", measured.key.deviceFingerprint)
                put("model_id", measured.key.modelId)
                put("input_signature", measured.key.inputSignature)
                put("backend", measured.key.backend.name)
                put("graph_coverage", p.graphCoverage)
                put("transfer_boundaries", p.transferBoundaries)
                put("p50_ms", p.p50LatencyMs)
                put("p95_ms", p.p95LatencyMs)
                put("peak_memory_mb", p.peakMemoryMb)
                put("thermal_power_proxy", p.sustainedPowerScore)
                put("samples", measured.samples)
                put("measured_at_ms", measured.measuredAtEpochMs)
                put("runtime_version", measured.runtimeVersion)
                put("valid", measured.valid)
            })
        }
        val tmp = File(file.parentFile, "${file.name}.tmp")
        tmp.writeText(JSONObject().put("schema", 1).put("profiles", array).toString())
        if (!tmp.renameTo(file)) {
            file.writeText(tmp.readText())
            tmp.delete()
        }
    }
}
