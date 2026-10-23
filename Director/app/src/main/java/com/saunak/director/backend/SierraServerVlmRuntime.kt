package com.saunak.director.backend

import android.content.Context
import android.graphics.Bitmap
import com.saunak.director.ai.VlmRuntime
import com.saunak.director.camera.CameraSnapshot
import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.director.EvaluationOutput
import com.saunak.director.director.PlanOutput
import com.saunak.director.processing.TechnicalQuality
import com.saunak.director.tools.ToolCatalog
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.UUID
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.max

class SierraServerVlmRuntime(
    context: Context,
    private val backend: SierraBackendClient
) : VlmRuntime {
    private val appContext = context.applicationContext
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val activeJobId = AtomicReference<String?>(null)
    private val prefs = appContext.getSharedPreferences("sierra_cloud_jobs_v1", Context.MODE_PRIVATE)
    @Volatile private var maxImageDimension = 704

    override suspend fun load(modelPath: String, projectorPath: String): Result<Unit> =
        if (isReady()) Result.success(Unit)
        else Result.failure(IllegalStateException("Sign in to Sierra Vision to use Sierra AI"))

    override suspend fun plan(
        image: Bitmap,
        capabilities: CapabilityRegistry,
        camera: CameraSnapshot,
        skillContext: String
    ): Result<PlanOutput> = plan(image, capabilities, camera, skillContext, null)

    override suspend fun plan(
        image: Bitmap,
        capabilities: CapabilityRegistry,
        camera: CameraSnapshot,
        skillContext: String,
        localQuality: TechnicalQuality?
    ): Result<PlanOutput> = preservingCancellation {
        retryPendingCompletion()
        val jpeg = withContext(Dispatchers.Default) { encodeForServer(image) }
        val request = SierraAnalysisJobRequest(
            imageJpeg = jpeg,
            idempotencyKey = UUID.randomUUID().toString(),
            cameraContextJson = cameraJson(camera),
            deviceCapabilitiesJson = capabilitiesJson(capabilities),
            localQualityJson = qualityJson(localQuality),
            skillContext = skillContext
        )

        var current = backend.createAnalysisJob(request).getOrThrow()
        activeJobId.set(current.id)
        try {
            while (current.state == SierraJobState.QUEUED || current.state == SierraJobState.PROCESSING) {
                delay(POLL_INTERVAL_MS)
                current = backend.job(current.id).getOrThrow()
            }
            when (current.state) {
                SierraJobState.READY -> current.plan
                    ?: throw IOException("Sierra server returned a ready job without a photography plan")
                SierraJobState.FAILED -> throw IOException(current.errorMessage ?: "Sierra AI analysis failed")
                SierraJobState.EXPIRED -> throw IOException("Sierra AI preparation expired")
                SierraJobState.CANCELLED -> throw IOException("Sierra AI preparation was cancelled")
                SierraJobState.COMPLETED -> current.plan
                    ?: throw IOException("Sierra job completed without a photography plan")
                else -> throw IOException("Unexpected Sierra job state: ${current.state}")
            }
        } catch (cancelled: CancellationException) {
            cancelReservation(current.id)
            throw cancelled
        } catch (t: Throwable) {
            cancelReservation(current.id)
            throw t
        }
    }

    override suspend fun evaluate(
        image: Bitmap,
        objective: String,
        attempt: Int,
        camera: CameraSnapshot,
        previousIssues: List<String>,
        evaluationContext: String,
        capabilities: CapabilityRegistry
    ): Result<EvaluationOutput> = Result.failure(
        UnsupportedOperationException("Sierra Vision performs final technical QA locally with Sierra beta")
    )

    override suspend fun confirmFinalCapture(captureId: String): Result<Unit> = preservingCancellation {
        val jobId = activeJobId.get() ?: prefs.getString(KEY_PENDING_JOB, null) ?: return@preservingCancellation Unit
        val safeCaptureId = captureId.take(120)
        require(safeCaptureId.length >= 8) { "capture id is invalid" }
        prefs.edit()
            .putString(KEY_PENDING_JOB, jobId)
            .putString(KEY_PENDING_CAPTURE, safeCaptureId)
            .apply()
        backend.complete(jobId, safeCaptureId).getOrThrow()
        clearPending(jobId)
    }

    suspend fun retryPendingCompletion(): Result<Unit> = preservingCancellation {
        val jobId = prefs.getString(KEY_PENDING_JOB, null) ?: return@preservingCancellation Unit
        val captureId = prefs.getString(KEY_PENDING_CAPTURE, null) ?: return@preservingCancellation Unit
        val result = backend.complete(jobId, captureId)
        if (result.isSuccess) clearPending(jobId)
        result.getOrThrow()
        Unit
    }

    override fun setMaxImageDimension(pixels: Int) {
        maxImageDimension = pixels.coerceIn(512, 896)
    }

    override fun cancel() {
        val jobId = activeJobId.getAndSet(null) ?: return
        cancelReservation(jobId)
    }

    override fun isReady(): Boolean = backend.configured && backend.authenticated

    override fun unload() = Unit

    override fun close() {
        cancel()
        backend.close()
    }

    private fun cancelReservation(jobId: String) {
        scope.launch {
            runCatching { backend.cancel(jobId) }
        }
    }

    private fun clearPending(jobId: String) {
        activeJobId.compareAndSet(jobId, null)
        if (prefs.getString(KEY_PENDING_JOB, null) == jobId) {
            prefs.edit().remove(KEY_PENDING_JOB).remove(KEY_PENDING_CAPTURE).apply()
        }
    }

    private fun cameraJson(camera: CameraSnapshot): String = buildJsonObject {
        put("lens_facing", camera.lensFacing)
        put("zoom_ratio", camera.zoomRatio)
        put("exposure_compensation_index", camera.exposureCompensationIndex)
        put("torch_enabled", camera.torchEnabled)
        put("flash_mode", camera.flashMode.wireName)
        put("extension", camera.extension.wireName)
        put("rotation_degrees", camera.rotationDegrees)
    }.toString()

    private fun capabilitiesJson(value: CapabilityRegistry): String = buildJsonObject {
        put("lens_facing", value.lensFacing)
        put("min_zoom", value.minZoom)
        put("max_zoom", value.maxZoom)
        put("min_exposure_ev", value.minExposureEv)
        put("max_exposure_ev", value.maxExposureEv)
        put("exposure_step_ev", value.exposureStepEv)
        put("has_flash", value.hasFlash)
        put("torch_supported", value.torchSupported)
        put("focus_metering_supported", value.focusMeteringSupported)
        put("manual_sensor_supported", value.manualSensorSupported)
        put("raw_supported", value.rawSupported)
        value.isoMin?.let { put("iso_min", it) }
        value.isoMax?.let { put("iso_max", it) }
        value.exposureTimeMinNs?.let { put("exposure_time_min_ns", it) }
        value.exposureTimeMaxNs?.let { put("exposure_time_max_ns", it) }
        value.minimumFocusDistance?.let { put("minimum_focus_distance", it) }
        put("focal_lengths_mm", buildJsonArray { value.focalLengthsMm.forEach { add(it) } })
        put("extensions", buildJsonArray { value.extensions.forEach { add(it.wireName) } })
        put("allowed_tool_ids", buildJsonArray { value.allowedToolIds.sorted().forEach { add(it) } })
        put("active_extension", value.activeExtension.wireName)
        put("tool_contract", ToolCatalog.contract(value, value.activeExtension))
    }.toString()

    private fun qualityJson(value: TechnicalQuality?): String = if (value == null) "{}" else buildJsonObject {
        put("mean_luma", value.meanLuma)
        put("luma_std_dev", value.lumaStdDev)
        put("dark_clip_fraction", value.darkClipFraction)
        put("bright_clip_fraction", value.brightClipFraction)
        put("edge_energy", value.edgeEnergy)
        put("catastrophic", value.catastrophic)
        put("heuristic_score", value.heuristicScore)
    }.toString()

    private fun encodeForServer(source: Bitmap): ByteArray {
        require(!source.isRecycled && source.width > 0 && source.height > 0)
        val longest = max(source.width, source.height)
        val prepared = if (longest > maxImageDimension) {
            val scale = maxImageDimension.toFloat() / longest.toFloat()
            Bitmap.createScaledBitmap(
                source,
                maxOf(1, (source.width * scale).toInt()),
                maxOf(1, (source.height * scale).toInt()),
                true
            )
        } else source
        return try {
            ByteArrayOutputStream(180 * 1024).use { out ->
                check(prepared.compress(Bitmap.CompressFormat.JPEG, 76, out)) {
                    "Could not encode Sierra analysis frame"
                }
                out.toByteArray()
            }
        } finally {
            if (prepared !== source && !prepared.isRecycled) prepared.recycle()
        }
    }

    private suspend fun <T> preservingCancellation(block: suspend () -> T): Result<T> = try {
        Result.success(block())
    } catch (t: CancellationException) {
        throw t
    } catch (t: Throwable) {
        Result.failure(t)
    }

    companion object {
        private const val POLL_INTERVAL_MS = 120L
        private const val KEY_PENDING_JOB = "pending_completion_job"
        private const val KEY_PENDING_CAPTURE = "pending_completion_capture"
    }
}
