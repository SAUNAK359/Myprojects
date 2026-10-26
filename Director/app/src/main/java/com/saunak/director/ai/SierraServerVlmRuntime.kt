package com.saunak.director.ai

import android.content.Context
import android.graphics.Bitmap
import com.saunak.director.backend.SierraJobDto
import com.saunak.director.backend.SierraServerClient
import com.saunak.director.camera.CameraSnapshot
import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.director.EvaluationOutput
import com.saunak.director.director.PlanOutput
import com.saunak.director.director.ToolCall
import com.saunak.director.processing.PhotoQuality
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.UUID
import kotlin.math.max
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

interface SierraCloudJobRuntime {
    suspend fun completeFinalCapture(captureId: String): Result<Unit>
    suspend fun retryPendingSettlement(): Result<Unit>
    fun cancelCloudJob()
}

class SierraServerVlmRuntime(
    context: Context,
    private val server: SierraServerClient = SierraServerClient(context),
    private val deviceSkill: DeviceConfigSkill = DeviceConfigSkill(context)
) : VlmRuntime, SierraCloudJobRuntime {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val settlementPrefs = context.getSharedPreferences("sierra_ai_settlement_v1", Context.MODE_PRIVATE)
    private val geminiKeyStore = GeminiApiKeyStore(context)
    private val directGemini = GeminiDirectVlmRuntime(context, keyStore = geminiKeyStore)

    @Volatile private var maxImageDimension = 704
    @Volatile private var activeJobId: String? = null

    override suspend fun load(modelPath: String, projectorPath: String): Result<Unit> =
        if (isReady()) Result.success(Unit) else Result.failure(IllegalStateException("Add a Gemini API key in Sierra or sign in to Sierra Vision"))

    override suspend fun plan(image: Bitmap, capabilities: CapabilityRegistry, camera: CameraSnapshot, skillContext: String): Result<PlanOutput> {
        if (geminiKeyStore.isConfigured()) {
            directGemini.setMaxImageDimension(maxImageDimension)
            return directGemini.plan(image, capabilities, camera, skillContext)
        }
        if (!server.configured || !server.authenticated) return Result.failure(IllegalStateException("Sierra AI requires a Gemini API key or signed-in Sierra Server session"))
        return try {
            val capabilityJson = readCapabilitySnapshot(capabilities, camera)
            val localQualityJson = buildQualityJson(image)
            val cameraContextJson = buildCameraContextJson(capabilities, camera)
            val created = server.createAnalysisJob(
                imageJpeg = encode(image), cameraContextJson = cameraContextJson,
                deviceCapabilitiesJson = capabilityJson, localQualityJson = localQualityJson,
                skillContext = skillContext, idempotencyKey = UUID.randomUUID().toString()
            ).getOrThrow()
            activeJobId = created.id
            Result.success(VlmContractValidator.plan(toPlanOutput(awaitReady(created))))
        } catch (t: Throwable) {
            activeJobId = null
            Result.failure(t)
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
    ): Result<EvaluationOutput> {
        return if (geminiKeyStore.isConfigured()) {
            directGemini.setMaxImageDimension(maxImageDimension)
            directGemini.evaluate(image, objective, attempt, camera, previousIssues, evaluationContext, capabilities)
        } else Result.failure(UnsupportedOperationException("Sierra Server final review is not enabled for this beta path"))
    }

    override suspend fun completeFinalCapture(captureId: String): Result<Unit> {
        if (geminiKeyStore.isConfigured()) return Result.success(Unit)
        val jobId = activeJobId ?: return Result.success(Unit)
        activeJobId = null
        settlementPrefs.edit().putString(KEY_PENDING_JOB, jobId).putString(KEY_PENDING_CAPTURE, captureId).apply()
        return retryPendingSettlement()
    }

    override suspend fun retryPendingSettlement(): Result<Unit> {
        val jobId = settlementPrefs.getString(KEY_PENDING_JOB, null) ?: return Result.success(Unit)
        val captureId = settlementPrefs.getString(KEY_PENDING_CAPTURE, null) ?: return Result.success(Unit)
        var lastError: Throwable? = null
        repeat(3) { attempt ->
            val result = server.complete(jobId, captureId)
            if (result.isSuccess) {
                settlementPrefs.edit().remove(KEY_PENDING_JOB).remove(KEY_PENDING_CAPTURE).apply()
                return Result.success(Unit)
            }
            lastError = result.exceptionOrNull()
            delay(250L * (1 shl attempt))
        }
        return Result.failure(lastError ?: IllegalStateException("Sierra credit settlement is pending"))
    }

    override fun cancelCloudJob() {
        directGemini.cancel()
        val jobId = activeJobId ?: return
        activeJobId = null
        scope.launch { server.cancel(jobId) }
    }

    override fun setMaxImageDimension(pixels: Int) {
        maxImageDimension = pixels.coerceIn(384, 896)
        directGemini.setMaxImageDimension(maxImageDimension)
    }

    override fun cancel() = cancelCloudJob()
    override fun isReady(): Boolean = geminiKeyStore.isConfigured() || (server.configured && server.authenticated)
    override fun waitForSemanticPlan(): Boolean = geminiKeyStore.isConfigured()
    override fun supportsFinalReview(): Boolean = geminiKeyStore.isConfigured()
    override fun diagnostics(): VlmDiagnostics = if (geminiKeyStore.isConfigured()) directGemini.diagnostics() else VlmDiagnostics("sierra_server")
    override fun unload() = cancelCloudJob()
    override fun close() { cancelCloudJob(); directGemini.close(); scope.cancel() }

    private suspend fun readCapabilitySnapshot(capabilities: CapabilityRegistry, camera: CameraSnapshot): String = withContext(Dispatchers.IO) {
        deviceSkill.refresh(capabilities, camera)
        val file = File(deviceSkill.machineFilePath())
        if (file.isFile) file.readText() else "{}"
    }

    private fun buildQualityJson(image: Bitmap): String {
        val quality = PhotoQuality.analyze(image)
        return JSONObject().apply {
            put("mean_luma", quality.meanLuma); put("luma_std_dev", quality.lumaStdDev)
            put("shadow_p05", quality.shadowP05); put("median_luma", quality.medianLuma)
            put("highlight_p95", quality.highlightP95); put("top_mean_luma", quality.topMeanLuma)
            put("bottom_mean_luma", quality.bottomMeanLuma); put("top_bottom_delta", quality.topMeanLuma - quality.bottomMeanLuma)
            put("highlight_clip", quality.brightClipFraction); put("shadow_clip", quality.darkClipFraction)
            put("edge_energy", quality.edgeEnergy); put("horizontal_edge_energy", quality.horizontalEdgeEnergy)
            put("vertical_edge_energy", quality.verticalEdgeEnergy); put("strong_edge_fraction", quality.strongEdgeFraction)
            put("heuristic_score", quality.heuristicScore); put("catastrophic", quality.catastrophic)
        }.toString()
    }

    private fun buildCameraContextJson(capabilities: CapabilityRegistry, camera: CameraSnapshot): String = JSONObject().apply {
        put("snapshot", camera.compactDescription()); put("lens_facing", capabilities.lensFacing)
        put("zoom", camera.zoomRatio); put("exposure_index", camera.exposureCompensationIndex); put("extension", camera.extension.wireName)
    }.toString()

    private suspend fun awaitReady(created: SierraJobDto): SierraJobDto {
        var job = created
        repeat(MAX_POLL_ATTEMPTS) {
            when (job.state) {
                "ready" -> return job
                "failed", "cancelled", "expired" -> throw IllegalStateException(job.error_message ?: "Sierra AI job ${job.state}")
                "completed" -> throw IllegalStateException("Sierra AI job completed before camera capture")
            }
            delay(POLL_INTERVAL_MS)
            job = server.job(created.id).getOrThrow()
        }
        throw IllegalStateException("Sierra AI job timed out")
    }

    private fun toPlanOutput(job: SierraJobDto): PlanOutput {
        val dto = checkNotNull(job.plan) { "Sierra Server returned ready without a plan" }
        return PlanOutput(
            scene = dto.scene, objective = dto.objective, skillBranch = dto.branch,
            referenceScore = dto.reference_score, referenceAccept = dto.reference_accept,
            toolCalls = dto.calls.map { ToolCall(id = it.id, value = it.value, x = it.x, y = it.y, stringValue = it.stringValue) },
            recapture = dto.recapture, aspectRatio = dto.aspect_ratio,
            cropCenterX = dto.crop_center_x, cropCenterY = dto.crop_center_y, planner = "sierra_ai_server"
        )
    }

    private fun encode(source: Bitmap): ByteArray {
        val longest = max(source.width, source.height)
        val output = if (longest > maxImageDimension) {
            val scale = maxImageDimension.toFloat() / longest
            Bitmap.createScaledBitmap(source, (source.width * scale).toInt().coerceAtLeast(1), (source.height * scale).toInt().coerceAtLeast(1), true)
        } else source
        return try { ByteArrayOutputStream().use { stream -> check(output.compress(Bitmap.CompressFormat.JPEG, 76, stream)); stream.toByteArray() } }
        finally { if (output !== source && !output.isRecycled) output.recycle() }
    }

    companion object {
        private const val KEY_PENDING_JOB = "pending_job_id"
        private const val KEY_PENDING_CAPTURE = "pending_capture_id"
        private const val MAX_POLL_ATTEMPTS = 80
        private const val POLL_INTERVAL_MS = 120L
    }
}
