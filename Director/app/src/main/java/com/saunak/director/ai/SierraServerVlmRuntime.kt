package com.saunak.director.ai

import android.content.Context
import android.graphics.Bitmap
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
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

interface SierraCloudJobRuntime {
    suspend fun completeFinalCapture(captureId: String): Result<Unit>
    fun cancelCloudJob()
}

/**
 * Sierra AI cloud runtime. The APK knows only the Sierra Server origin and user session.
 * Provider endpoints, model names and provider credentials never exist in this runtime.
 */
class SierraServerVlmRuntime(
    context: Context,
    private val server: SierraServerClient = SierraServerClient(context),
    private val deviceSkill: DeviceConfigSkill = DeviceConfigSkill(context)
) : VlmRuntime, SierraCloudJobRuntime {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    @Volatile private var maxImageDimension = 704
    @Volatile private var activeJobId: String? = null

    override suspend fun load(modelPath: String, projectorPath: String): Result<Unit> =
        if (isReady()) Result.success(Unit) else Result.failure(IllegalStateException("Sign in to Sierra Vision"))

    override suspend fun plan(
        image: Bitmap,
        capabilities: CapabilityRegistry,
        camera: CameraSnapshot,
        skillContext: String
    ): Result<PlanOutput> = runCatching {
        check(isReady()) { "Sierra AI requires a signed-in Sierra Server session" }
        val capabilitySkill = withContext(Dispatchers.IO) {
            deviceSkill.refresh(capabilities, camera)
            File(deviceSkill.machineFilePath()).takeIf { it.isFile }?.readText() ?: "{}"
        }
        val quality = PhotoQuality.analyze(image)
        val localQuality = JSONObject().apply {
            put("mean_luma", quality.meanLuma)
            put("highlight_clip", quality.highlightClip)
            put("shadow_clip", quality.shadowClip)
            put("detail", quality.detail)
            put("heuristic_score", quality.heuristicScore)
            put("catastrophic", quality.catastrophic)
        }.toString()
        val cameraContext = JSONObject().apply {
            put("snapshot", camera.compactDescription())
            put("lens_facing", capabilities.lensFacing)
            put("zoom", camera.zoom)
            put("exposure_index", camera.exposureIndex)
            put("extension", camera.extension.wireName)
        }.toString()

        val jpeg = encode(image)
        val created = server.createAnalysisJob(
            imageJpeg = jpeg,
            cameraContextJson = cameraContext,
            deviceCapabilitiesJson = capabilitySkill,
            localQualityJson = localQuality,
            skillContext = skillContext,
            idempotencyKey = UUID.randomUUID().toString()
        ).getOrThrow()
        activeJobId = created.id
        var job = created
        while (true) {
            when (job.state) {
                "ready" -> {
                    val dto = checkNotNull(job.plan) { "Sierra Server returned ready without a plan" }
                    val plan = PlanOutput(
                        scene = dto.scene,
                        objective = dto.objective,
                        skillBranch = dto.branch,
                        referenceScore = dto.reference_score,
                        referenceAccept = dto.reference_accept,
                        toolCalls = dto.calls.map { ToolCall(it.id, it.value, it.x, it.y, it.stringValue) },
                        recapture = dto.recapture,
                        aspectRatio = dto.aspect_ratio,
                        cropCenterX = dto.crop_center_x,
                        cropCenterY = dto.crop_center_y,
                        planner = "sierra_ai_server"
                    )
                    return@runCatching VlmContractValidator.plan(plan)
                }
                "failed", "cancelled", "expired" -> error(job.error_message ?: "Sierra AI job ${job.state}")
                "completed" -> error("Sierra AI job completed before camera capture")
            }
            delay(120)
            job = server.job(created.id).getOrThrow()
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
        UnsupportedOperationException("Sierra Vision uses one cloud plan; final technical QA is local")
    )

    override suspend fun completeFinalCapture(captureId: String): Result<Unit> {
        val jobId = activeJobId ?: return Result.success(Unit)
        return server.complete(jobId, captureId).map {
            activeJobId = null
            Unit
        }
    }

    override fun cancelCloudJob() {
        val jobId = activeJobId ?: return
        activeJobId = null
        scope.launch { server.cancel(jobId) }
    }

    override fun setMaxImageDimension(pixels: Int) {
        maxImageDimension = pixels.coerceIn(384, 896)
    }

    override fun cancel() = cancelCloudJob()
    override fun isReady(): Boolean = server.configured && server.authenticated
    override fun unload() = cancelCloudJob()
    override fun close() {
        cancelCloudJob()
        scope.cancel()
    }

    private fun encode(source: Bitmap): ByteArray {
        val longest = max(source.width, source.height)
        val output = if (longest > maxImageDimension) {
            val scale = maxImageDimension.toFloat() / longest.toFloat()
            Bitmap.createScaledBitmap(
                source,
                (source.width * scale).toInt().coerceAtLeast(1),
                (source.height * scale).toInt().coerceAtLeast(1),
                true
            )
        } else source
        return try {
            ByteArrayOutputStream().use { stream ->
                check(output.compress(Bitmap.CompressFormat.JPEG, 76, stream)) { "Could not encode Sierra analysis frame" }
                stream.toByteArray()
            }
        } finally {
            if (output !== source && !output.isRecycled) output.recycle()
        }
    }
}
