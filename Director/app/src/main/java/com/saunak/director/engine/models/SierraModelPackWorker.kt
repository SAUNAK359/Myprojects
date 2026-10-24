package com.saunak.director.engine.models

import android.content.Context
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.saunak.director.engine.SierraEngine
import com.saunak.director.engine.experts.DepthAnythingV2Expert
import java.util.concurrent.TimeUnit

/**
 * Reliable deferred provisioning for large WARM/CAPTURE model artifacts. A successful install is
 * followed by best-effort backend profiling so shutter-time routing already has real device data.
 */
class SierraModelPackWorker(
    appContext: Context,
    workerParams: WorkerParameters
) : CoroutineWorker(appContext, workerParams) {

    override suspend fun doWork(): Result {
        val modelId = inputData.getString(KEY_MODEL_ID) ?: return Result.failure()
        val version = inputData.getString(KEY_MODEL_VERSION) ?: return Result.failure()
        val engine = SierraEngine.initialize(applicationContext)
        val manifest = engine.models.get(modelId, version) ?: return Result.failure()

        if (manifest.tier == ModelTier.HOT || manifest.tier == ModelTier.REFERENCE) return Result.failure()
        if (!manifest.allowedForCommercialRelease()) return Result.failure()

        val existing = engine.modelArtifacts.findVerified(manifest)
        val artifact = existing ?: engine.modelArtifacts.ensure(manifest).getOrElse {
            return if (runAttemptCount < MAX_ATTEMPTS - 1) Result.retry() else Result.failure()
        }

        // Profiling failure does not invalidate verified model bytes. The runtime keeps a portable
        // correctness fallback and may retry profiling in a later maintenance cycle.
        if (manifest.id == SierraShippingManifests.depthAnythingV2Small.id) {
            runCatching { DepthAnythingV2Expert(engine).profileIfProvisioned().getOrThrow() }
        }

        return Result.success(
            workDataOf(
                KEY_BYTES_READY to true,
                KEY_INSTALLED_BYTES to artifact.bytes,
                KEY_PROFILE_ATTEMPTED to true
            )
        )
    }

    companion object {
        private const val KEY_MODEL_ID = "model_id"
        private const val KEY_MODEL_VERSION = "model_version"
        private const val KEY_BYTES_READY = "bytes_ready"
        private const val KEY_INSTALLED_BYTES = "installed_bytes"
        private const val KEY_PROFILE_ATTEMPTED = "profile_attempted"
        private const val MAX_ATTEMPTS = 3

        fun uniqueWorkName(manifest: ModelManifest): String =
            "sierra-model-${manifest.id}-${manifest.version}"

        fun request(manifest: ModelManifest) = OneTimeWorkRequestBuilder<SierraModelPackWorker>()
            .setInputData(
                workDataOf(
                    KEY_MODEL_ID to manifest.id,
                    KEY_MODEL_VERSION to manifest.version
                )
            )
            .setConstraints(
                Constraints.Builder()
                    .setRequiredNetworkType(NetworkType.UNMETERED)
                    .setRequiresBatteryNotLow(true)
                    .setRequiresStorageNotLow(true)
                    .build()
            )
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
            .addTag("sierra-model-pack")
            .addTag("sierra-model-${manifest.id}")
            .build()
    }
}

object SierraModelProvisioning {
    fun scheduleProductionPacks(context: Context, registry: ModelRegistry) {
        val work = WorkManager.getInstance(context)
        registry.commercialReleaseReady()
            .filter { it.tier == ModelTier.WARM || it.tier == ModelTier.CAPTURE }
            .filter { it.sourceUrl != null }
            .forEach { manifest ->
                work.enqueueUniqueWork(
                    SierraModelPackWorker.uniqueWorkName(manifest),
                    ExistingWorkPolicy.KEEP,
                    SierraModelPackWorker.request(manifest)
                )
            }
    }
}
