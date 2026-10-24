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
import java.util.concurrent.TimeUnit

/**
 * Reliable, deferred provisioning for large WARM/CAPTURE model artifacts. HOT models are prepared
 * separately at startup; shutter-critical code never performs network I/O.
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

        // Only deferred model tiers belong here; HOT artifacts must never be delayed behind WorkManager.
        if (manifest.tier == ModelTier.HOT || manifest.tier == ModelTier.REFERENCE) return Result.failure()
        if (!manifest.allowedForCommercialRelease()) return Result.failure()

        if (engine.modelArtifacts.findVerified(manifest) != null) {
            return Result.success(workDataOf(KEY_BYTES_READY to true))
        }

        val installed = engine.modelArtifacts.ensure(manifest)
        if (installed.isSuccess) {
            val artifact = installed.getOrThrow()
            return Result.success(
                workDataOf(
                    KEY_BYTES_READY to true,
                    KEY_INSTALLED_BYTES to artifact.bytes
                )
            )
        }

        // Network/server/transient storage failures may recover. A small capped retry budget avoids
        // endless retries for a bad checksum or a withdrawn artifact.
        return if (runAttemptCount < MAX_ATTEMPTS - 1) Result.retry() else Result.failure()
    }

    companion object {
        private const val KEY_MODEL_ID = "model_id"
        private const val KEY_MODEL_VERSION = "model_version"
        private const val KEY_BYTES_READY = "bytes_ready"
        private const val KEY_INSTALLED_BYTES = "installed_bytes"
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
    /**
     * Enqueue only redistributable WARM/CAPTURE artifacts. KEEP makes startup idempotent and lets
     * WorkManager retain a successful download across process restarts.
     */
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
