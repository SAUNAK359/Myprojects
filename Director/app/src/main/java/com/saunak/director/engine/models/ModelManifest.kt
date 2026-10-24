package com.saunak.director.engine.models

import com.saunak.director.engine.device.ComputeBackend

enum class ModelTier { HOT, WARM, CAPTURE, REFERENCE }
enum class ModelTask {
    SEGMENTATION, FACE_DETECTION, BODY_DETECTION, LANDMARKS, FACE_PARSING, MATTING,
    SCENE_CLASSIFICATION, DEPTH, OPTICAL_FLOW, DENOISE, BURST_RECONSTRUCTION,
    SUPER_RESOLUTION, ISP, QUALITY, AWB, UNKNOWN
}

data class ModelLicense(
    val codeLicense: String,
    val weightLicense: String,
    val commercialUse: Boolean,
    val redistributionAllowed: Boolean,
    val attributionRequired: Boolean
)

data class ModelInputSpec(
    val width: Int?,
    val height: Int?,
    val channels: Int,
    val dynamicShape: Boolean,
    val pixelFormat: String,
    val normalization: String
) {
    init {
        require(width == null || width > 0)
        require(height == null || height > 0)
        require(channels > 0)
        require(pixelFormat.isNotBlank())
    }
}

data class ModelManifest(
    val id: String,
    val version: String,
    val task: ModelTask,
    val tier: ModelTier,
    val input: ModelInputSpec,
    val supportedBackends: Set<ComputeBackend>,
    val preferredBackends: List<ComputeBackend>,
    val sha256: String,
    val license: ModelLicense,
    val minimumGraphCoverage: Double = 0.98,
    val maxTransferBoundaries: Int = 1,
    val nominalHalfLifeMs: Long,
    val sourceUrl: String? = null
) {
    init {
        require(id.matches(Regex("[a-z0-9_.-]+"))) { "Model id must be filesystem-safe" }
        require(version.isNotBlank())
        require(supportedBackends.isNotEmpty())
        require(preferredBackends.all { it in supportedBackends })
        require(sha256.matches(Regex("[0-9a-fA-F]{64}"))) { "Expected SHA-256 hex" }
        require(minimumGraphCoverage in 0.0..1.0)
        require(maxTransferBoundaries >= 0)
        require(nominalHalfLifeMs > 0)
        require(sourceUrl == null || sourceUrl.startsWith("https://"))
    }

    fun allowedForCommercialRelease(): Boolean =
        license.commercialUse && license.redistributionAllowed
}
