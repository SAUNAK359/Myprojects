package com.saunak.director.engine.models

import com.saunak.director.engine.device.ComputeBackend

/**
 * Pinned production-candidate artifacts with immutable source revisions and SHA-256 digests.
 * Adding a model here is a release-affecting change: URL, hash, licence and tensor contract must all
 * be independently verified before merge.
 */
object SierraShippingManifests {
    val yunet2026May = ModelManifest(
        id = "yunet_2026may",
        version = "47534e27c9851bb1",
        task = ModelTask.FACE_DETECTION,
        tier = ModelTier.HOT,
        input = ModelInputSpec(
            width = 320,
            height = 320,
            channels = 3,
            dynamicShape = true,
            pixelFormat = "bgr-f32-nchw",
            normalization = "none; values 0..255"
        ),
        supportedBackends = setOf(ComputeBackend.CPU_XNNPACK, ComputeBackend.NNAPI),
        preferredBackends = listOf(ComputeBackend.NNAPI, ComputeBackend.CPU_XNNPACK),
        sha256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
        license = ModelLicense(
            codeLicense = "MIT",
            weightLicense = "MIT",
            commercialUse = true,
            redistributionAllowed = true,
            attributionRequired = true
        ),
        minimumGraphCoverage = 0.98,
        maxTransferBoundaries = 1,
        nominalHalfLifeMs = 120,
        sourceUrl = "https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/face_detection_yunet_2026may.onnx"
    )

    val depthAnythingV2Small = ModelManifest(
        id = "depth_anything_v2_small",
        version = "0d8d94ea44f483e2",
        task = ModelTask.DEPTH,
        tier = ModelTier.WARM,
        input = ModelInputSpec(
            width = 518,
            height = 518,
            channels = 3,
            dynamicShape = true,
            pixelFormat = "rgb-f32-nchw",
            normalization = "ImageNet mean/std"
        ),
        supportedBackends = setOf(ComputeBackend.CPU_XNNPACK, ComputeBackend.NNAPI),
        preferredBackends = listOf(ComputeBackend.NNAPI, ComputeBackend.CPU_XNNPACK),
        sha256 = "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c",
        license = ModelLicense(
            codeLicense = "Apache-2.0",
            weightLicense = "Apache-2.0",
            commercialUse = true,
            redistributionAllowed = true,
            attributionRequired = true
        ),
        minimumGraphCoverage = 0.98,
        maxTransferBoundaries = 1,
        nominalHalfLifeMs = 350,
        sourceUrl = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/0d8d94ea44f483e26c9c33dad1a97c453ef9b80c/onnx/model.onnx?download=true"
    )

    val all: List<ModelManifest> = listOf(yunet2026May, depthAnythingV2Small)

    fun registerAll(registry: ModelRegistry) = all.forEach(registry::register)
}
