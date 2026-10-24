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

    val personSegmentation = ModelManifest(
        id = "mediapipe_selfie_segmentation_web",
        version = "f2fc46440eb24781",
        task = ModelTask.SEGMENTATION,
        tier = ModelTier.WARM,
        input = ModelInputSpec(
            width = null,
            height = null,
            channels = 3,
            dynamicShape = true,
            pixelFormat = "rgb-u8-nhwc",
            normalization = "embedded preprocessing; raw 0..255 UINT8"
        ),
        supportedBackends = setOf(ComputeBackend.CPU_XNNPACK, ComputeBackend.NNAPI),
        preferredBackends = listOf(ComputeBackend.NNAPI, ComputeBackend.CPU_XNNPACK),
        sha256 = "44f328e21a3840b7ddb177c8efe50177248d15d25b4905479b06e50d2470584e",
        license = ModelLicense(
            codeLicense = "Apache-2.0",
            weightLicense = "Apache-2.0",
            commercialUse = true,
            redistributionAllowed = true,
            attributionRequired = true
        ),
        minimumGraphCoverage = 0.94,
        maxTransferBoundaries = 2,
        nominalHalfLifeMs = 140,
        sourceUrl = "https://huggingface.co/onnx-community/mediapipe_selfie_segmentation-web/resolve/f2fc46440eb2478139cc0f122c24fab9f00124bc/onnx/model.onnx?download=true"
    )

    val rtmposeSmall = ModelManifest(
        id = "rtmpose_s_coco17",
        version = "8508e7b1ffb82fe0",
        task = ModelTask.LANDMARKS,
        tier = ModelTier.WARM,
        input = ModelInputSpec(
            width = 192,
            height = 256,
            channels = 3,
            dynamicShape = false,
            pixelFormat = "rgb-f32-nchw",
            normalization = "mean=[123.675,116.28,103.53]; std=[58.395,57.12,57.375]"
        ),
        supportedBackends = setOf(ComputeBackend.CPU_XNNPACK, ComputeBackend.NNAPI),
        preferredBackends = listOf(ComputeBackend.NNAPI, ComputeBackend.CPU_XNNPACK),
        sha256 = "d869413ccfbc0ed027337c773f124d6362436239a83fd0a6e2d43b7a8377792e",
        license = ModelLicense(
            codeLicense = "Apache-2.0",
            weightLicense = "Apache-2.0",
            commercialUse = true,
            redistributionAllowed = true,
            attributionRequired = true
        ),
        minimumGraphCoverage = 0.96,
        maxTransferBoundaries = 2,
        nominalHalfLifeMs = 120,
        sourceUrl = "https://huggingface.co/pzoltowski/rtmpose-s-onnx/resolve/8508e7b1ffb82fe0ce6e11b72ea65b96101f44a2/rtmpose-s_working.onnx?download=true"
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

    val all: List<ModelManifest> = listOf(
        yunet2026May,
        personSegmentation,
        rtmposeSmall,
        depthAnythingV2Small
    )

    fun registerAll(registry: ModelRegistry) = all.forEach(registry::register)
}
