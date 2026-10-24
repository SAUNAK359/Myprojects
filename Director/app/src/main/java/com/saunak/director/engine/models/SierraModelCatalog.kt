package com.saunak.director.engine.models

/**
 * Architecture-level model catalog. This intentionally does not invent artifact URLs/checksums:
 * a candidate becomes installable only after a concrete ModelManifest passes the artifact/licence
 * pipeline. Keeping selection and provisioning separate prevents a research checkpoint from
 * silently becoming a production dependency.
 */
enum class DeploymentClearance { SHIPPING_APPROVED, LEGAL_REVIEW, RESEARCH_ONLY }
enum class ModelRole {
    SCENE_SEGMENTATION,
    FACE_DETECTION,
    BODY_DETECTION,
    WHOLE_BODY_LANDMARKS,
    FACE_PARSING,
    PORTRAIT_MATTING,
    SCENE_SEMANTICS,
    DEPTH,
    OPTICAL_FLOW,
    DENOISE,
    BURST_RECONSTRUCTION,
    BURST_SUPER_RESOLUTION,
    ISP_FINISHING,
    FAST_IQA,
    DEEP_IQA
}

data class ModelCandidate(
    val role: ModelRole,
    val name: String,
    val upstream: String,
    val intendedUse: String,
    val clearance: DeploymentClearance,
    val codeLicense: String,
    val weightsLicense: String,
    val mobileNotes: String,
    val oracle: Boolean = false
)

object SierraModelCatalog {
    val candidates: List<ModelCandidate> = listOf(
        ModelCandidate(
            ModelRole.SCENE_SEGMENTATION,
            "EoMT-L / DINOv3",
            "tue-mps/eomt",
            "maximum-quality segmentation oracle and capture-time validation",
            DeploymentClearance.RESEARCH_ONLY,
            "MIT upstream EoMT",
            "DINOv3 terms apply to DINOv3-backed checkpoints",
            "Never resident by default; benchmark/export before any phone deployment.",
            oracle = true
        ),
        ModelCandidate(
            ModelRole.SCENE_SEGMENTATION,
            "EoMT DINOv2 (permissive checkpoint)",
            "tue-mps/eomt / verified permissive redistribution",
            "shipping segmentation candidate",
            DeploymentClearance.LEGAL_REVIEW,
            "MIT",
            "Checkpoint-specific review required",
            "Prefer smaller backbone for monitor; capture oracle may use larger variant."
        ),
        ModelCandidate(
            ModelRole.FACE_DETECTION,
            "SCRFD-34G",
            "deepinsight/insightface",
            "research face-detection oracle",
            DeploymentClearance.RESEARCH_ONLY,
            "MIT code",
            "official pretrained assets have separate non-commercial restrictions",
            "Oracle only until weights are independently trained/licensed.",
            oracle = true
        ),
        ModelCandidate(
            ModelRole.FACE_DETECTION,
            "YuNet 2026 ONNX",
            "opencv/opencv_zoo",
            "commercial-safe live face monitor and fallback",
            DeploymentClearance.SHIPPING_APPROVED,
            "MIT",
            "MIT",
            "ONNX-ready, tiny, dynamic-input model; suitable for always-on monitoring."
        ),
        ModelCandidate(
            ModelRole.BODY_DETECTION,
            "RTMDet",
            "open-mmlab/mmdetection",
            "person/body localization",
            DeploymentClearance.LEGAL_REVIEW,
            "Apache-2.0",
            "checkpoint-specific verification required",
            "Use ROI hierarchy; do not run full detector for face-only updates."
        ),
        ModelCandidate(
            ModelRole.WHOLE_BODY_LANDMARKS,
            "RTMW-x",
            "open-mmlab/mmpose",
            "capture-time whole-body/hand/foot geometry oracle",
            DeploymentClearance.LEGAL_REVIEW,
            "Apache-2.0",
            "checkpoint-specific verification required",
            "Run on detected person ROIs; smaller RTMW variant may serve live monitoring.",
            oracle = true
        ),
        ModelCandidate(
            ModelRole.FACE_PARSING,
            "BiSeNet face parsing ONNX",
            "yakhyo/face-parsing",
            "skin/hair/eyes/lips/glasses semantic masks",
            DeploymentClearance.SHIPPING_APPROVED,
            "MIT",
            "MIT repository/pretrained distribution",
            "ROI-only ONNX path; much cheaper and more precise than generic scene segmentation."
        ),
        ModelCandidate(
            ModelRole.PORTRAIT_MATTING,
            "MODNet-class portrait matting",
            "ZHKKKe/MODNet",
            "fractional alpha around hair and translucent boundaries",
            DeploymentClearance.LEGAL_REVIEW,
            "repository licence review",
            "checkpoint-specific review required",
            "Only invoke for portrait/bokeh output; propagate matte temporally between refreshes."
        ),
        ModelCandidate(
            ModelRole.SCENE_SEMANTICS,
            "MobileCLIP2-S2",
            "apple/ml-mobileclip",
            "semantic monitor / zero-shot scene concepts",
            DeploymentClearance.LEGAL_REVIEW,
            "MIT code",
            "Apple ML Research Model TOU",
            "35.7M image parameters; preferred live semantic candidate if model terms are approved."
        ),
        ModelCandidate(
            ModelRole.SCENE_SEMANTICS,
            "MobileCLIP2-S4",
            "apple/ml-mobileclip",
            "high-accuracy semantic oracle",
            DeploymentClearance.RESEARCH_ONLY,
            "MIT code",
            "Apple ML Research Model TOU",
            "321.6M image parameters; event/capture oracle only, never always resident.",
            oracle = true
        ),
        ModelCandidate(
            ModelRole.DEPTH,
            "Metric3D v2",
            "YvanYin/Metric3D",
            "metric-depth research oracle",
            DeploymentClearance.RESEARCH_ONLY,
            "BSD-2-Clause code",
            "commercial terms require explicit author review",
            "Requires correct intrinsics/crop normalization; capture oracle only.",
            oracle = true
        ),
        ModelCandidate(
            ModelRole.DEPTH,
            "Depth Anything V2 Small",
            "DepthAnything/Depth-Anything-V2",
            "portable relative-depth monitor / geometry fallback",
            DeploymentClearance.SHIPPING_APPROVED,
            "Apache-2.0",
            "Apache-2.0 Small checkpoint",
            "24.8M parameters; official project lists Android community ports and ONNX exports exist."
        ),
        ModelCandidate(
            ModelRole.OPTICAL_FLOW,
            "WAFT",
            "princeton-vl/WAFT",
            "dense capture-time optical-flow oracle",
            DeploymentClearance.LEGAL_REVIEW,
            "BSD-3-Clause",
            "checkpoint dependency/backbone licences must also be checked",
            "Backbone-specific deployment; use gyro/tracking monitor between oracle refreshes.",
            oracle = true
        ),
        ModelCandidate(
            ModelRole.DENOISE,
            "NAFNet",
            "megvii-research/NAFNet",
            "single-frame/local residual restoration fallback",
            DeploymentClearance.LEGAL_REVIEW,
            "repository licence",
            "checkpoint-specific review required",
            "Not Sierra's primary RAW denoiser; multi-frame RAW reconstruction has priority."
        ),
        ModelCandidate(
            ModelRole.BURST_RECONSTRUCTION,
            "DEBIR / Burstormer-class",
            "research burst-restoration stack",
            "neural RAW burst reconstruction challenger",
            DeploymentClearance.RESEARCH_ONLY,
            "research repositories",
            "verify each checkpoint before shipping",
            "Always competes against deterministic evidence-preserving baseline.",
            oracle = true
        ),
        ModelCandidate(
            ModelRole.BURST_SUPER_RESOLUTION,
            "QMambaBSR",
            "CVPR 2025 burst-SR implementation",
            "conditional multi-frame super-resolution challenger",
            DeploymentClearance.RESEARCH_ONLY,
            "research repository",
            "verify checkpoint terms",
            "Only activate when predicted detail gain exceeds artifact/compute cost.",
            oracle = true
        ),
        ModelCandidate(
            ModelRole.FAST_IQA,
            "TOPIQ-class IQA",
            "chaofengc/IQA-PyTorch",
            "synchronous perceptual quality signal",
            DeploymentClearance.LEGAL_REVIEW,
            "repository/checkpoint review",
            "checkpoint-specific review required",
            "One vote inside Quality Council, never sole release criterion."
        ),
        ModelCandidate(
            ModelRole.DEEP_IQA,
            "Q-ReAlign Mini",
            "Q-Future/Q-ReAlign",
            "asynchronous deep image-quality oracle",
            DeploymentClearance.RESEARCH_ONLY,
            "research repository",
            "verify model/base-model terms",
            "0.8B class model: never blocks shutter or initial photo save.",
            oracle = true
        )
    )

    fun shippingApproved(): List<ModelCandidate> = candidates.filter { it.clearance == DeploymentClearance.SHIPPING_APPROVED }
    fun researchOracles(): List<ModelCandidate> = candidates.filter(ModelCandidate::oracle)
    fun forRole(role: ModelRole): List<ModelCandidate> = candidates.filter { it.role == role }
}
