# Sierra Vision / Director — v3 Beta

Sierra Vision is an Android computational-camera project. The current `director-sierra-v3` branch combines a CameraX camera, local technical analysis and on-device experts, an optional Sierra Server planning path, capability-gated camera tools, adaptive capture planning, smart burst selection, deterministic finishing, encrypted account state, and a growing Sierra engine for deeper computational photography.

This document describes the code that is actually live in the v3 Beta branch. Research architecture that is not yet connected to the shutter path is called out explicitly rather than presented as a finished feature.

## Current end-to-end workflow

The current Beta is a two-stage Director workflow:

`live CameraX preview → first shutter/prepare → analysis frame → local technical quality + YuNet face expert → local Sierra planner and optional Sierra Server plan → contract/tool validation → temporary camera controls → verification → PREPARED → second shutter → OEM single capture or Sierra smart JPEG burst → technical frame selection → deterministic crop/look finishing → MediaStore save`

The camera remains usable when cloud AI or on-device experts fail. Sierra Server is optional; Private Photography and guest mode keep planning local.

### Capture safety

- Camera mutations are capability-gated and range-clamped before CameraX/Camera2 APIs are touched.
- User camera state is restored after completion, cancellation, stale work, or recoverable failure.
- An active OEM HDR/Night/Bokeh/Face Retouch extension is never multiplied into a Sierra multi-frame burst; the extension gets one final still.
- The current universal multi-frame path is a **smart JPEG burst selector**, not RAW/HDR fusion. It captures independent full-quality CameraX JPEGs, rejects catastrophic frames, and chooses the strongest technical frame.
- If finishing fails after a valid still was captured, Sierra attempts to save the untouched JPEG instead of discarding the photograph.
- Burst size is bounded by thermal policy and working-storage checks.

## On-device models actually implemented

### YuNet face detection — live

A checksum-pinned YuNet 2026 ONNX checkpoint is downloaded into app-private storage, verified before installation, and used during shot preparation when available. NNAPI is attempted with XNNPACK as the portable correctness fallback. The model-specific MIT license is bundled in the APK assets.

### Depth Anything V2 Small — provisioned, not yet in the live shutter decision

A checksum-pinned Apache-2.0 ONNX checkpoint can be provisioned by constrained WorkManager jobs. The expert returns normalized **relative inverse depth**, never fabricated metric metres. The model is implemented and testable, but v3 Beta does not yet use its output to drive portrait rendering, capture policy, or final-image arbitration.

## Sierra engine implemented as infrastructure

The repository contains production-oriented primitives for:

- immutable frame metadata contracts;
- Camera2 device/calibration twins;
- explicit coordinate-space transforms;
- time-decaying probabilistic scene beliefs;
- model manifests, SHA-256 verification and release licence gates;
- ONNX Runtime session management with NNAPI/XNNPACK fallbacks;
- value-of-information expert scheduling;
- measured-backend/pressure-aware scheduling models;
- capture-program compilation;
- evidence ledgers and deterministic-vs-neural reconstruction arbitration;
- multi-axis quality-council decisions;
- thermal and memory-pressure handling.

These primitives are not all connected to the current shutter path yet. In particular, the SceneBeliefGraph, CoordinateGraph, ExpertScheduler/ComputeHypervisor, EvidenceLedger, ReconstructionArbiter and QualityCouncil are currently engine infrastructure rather than mandatory final-capture gates.

## Computational-photography work still required for the full Sierra architecture

The following are **not** complete in the live Beta path and must not be represented as shipping features yet:

- certified RAW+JPEG/manual burst execution;
- YUV reprocessing path;
- CFA-aware RAW alignment/reconstruction;
- gyro/rolling-shutter/optical-flow fusion;
- HDR radiance fusion and deghosting;
- temporal denoise and true burst super-resolution;
- face parsing, portrait matting and whole-body pose experts;
- scene-semantic foundation model;
- semantic/calibrated neural ISP and device-specific colour calibration;
- multi-camera pixel fusion;
- TOPIQ/Q-ReAlign-class final IQA;
- live EvidenceLedger/ReconstructionArbiter/QualityCouncil integration;
- measured backend profiling/interference tables for true CPU/GPU/NPU load balancing.

The model catalog contains research candidates for these roles, but a catalog entry is not treated as an implemented model. Only checksum-pinned manifests with acceptable licence terms can become installable production assets.

## Security and privacy

- Android permissions: Camera and Internet only.
- Cleartext network traffic is disabled.
- Sierra Server URL must be blank or HTTPS at build time.
- Provider API keys are forbidden in Android source and checked in CI.
- Account access/refresh tokens are stored with Android Keystore AES/GCM encryption.
- Model downloads are HTTPS-only, size-bounded, disk-space-gated, SHA-256 verified and atomically installed.
- Current account/session, settlement and Sierra preference files are explicitly excluded from backup/device-transfer rules in addition to `allowBackup=false`.
- Private Photography disables cloud planning; only local Beta logic is used.
- Shipping model licences are bundled under `app/src/main/assets/third_party_licenses/`.

## Build stack

- Android compile/target SDK 36
- JDK 17
- Kotlin / Compose
- CameraX 1.6.2
- ONNX Runtime Android 1.31.0
- WorkManager 2.12.0
- OkHttp 4.12.0
- minSdk 29

## CI gates

GitHub Actions and CircleCI build the Android project. CircleCI contains independent compile-isolation jobs for UI, cloud runtime and backend boundaries, plus unit tests and APK verification. Android lint is a blocking gate. APK verification checks signing metadata, alignment, ZIP integrity and DEX presence.

The current CI validates a debug APK. A production release still requires a dedicated signed release pipeline, R8/shrinker validation, dependency/SBOM and licence compliance checks, and physical-device qualification.

## Release status

Sierra v3 Beta is an actively integrated deep-tech camera branch, not yet a justified claim of outperforming Pixel/Samsung/Xiaomi stock cameras. The current product path is substantially more conservative than the full research architecture: it protects camera availability and captured evidence first, while RAW reconstruction, semantic ISP, full expert fusion and measured heterogeneous scheduling are completed and benchmarked.

No release should claim OEM superiority until controlled device tests demonstrate it across daylight, low light, motion, portrait, HDR, zoom, skin tone, colour error, detail, ghosting, thermal behaviour, shutter latency and blind human preference.
