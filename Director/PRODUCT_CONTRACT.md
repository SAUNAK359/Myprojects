# Sierra Vision v3 Beta — Product Contract

This contract describes the behavior of `director-sierra-v3`. It supersedes the earlier one-local-VLM/Qwen reference-capture contract.

## Current Beta interaction

Sierra v3 currently uses a two-stage Director interaction rather than a conventional one-tap stock-camera shutter:

1. The white shutter starts preparation.
2. Sierra samples a current analysis frame, measures local technical quality, runs the local YuNet face expert when available, and builds a local plan. A signed-in non-private session may additionally request one bounded Sierra Server plan.
3. Cloud output is treated as untrusted structured data. It is normalized, reconciled with local planning, schema/range checked and capability-gated before camera APIs are touched.
4. Temporary focus/zoom/exposure/flash/extension changes are applied locally, followed by a verification frame when the active camera session supports analysis.
5. Sierra enters `PREPARED` state. The second shutter performs the final still capture.
6. With an active OEM computational extension, Sierra captures one final still and does not multiply the OEM pipeline into an additional burst.
7. Without an OEM extension, Sierra may perform a thermally bounded smart CameraX JPEG burst. The current fallback burst performs technical frame selection; it does not claim RAW/HDR fusion.
8. The selected still is cropped/rendered when requested and saved to MediaStore. If rendering/saving the intended final fails after an untouched still exists, Sierra attempts to preserve that untouched still instead.
9. Temporary burst files are cleaned after finalization.

## Non-negotiable safety rules

- Camera availability is more important than AI availability.
- Private Photography never invokes Sierra Server planning.
- A model, cloud planner or quality component may fail without making the normal CameraX camera unusable.
- A cloud/model output never executes arbitrary Android operations. Only typed Sierra tool calls exist.
- Tool calls are count-bounded, parameter-validated, range-clamped, conflict-resolved and filtered through live device capabilities.
- Unsupported hardware capability is never fabricated.
- An active CameraX OEM extension owns its private capture pipeline; Sierra does not force manual RAW controls into that session.
- Stale/cancelled preparation restores user camera state before it exits.
- Thermal policy may reduce AI work and burst length.
- Model files are never trusted by filename alone; installable production models require a pinned SHA-256 and an approved redistribution/commercial-use manifest.
- Relative monocular depth must never be represented as metric metres.
- Research catalog entries are not shipping features.

## Cloud and credits

Sierra Server is optional. The Android application stores only Sierra session tokens and uses the server as a planning boundary; third-party provider credentials are forbidden from Android source. The analysis image sent to Sierra Server is a reduced JPEG rather than the final full-resolution photograph.

A cloud analysis job is completed/settled only after a final photo is successfully saved. Pending settlement identifiers are persisted so account synchronization can retry after transient failure. Cancellation clears the active local cloud job and requests server cancellation asynchronously.

## Current local intelligence

Live:

- deterministic technical image statistics;
- YuNet face detection when its verified model is available;
- Sierra scene heuristics/local planner;
- capability-constrained camera tool execution;
- thermal policy;
- smart-burst technical frame selection;
- deterministic look/crop finishing.

Implemented but not yet mandatory in the live shutter path:

- Depth Anything V2 Small relative depth;
- DeviceTwin calibration metadata;
- SceneBeliefGraph;
- CoordinateGraph;
- ExpertScheduler / ComputeHypervisor;
- EvidenceLedger / ReconstructionArbiter;
- QualityCouncil.

## Current image-quality boundary

The v3 Beta must not call the JPEG smart-burst selector a fused HDR/denoise/SR pipeline. True Sierra computational photography remains a separate implementation milestone requiring certified RAW/YUV acquisition, alignment, motion/occlusion reasoning, deghosting, radiance fusion, denoise/SR, calibrated ISP, semantic rendering and evidence-gated quality arbitration.

Creative/cropped output currently uses a bounded Android bitmap finishing path. Natural identity output can preserve the captured JPEG bytes, but the current creative-processing path is not a full-resolution RAW ISP and must not be marketed as one.

## Release gate

A debug APK compiling is necessary but not sufficient for production. Release requires:

- strict unit/lint/build CI;
- signed release APK/AAB validation;
- production model-licence validation and bundled notices;
- dependency/security/SBOM review;
- physical Android device tests for CameraX extensions and lifecycle/state restoration;
- model execution tests on actual CPU/GPU/NPU paths;
- shutter-latency, memory and sustained thermal testing;
- controlled image-quality comparison against native camera apps.

The product must not claim that Sierra beats native OEM cameras until those device/image benchmarks demonstrate it.
