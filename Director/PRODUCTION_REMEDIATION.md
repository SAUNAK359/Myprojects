# Director Production Remediation Register

Status values: `OPEN`, `IMPLEMENTING`, `DONE-CODE`, `DEVICE-VALIDATION`.

This register is the source of truth for the hardening work on branch `director-production-remediation`.

## Gate A — Product / shutter contract

| ID | Problem | Production solution | Fallback / option | Status |
|---|---|---|---|---|
| A1 | AI starts from sampled preview frames | Remove VLM preview trigger; shutter capture is sole session trigger | Preview may expose native 3A state only | IMPLEMENTING |
| A2 | White shutter bypasses Director | Shutter captures private full-quality reference then starts Director | If AI unavailable, save reference unchanged | IMPLEMENTING |
| A3 | Wrong image enters planner | Decode the exact reference JPEG with EXIF orientation and use it for plan inference | If decode fails, copy original JPEG to gallery | IMPLEMENTING |
| A4 | Autonomous photos can occur without consent | No autonomous `captureTemp()` path exists outside a shutter session | None | IMPLEMENTING |
| A5 | User action can race stale AI | Generation/session token invalidates old work before every mutation/capture/save decision | New shutter has priority; stale session saves best/reference only | IMPLEMENTING |

## Gate B — Camera capability / transactions

| ID | Problem | Production solution | Fallback / option | Status |
|---|---|---|---|---|
| B1 | Capability registry too small | Probe CameraX ranges plus conservative Camera2 characteristics: AF, manual sensor, ISO/exposure ranges, focus distance, RAW, focal lengths, stabilization, flash and extensions | Unknown capability = unsupported | IMPLEMENTING |
| B2 | Focus tool is a no-op | Implement normalized AF/AE/AWB metering point tool | Reject if camera has no controllable focus/metering | IMPLEMENTING |
| B3 | No one-shot flash tool | Add OFF/AUTO/ON still-capture flash mode separate from torch | Unsupported flash => reject | IMPLEMENTING |
| B4 | No HDR/Night/Bokeh/Face Retouch tool | Probe CameraX Extensions and expose only modes reported available | Extension bind failure rolls back to normal selector | IMPLEMENTING |
| B5 | AI settings persist after session | Track user state separately; restore after finish/cancel before accepting next reference | Restore failure => rebind normal camera | IMPLEMENTING |
| B6 | No conflict resolver | Convert plan into one normalized transaction; resolve extension/manual/AE/torch/flash conflicts deterministically | Reject unsafe/conflicting calls individually | IMPLEMENTING |
| B7 | Fixed 220 ms settle delay | Observe Camera2 capture results for AF/AE/AWB convergence with bounded timeout | Timeout falls back to short grace period then capture | IMPLEMENTING |
| B8 | UI controls do not reflect real device ranges | Publish camera UI state/capabilities from engine | Hide unsupported controls | IMPLEMENTING |

## Gate C — Typed tool system

| ID | Problem | Production solution | Fallback / option | Status |
|---|---|---|---|---|
| C1 | VLM receives IDs without semantics | Generate capability-filtered tool contract containing ID, name, params and bounds | No contract => no AI execution | IMPLEMENTING |
| C2 | Evaluator lacks tool contract | Give evaluator the same capability-filtered delta contract | Invalid delta ignored; keep previous state | IMPLEMENTING |
| C3 | Rejected tools silently disappear | Executor returns applied/rejected with reasons; orchestrator records rejection in evaluation context | Continue with safe subset; never fabricate support | IMPLEMENTING |
| C4 | Numeric/string parameters weakly checked | Finite checks, string enums, coordinate bounds, length limits, max call counts | Invalid call is rejected before executor | IMPLEMENTING |
| C5 | Creative looks are all-or-nothing | Add bounded strength parameter and deterministic blending with identity | Strength omitted => curated default | IMPLEMENTING |

## Gate D — Skills architecture

| ID | Problem | Production solution | Fallback / option | Status |
|---|---|---|---|---|
| D1 | All full scene skills dumped into planner prompt | Compile branch routing summaries separately from full branch bodies | General branch if output branch is unknown | IMPLEMENTING |
| D2 | VLM branch output unused | Validate selected branch and load only that branch for evaluation/correction | General branch | IMPLEMENTING |
| D3 | Skills are tiny unstructured fragments | Expand each V1 branch with subject, lighting, composition, camera, look, failure and correction guidance | Keep concise to respect mobile context | IMPLEMENTING |
| D4 | Compiler has no real IR | Emit versioned IR: root, routing entries, branches, evaluation and tools | Compiler fails build on duplicate/missing IDs | IMPLEMENTING |

Selected routing design: one planning inference receives root + compact routing summaries. Alternative is two VLM calls (route then plan), which is more selective but doubles initial latency.

## Gate E — VLM contract / runtime safety

| ID | Problem | Production solution | Fallback / option | Status |
|---|---|---|---|---|
| E1 | Free-form JSON can be malformed | Native llama.cpp GBNF grammar constrains plan/evaluation JSON shape; Kotlin semantic validator remains mandatory | Parse failure => save reference | IMPLEMENTING |
| E2 | No inference timeout | Bounded plan/evaluation timeouts; timeout cancels generation and falls back safely | Reference/best candidate saved | IMPLEMENTING |
| E3 | Cancellation checked too late | Check cancellation before/after multimodal tokenization/eval and every decode step | Vision prefill cannot be force-aborted safely; stale result is discarded | IMPLEMENTING |
| E4 | Output strings/lists unbounded | Length/count bounds and sanitization in validator | Truncate safe metadata, reject unsafe tools | IMPLEMENTING |
| E5 | No static context optimization | Keep prompts compact and branch-selective; leave backend KV-prefix cache behind runtime interface | Correctness first; cache enabled only after profiling | IMPLEMENTING |
| E6 | CPU runtime may be slow | Adaptive image dimension, bounded tokens/threads, thermal retry reduction | Camera remains functional; AI timeout saves reference | IMPLEMENTING |

Acceleration option: add Vulkan/GPU backend only after device-matrix validation. CPU remains the compatibility baseline because an unqualified GPU path can reduce reliability across Android vendors.

## Gate F — Image / storage / quality

| ID | Problem | Production solution | Fallback / option | Status |
|---|---|---|---|---|
| F1 | Plain BitmapFactory ignores EXIF orientation | EXIF-aware decode/rotate/flip | Decode failure => save raw JPEG bytes | IMPLEMENTING |
| F2 | Natural result is needlessly recompressed | If no pixel processing is required, copy original JPEG bytes directly to MediaStore | Processed images use JPEG quality 96 | IMPLEMENTING |
| F3 | Processed save drops useful metadata | Copy safe EXIF capture metadata and normalize orientation | Metadata failure does not block photo save | IMPLEMENTING |
| F4 | Technical validity is only file size | Deterministic luma/dynamic-range/clipping/edge metrics; only catastrophic failures are hard vetoes | VLM remains aesthetic judge | IMPLEMENTING |
| F5 | Best fallback can ignore original | Reference is always a candidate with planner-provided bounded reference score | If score unavailable, reference receives conservative fallback priority | IMPLEMENTING |
| F6 | Process death can lose private reference | Store pending shutter references in app no-backup storage and recover or finalize them on next launch | Cache retries remain disposable | IMPLEMENTING |
| F7 | Storage-full behavior unspecified | Preflight free space and catch ENOSPC/MediaStore errors; keep pending reference until user can retry | Never delete last recoverable reference after failed save | IMPLEMENTING |

## Gate G — UX / lifecycle / resource governance

| ID | Problem | Production solution | Fallback / option | Status |
|---|---|---|---|---|
| G1 | Zoom UI lies about range | Drive zoom controls from active capability range | Only show 1x when exact ranges unavailable | IMPLEMENTING |
| G2 | EV UI is hard-coded | Drive min/max/index/step from CameraX state | Hide if unsupported | IMPLEMENTING |
| G3 | Torch local state can desync | Engine owns UI state and emits actual values after rebind/switch | Reset to actual state on failure | IMPLEMENTING |
| G4 | Generic gallery button only | Show last saved Director thumbnail and open its URI/gallery | Generic gallery when no image exists | IMPLEMENTING |
| G5 | Error state is immediately overwritten | Explicit terminal/session state reducer; error persists until next user action/readiness transition | Camera stays usable | IMPLEMENTING |
| G6 | Memory pressure does not unload AI | Application forwards trim-memory events; critical pressure cancels/unloads VLM before camera | Reload model when safe/foreground | IMPLEMENTING |
| G7 | Thermal policy too shallow | Thermal state controls AI resolution and recapture budget; critical disables AI session | Always preserve reference | IMPLEMENTING |

## Gate H — Tests / CI / release

| ID | Problem | Production solution | Fallback / option | Status |
|---|---|---|---|---|
| H1 | Only JSON parsing test exists | Add validator/tool/skill/quality/session-policy tests | Device tests remain separate | IMPLEMENTING |
| H2 | No shutter-contract regression test | Pure session policy tests assert no AI capture without shutter and max capture budget | Instrumented camera test required before release | IMPLEMENTING |
| H3 | Debug lint blocks APK iteration | Debug publishes APK + report; release remains strict | Existing V4 behavior retained | DONE-CODE |
| H4 | No device validation evidence | Add release checklist and device-matrix template | Cannot claim production-certified until run | OPEN |
| H5 | Release signing/distribution incomplete | Keep release build strict and document Codemagic signing variables | Debug remains installable | OPEN |

## Production fallback hierarchy

For every shutter event:

1. Best accepted Director result.
2. Best technically valid Director result.
3. Reference photo, optionally with safe deterministic look if processing succeeded.
4. Raw reference JPEG copied unchanged.
5. If MediaStore save fails, retain the pending reference privately and surface a recoverable storage error.

At no point may AI failure cause the user to lose the shutter capture.
