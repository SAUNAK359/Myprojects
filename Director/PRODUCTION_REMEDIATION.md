# Director Production Remediation Register

Status values: `OPEN`, `IMPLEMENTING`, `DONE-CODE`, `DEVICE-VALIDATION`.

This register is the source of truth for Director hardening. `DONE-CODE` means implemented and covered by static/unit checks where applicable. `DEVICE-VALIDATION` means code exists but physical-device qualification remains mandatory.

## Gate A — Product / shutter contract

| ID | Production solution | Status |
|---|---|---|
| A1 | Remove VLM preview trigger; shutter capture is sole session trigger | DONE-CODE |
| A2 | White shutter captures private full-quality reference then starts Director; AI-unavailable saves reference | DONE-CODE |
| A3 | Exact shutter JPEG is preserved and an EXIF-correct sampled decode of it is the first VLM image | DONE-CODE |
| A4 | No autonomous `captureTemp`/preview-driven photo path exists | DONE-CODE |
| A5 | Generation tokens invalidate stale AI before mutations/recaptures; every superseded shutter retains its reference fallback | DONE-CODE |

## Gate B — Camera capability / transactions

| ID | Production solution | Status |
|---|---|---|
| B1 | CameraX executable ranges + conservative CameraManager informational characteristics; unknown controls unsupported | DEVICE-VALIDATION |
| B2 | Normalized AF/AE/AWB metering tool with capability gating | DEVICE-VALIDATION |
| B3 | OFF/AUTO/ON still-flash separate from torch | DEVICE-VALIDATION |
| B4 | ExtensionsManager-gated AUTO/HDR/Night/Bokeh/Face Retouch; bind failure rolls back to normal camera | DEVICE-VALIDATION |
| B5 | User camera state restored after session/cancellation | DEVICE-VALIDATION |
| B6 | Typed validation, clamping and deterministic OEM-extension conflict isolation | DONE-CODE |
| B7 | Passive Camera2 AF/AE/AWB convergence observation with bounded timeout/grace fallback | DEVICE-VALIDATION |
| B8 | UI state/ranges originate from active CameraX camera state | DEVICE-VALIDATION |

Manual ISO/shutter/WB and RAW capture are intentionally **not executable AI tools** yet. Hardware support may be reported, but exposing lower-level overrides without vendor/device qualification would violate the unsupported-capability rule. Option: add per-device manual-control profiles after the release matrix proves them stable.

## Gate C — Typed tool system

| ID | Production solution | Status |
|---|---|---|
| C1 | Capability-filtered contract defines every tool ID, meaning, parameter and bounds | DONE-CODE |
| C2 | Evaluator receives the same current capability-filtered delta contract | DONE-CODE |
| C3 | Executor returns applied/rejected calls with reasons and passes rejection context forward | DONE-CODE |
| C4 | Finite/range/enum/count validation before execution | DONE-CODE |
| C5 | Creative looks have bounded strength and deterministic identity blending | DONE-CODE |

## Gate D — Skills architecture

| ID | Production solution | Status |
|---|---|---|
| D1 | Skills IR v2 separates compact routing summaries from full branch bodies | DONE-CODE |
| D2 | Selected branch is validated/falls back to `general`; evaluation gets only that branch | DONE-CODE |
| D3 | Scene branches contain subject/lighting/composition/camera/look/failure/correction guidance | DONE-CODE |
| D4 | Compiler emits versioned root/routing/branches/evaluation/tools IR and fails malformed source | DONE-CODE |

Selected option: one planning inference receives root + compact routing index. Alternative: route-then-plan with two VLM calls for stricter branch isolation but higher first-shot latency.

## Gate E — VLM contract / runtime safety

| ID | Production solution | Status |
|---|---|---|
| E1 | Native pinned llama.cpp JSON GBNF guarantees syntactically valid JSON; Kotlin serialization + semantic + tool validators enforce exact schema/ranges | DONE-CODE |
| E2 | Plan/evaluation bounded timeouts cancel native generation and preserve reference/best candidate | DEVICE-VALIDATION |
| E3 | Native cancellation checks before/after tokenization/prefill and every decode; stale sessions are discarded | DEVICE-VALIDATION |
| E4 | Bounded strings/scores/lists/call counts | DONE-CODE |
| E5 | Compact branch-selective prompts are implemented; KV-prefix caching intentionally deferred until profiling proves benefit | DONE-CODE |
| E6 | Adaptive AI image size/thread/token/capture budget; CPU baseline requires latency/thermal qualification | DEVICE-VALIDATION |

Option: add Vulkan/GPU acceleration only after device-matrix testing. CPU is the compatibility baseline rather than shipping an unqualified vendor-fragile GPU path.

## Gate F — Image / storage / quality

| ID | Production solution | Status |
|---|---|---|
| F1 | EXIF-aware sampled decode/rotate/flip | DEVICE-VALIDATION |
| F2 | Identity result raw-copies original full-resolution JPEG; processed path JPEG quality 96 | DEVICE-VALIDATION |
| F3 | Processed results copy safe EXIF and normalize orientation | DEVICE-VALIDATION |
| F4 | Deterministic luma/clipping/dynamic/edge catastrophic guard plus VLM aesthetic evaluation | DEVICE-VALIDATION |
| F5 | Original reference is always a candidate; best valid candidate fallback | DONE-CODE |
| F6 | Pending references stored in no-backup private storage and recovered after process death | DEVICE-VALIDATION |
| F7 | Free-space preflight; MediaStore failure keeps pending reference rather than deleting last recoverable copy | DEVICE-VALIDATION |

Selected option: creative processing uses a 4096px long edge to bound memory. Identity results remain original full-resolution JPEG bytes. Alternative full-resolution creative processing should use a tiled/native/GPU pipeline after device qualification, not a giant Java bitmap.

## Gate G — UX / lifecycle / resource governance

| ID | Production solution | Status |
|---|---|---|
| G1 | Zoom UI generated from real min/max zoom | DEVICE-VALIDATION |
| G2 | EV UI generated from active index range/step | DEVICE-VALIDATION |
| G3 | Torch/flash UI reads actual engine state after changes/rebinds | DEVICE-VALIDATION |
| G4 | Last saved URI renders as thumbnail and opens specific photo; generic gallery fallback | DEVICE-VALIDATION |
| G5 | Session terminal/error state is not overwritten by an orchestrator `finally` readiness emission | DONE-CODE |
| G6 | Critical memory pressure cancels/unloads VLM while camera stays alive; foreground can reload | DEVICE-VALIDATION |
| G7 | Thermal state controls AI resolution and physical-capture budget; critical state saves reference without AI | DEVICE-VALIDATION |

## Gate H — Tests / CI / release

| ID | Production solution | Status |
|---|---|---|
| H1 | Schema, semantic validator, tool validation and capture-policy unit tests | DONE-CODE |
| H2 | CI product-contract guard forbids preview autonomy/bypass and enforces capture-before-plan ordering; physical camera regression remains required | DEVICE-VALIDATION |
| H3 | Debug CI publishes APK/lint diagnostics; Release V5 remains strict | DONE-CODE |
| H4 | `RELEASE_VALIDATION.md` defines physical-device, soak, failure and quality-calibration gates | DEVICE-VALIDATION |
| H5 | Release signing remains environment/keystore configuration; no credentials are committed | OPEN |

## Production fallback hierarchy

For every shutter event:

1. First accepted Director result.
2. Best technically valid Director result.
3. Reference photo with safe deterministic look if that is the chosen best candidate.
4. Raw reference JPEG unchanged.
5. If MediaStore save fails, keep the pending reference privately for recovery.

At no point may AI failure be allowed to erase the user's shutter capture.

## Quality thresholds

The current acceptance defaults (`score >= 0.78`, `technical >= 0.65`) are engineering defaults, not calibrated scientific thresholds. They remain a `DEVICE-VALIDATION` item until measured against a representative labeled photo corpus.
