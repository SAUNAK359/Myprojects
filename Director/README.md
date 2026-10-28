# Sierra Vision / Director — v11 Production Candidate

Sierra Vision is an Android computational camera built around a strict authority split:

- **Gemini / Sierra** chooses the photograph when AI mode is available.
- **Sierra β** measures the scene, enforces device/API feasibility, executes camera controls, coaches the photographer and verifies readiness.
- **CameraX / Camera2 + hardware** define the physical boundary.
- **P-RIF** is the computational reconstruction path when selected and supported.
- **SierraRenderExecutor** is the active deterministic final renderer. ANIP-era creative finishing is legacy/historical, not the active v11 authority path.

The current production architecture is documented in [`docs/sierra-v11/ARCHITECTURE.md`](docs/sierra-v11/ARCHITECTURE.md).
Physical-device qualification is deliberately separate from hosted build validation and is defined in [`docs/sierra-v11/DEVICE_QUALIFICATION.md`](docs/sierra-v11/DEVICE_QUALIFICATION.md).

## Active v11 shot pipeline

```text
Device Foundation + current camera state + situation signature + recent creative memory
        ↓
Gemini Call #1
  photographic intent
  camera intent
  capability-aware DirectorSpec
  complete RenderSpec
        ↓
Sierra β camera feasibility/execution
        ↓
requested → resolved → applied → actual camera verification
        ↓
Live Director
  track → measure → compare → coach → stable readiness
        ↓
Gemini Call #2 — PRE-SHUTTER and FINAL Gemini call
  PASS or exactly one bounded correction
        ↓
local execution + fresh local verification
        ↓
FINAL SHUTTER
        ↓
JPEG / smart capture / P-RIF as selected by capture compiler
        ↓
cached Call #1 RenderSpec
        ↓
SierraRenderExecutor
        ↓
technical safety / MediaStore
        ↓
Saved-to-Gallery completion + optional before/Sierra comparison
```

There is no third Gemini correction loop and no post-capture Gemini creative call.

## Capability-aware Live Director

The backend-owned `director.v2` contract is generated from verified local capabilities. Gemini can only request goal types Sierra can measure locally.

Current goal families include:

- primary-entity placement / scale / visibility
- subject luminance
- left/right light balance
- subject-to-background image-space EV separation
- background distraction reduction through physical reframing/repositioning
- motion stability
- technical sharpness
- relative-depth separation when depth is installed
- pose alignment when pose is installed
- face placement when face detection is installed

Director strategies include portrait, candid, group, fashion, food, product, architecture, landscape, night and general. The intervention policy controls how intrusive coaching may be. In candid/minimal/observe modes, pose and subject repositioning cannot become blocking requirements.

## AI and β rendering converge

AI mode:

```text
Gemini RenderSpec → SierraRenderExecutor
```

β mode:

```text
measured TechnicalQuality + SceneSignals + SituationSignature
→ SierraBetaRenderPlanner
→ SierraRenderSpec
→ SierraRenderExecutor
```

Installed theme names are optional descriptive priors only. The rich RenderSpec is the artistic execution contract.

## Production memory controls

- Crop is determined before expensive rendering in the active file path.
- The final renderer uses a bounded tile pixel buffer instead of a full-frame auxiliary `IntArray`.
- P-RIF reconstruction evidence is protected from accidental in-place mutation when required.
- Before/after comparison uses a bounded ~1280 px cache JPEG rather than a second full-resolution image.
- Per-shot Gemini image requests stream JPEG → Base64 directly into OkHttp, avoiding simultaneous JPEG byte-array + Base64-string + full-request-string heap copies.
- Gemini SSE uses cancellable asynchronous OkHttp calls.

## Observability

Sierra v11 exposes actual runtime events rather than collapsing the shot into a single opaque status string.

The UI can show:

- Gemini planning
- plan validation
- requested / resolved / applied / actual camera controls
- clamped/rejected camera actions
- Live Director coaching/readiness
- pre-shutter semantic validation
- one final correction
- capture
- P-RIF reconstruction
- Sierra render
- technical safety
- MediaStore save
- explicit Saved to Gallery completion

Each saved shot also writes a bounded app-private audit JSON containing observable execution evidence; hidden model reasoning is never stored.

## Build / validation

The GitHub Android workflow performs:

1. Java 17 / Android SDK 36 setup
2. Kotlin compile
3. unit tests
4. lint
5. debug APK assembly
6. APK signature verification
7. zipalign verification
8. ZIP integrity / `classes.dex` check
9. artifact checksum/upload

A green hosted build proves source/package integrity. It does **not** prove OEM-camera behavior on a physical phone. Flash, extensions, RAW/P-RIF, thermal behavior, real memory peaks and Camera2 telemetry must still pass the device qualification contract for each hardware/OS combination.

## Privacy / security

- Camera and Internet are the primary runtime permissions.
- Cleartext network traffic is disabled.
- Server URL must be blank or HTTPS at build time.
- Provider API keys are not hard-coded in source.
- Model artifacts are checksum-pinned and verified before use.
- Private Photography disables cloud analysis and keeps Sierra β local.
- Per-shot audit records store observable execution metadata, not hidden chain-of-thought.

## Historical documentation

Any file that describes Sierra v3/v8/v10 or ANIP as the active final creative authority should be treated as historical unless it explicitly references this v11 architecture document.
