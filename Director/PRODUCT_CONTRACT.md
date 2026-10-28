# Sierra Vision v11 — Product Contract

This is the current behavioral contract for the Android application. Detailed architecture lives in `docs/sierra-v11/ARCHITECTURE.md`.

## Core promise

Sierra Vision is a computational camera, not an image-generation app. It may direct, capture, reconstruct and deterministically render a real camera photograph, but it must preserve scene evidence, identity, geometry, readable text and material truth.

## Authority

When Sierra AI is available:

- Gemini/Sierra is the photographic creative authority.
- Sierra β is the deterministic feasibility, perception, coaching and execution layer.
- Hardware/API limits are authoritative physical constraints.

When AI is unavailable or Private Photography is enabled:

- Sierra β remains functional locally.
- The UI must identify the β path rather than implying Gemini direction was used.

## User interaction

A Sierra-directed photograph is intentionally prepared before the final shutter:

1. Sierra samples a reduced analysis frame.
2. AI mode performs Gemini Call #1 and receives camera intent, `DirectorSpec` and complete `RenderSpec`. β mode creates a measured local camera/render plan.
3. Camera controls are validated, resolved to executable values, applied and compared with actual camera state.
4. Live Director measures the preview against executable photographic goals and coaches only allowed interventions.
5. AI mode performs Gemini Call #2 immediately before capture. This call may PASS or issue exactly one correction package.
6. Sierra β executes/verifies any one-time correction locally. There is no third Gemini call.
7. The final shutter captures the photograph. A safe capture program or P-RIF may be selected when supported.
8. The approved RenderSpec is executed deterministically.
9. The result is saved through MediaStore and only then shown as “Saved to Gallery.”

## Live Director

The capability-aware `director.v2` graph is machine executable, not prose guidance.

- Gemini identifies the exact primary entity in the planning frame and the target position in normalized final-composition coordinates.
- Local tracking continuously measures the same entity.
- Readiness uses hard/important/preference goals, confidence floors, ideal/acceptable tolerances, hysteresis and time-based stability.
- Unknown measurements never count as success.
- A stale READY state is reverified before shutter.
- Persistent no-progress is treated as a scene constraint; Gemini Call #2 may move the target once to the closest achievable version of the same concept.

## Director strategies / human intervention

Supported strategy vocabulary includes portrait, candid, group, fashion, food, product, architecture, landscape, night and general.

Intervention levels are observe, minimal, directed and high-direction.

For candid / observe / minimal behavior:

- posing cannot be a hard capture gate;
- required subject repositioning is forbidden;
- Sierra prioritizes photographer position, framing, timing, exposure and motion safety.

## Physical lighting and background direction

Sierra only coaches conditions it can measure locally.

Examples:

- left/right subject light balance;
- subject/background luminance separation expressed as an image-space EV-like ratio;
- off-subject background distraction score based on observable contrast/edge saliency;
- physical reframing/repositioning to simplify a background.

Sierra does not claim metric light placement without metric sensing and does not perform generative background deletion.

## Camera execution receipts

Every typed camera action can produce an auditable chain:

```text
requested → resolved → applied → actual
```

with one of:

- APPLIED
- CLAMPED
- REJECTED
- FAILED

Zoom/EV are numerically compared against actual camera state. AF/AE/AWB are treated as state machines rather than fake scalar values.

## Rendering

The active final renderer is `SierraRenderExecutor`.

AI:

```text
Gemini Call #1 RenderSpec → RenderExecutor
```

β:

```text
measured scene/quality → SierraBetaRenderPlanner → RenderSpec → RenderExecutor
```

Themes are optional descriptive priors only. They are not required for creative expression and do not override the RenderSpec.

ANIP-era creative finishing is legacy/historical and must not be presented as the current release gate.

## Failure behavior

- Invalid AI structured output is rejected before camera execution.
- Unsupported controls are not fabricated.
- Values such as EV may be quantized to the closest physically executable setting and then verified.
- A failed AI preparation does not silently claim AI success.
- A failed final correction does not trigger another AI loop.
- If final rendering cannot safely complete after real capture evidence exists, Sierra preserves the best safe evidence available.
- A raw/evidence-only save must not be labeled as a Sierra-rendered result.

## Memory / resource behavior

- Final processing is crop-first where the active path allows it.
- The render executor uses a bounded tile pixel buffer.
- P-RIF evidence is protected when in-place rendering would mutate it.
- Before/after comparison is a bounded cache preview, not another full-resolution persistent original.
- Gemini image-bearing requests stream JPEG/Base64 into the network request rather than retaining multiple large payload copies.
- Cancellation terminates active Gemini OkHttp calls.

## Privacy

- Private Photography disables cloud analysis.
- Cloud analysis uses a reduced analysis frame, not the final full-resolution photograph.
- Per-shot audit records contain observable execution data and never store hidden model reasoning.
- Original full-resolution JPEGs are saved separately only when Keep Original is enabled or when a safety fallback requires evidence preservation.

## Transparency

The UI exposes real pipeline events rather than fake progress:

- AI planning
- camera application/verification
- Live Director
- pre-shutter validation/correction
- capture
- P-RIF when used
- render
- technical safety
- MediaStore save

Successful MediaStore completion produces an explicit Saved to Gallery experience with final preview and, where applicable, a bounded before-Sierra-render comparison.

## Release claims

A hosted green compile/unit/lint/APK workflow proves software/package integrity for the commit that ran it.

Physical CameraX/Camera2/OEM/RAW/flash/thermal/memory behavior is a separate per-device qualification and must not be claimed from CI alone. See `docs/sierra-v11/DEVICE_QUALIFICATION.md`.
