# Sierra Vision v11 Physical Device Qualification

This checklist is intentionally separate from hosted CI. A green Kotlin/unit/lint/APK build proves software/package integrity, not real sensor/OEM behavior.

A device is **qualified** only after the applicable checks below pass on that physical phone/OS/camera stack.

## Identity

Record:

- manufacturer / model
- Android version / security patch
- app commit SHA / APK checksum
- CameraX version
- available camera IDs and lens facing
- supported OEM extension modes
- RAW/manual-sensor capability
- flash / torch capability
- verified local expert-model capability hash (`director.v2`)

## Camera lifecycle

Pass criteria:

- cold launch binds preview
- foreground/background releases and rebinds cleanly
- front/back switch works repeatedly
- no camera-in-use leak after 20 lifecycle cycles
- tap focus remains aligned with preview coordinates
- pinch zoom remains within reported bounds

## Requested → actual telemetry

For each supported lens:

1. Request minimum/mid/maximum zoom; verify CameraX actual ratio within configured tolerance.
2. Request EV values around negative/middle/positive range; verify quantization to device EV step and actual compensation index.
3. Trigger focus on near/center/far targets; record AF state transitions and timeout behavior.
4. Observe AE/AWB searching → converged/locked behavior under stable light.
5. Verify flash OFF/AUTO/ON and torch mutual exclusion.
6. Verify OEM-extension entry and safe release before direct camera controls.

No feature should enter the Device Foundation as reliable if this hardware does not execute it reliably.

## Live Director

Exercise at minimum:

- portrait placement
- photographer move left/right and closer/farther
- target-box stability
- pose goal when pose expert is installed
- subject/background lighting separation
- left/right light-balance target
- background-distraction reframing
- candid/minimal strategy: confirm no intrusive blocking pose instruction
- object/product primary entity if generic visual anchor supports it on the scene

Verify:

- UNKNOWN never becomes READY
- noisy thresholds do not flicker READY due to hysteresis
- readiness requires sustained time-window stability
- tracking does not jump to a second similar subject
- stale READY after movement returns to coaching

## Capture / P-RIF

Where supported:

- normal JPEG capture
- smart burst
- RAW/P-RIF path
- P-RIF safety JPEG preservation
- P-RIF reconstruction integrity failure path
- fallback when RAW execution cannot start
- sensor/capture telemetry association
- final capture after a one-time Gemini correction does not reuse a stale RAW/burst program

## Rendering and memory

Test representative sensor resolutions and final crops while recording Java/native heap.

Required observations:

- no OOM under repeated final renders
- render pixel auxiliary buffer remains bounded (tiled executor)
- comparison artifact never decodes/retains a second full-resolution original
- source/crop/render ownership releases buffers after save
- P-RIF reconstruction evidence is not mutated accidentally
- low-memory device path obeys processing dimension limits

Recommended stress case:

- 20 consecutive photos
- alternating AI / β
- portrait + night + high-DR scenes
- keep-original ON/OFF
- background/foreground app cycles between some shots

## Thermal

Run sustained camera session long enough to reach device thermal pressure.

Confirm:

- live analysis cadence reduces according to thermal policy
- AI/local expert scheduling does not deadlock
- capture still completes or fails visibly
- no hidden quality claim when thermal policy forces β/local degradation

## Gemini/network

On a device with direct Gemini configured:

- normal Call #1 + pre-shutter Call #2
- cancel during upload
- cancel during SSE response
- network loss before Call #1
- network loss between live convergence and Call #2
- slow streaming without total-call timeout
- verify UI clearly indicates β fallback/degradation
- confirm no third Gemini call after a correction

## MediaStore / completion

- final URI opens from external Gallery
- save completion appears only after MediaStore commit
- Open and Share work with granted URI permissions
- before/Sierra comparison displays bounded cache preview when render exists
- raw/evidence-only save does not pretend a Sierra render was applied
- app restart preserves Gallery image even though comparison cache may be gone

## Audit

For representative success/failure shots, inspect app-private audit JSON and verify it contains:

- mode / scene / objective
- execution receipts
- pipeline stages
- actual camera-state summary
- fallback reason when present
- final URI and final processing result

It must not contain hidden chain-of-thought/reasoning text.

## Qualification result

Use one of:

- `QUALIFIED` — all applicable checks passed on this exact device/software combination.
- `QUALIFIED_WITH_LIMITS` — listed capabilities are disabled/removed from Device Foundation; remaining path passed.
- `NOT_QUALIFIED` — blocking camera, memory, telemetry or stability defect remains.

Hosted CI must never be reported as physical qualification.
