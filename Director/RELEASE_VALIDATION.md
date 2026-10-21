# Director Release Validation Gate

Passing compilation is necessary but not sufficient for a production camera application. Director must not be described as production-certified until the applicable checks below are executed on physical devices and recorded.

## Required device coverage

Use at least one device from each supported hardware family where possible, and cover API 29 plus modern API levels through the current target API 36. The matrix should include multiple SoC/camera vendors rather than relying on one emulator or one Pixel.

Recommended families: Pixel/Tensor; Samsung (Exynos and Snapdragon where available); Qualcomm reference-like devices such as OnePlus/Motorola; Xiaomi/Redmi/Poco; Oppo/Realme/Vivo; and at least one lower-memory arm64 device.

Record for every device: model, SoC, RAM, Android/API, rear/front camera IDs, supported extensions, max zoom, EV range, model-load time, plan latency, evaluation latency, peak RSS/PSS, thermal status, crashes/ANRs and result.

## Functional gates

- Camera opens immediately with AI absent, declined, downloading, unavailable or corrupt.
- White shutter always captures a protected reference before any VLM call.
- Airplane mode after model installation does not break Director.
- Declining model download leaves a fully usable camera.
- Interrupted model download resumes or safely restarts; corrupt model/projector is rejected and self-heals.
- Rear/front switch, tap focus, pinch zoom, EV, torch and still-flash reflect real device state.
- Unsupported controls never appear in the VLM executable contract.
- HDR/Night/Bokeh/Face Retouch are exposed only when ExtensionsManager reports support; failed extension bind restores normal camera.
- New shutter while AI is processing supersedes stale AI without losing either shutter reference.
- User focus/zoom/EV/torch/lens change cancels stale AI camera mutations.
- Total physical captures never exceeds the configured per-shutter budget.
- AI timeout, malformed semantic output, native cancellation, thermal suspension and memory pressure all preserve/save the reference.
- Process kill after reference capture but before final save recovers the pending reference on next launch.
- MediaStore failure / low storage leaves the pending reference recoverable.
- All four device rotations and mirrored front-camera captures have correct final orientation.
- Natural/identity result remains original JPEG bytes; processed result has expected orientation and safe EXIF metadata.
- Gallery thumbnail opens the actual last saved URI.

## Reliability / soak gates

- 100 consecutive shutter sessions with no leaked pending/attempt files after successful finalization.
- Rapid-shutter test: repeated presses during planning/evaluation/recapture.
- 30-minute camera preview + periodic AI capture soak.
- Thermal soak until MODERATE/SEVERE policy is exercised; camera must remain available.
- Memory-pressure test with AI unload/reload while camera remains operational.
- Background/foreground and screen rotation loops.
- App process death at each session stage: reference captured, planning, applying, recapture, evaluating, saving.

## Quality calibration gate

`CapturePolicy` acceptance thresholds (score 0.78, technical 0.65) are engineering defaults, not scientifically calibrated values. Before release, evaluate a representative labeled corpus across portrait, landscape, night, food, architecture, automotive, macro and general scenes. Measure false accepts, unnecessary recaptures, final-vs-reference preference rate, latency and subjective quality. Adjust thresholds only with recorded evidence.

## Performance targets to record

- Camera-first-launch time and preview responsiveness.
- Reference shutter latency.
- Model load time and memory footprint.
- Plan and evaluation latency p50/p95.
- Peak process memory during 1024px AI decode and 4096px creative processing.
- Battery/thermal behavior over a 30-minute session.

No fixed latency number is claimed until physical-device measurement exists.

## Release-only gates

- `Director Android Release V5` passes unit tests and strict Android Lint.
- R8/minification release build installs and launches on a physical arm64 device.
- Release keystore is supplied through protected Codemagic environment variables; no key or password is committed.
- Verify model URLs/hashes and pinned llama.cpp revision before tagging.
- Review Play/privacy disclosures if distributing through a store.
