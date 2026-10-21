# Director

Director is an offline-first AI-native Android camera built around a **shutter-triggered** workflow. It behaves as a normal CameraX camera at all times. AI begins only after the user presses the white shutter.

## Product contract

The normal flow is:

`white shutter → full-quality private reference JPEG → that exact captured photo is analyzed by the on-device VLM → capability-filtered typed tool plan → optional corrected recapture → same VLM evaluates → bounded correction → save accepted/best result`.

Key guarantees:

- Preview frames never autonomously start VLM planning or photography.
- Every shutter first creates a protected reference image.
- AI failure, timeout, memory/thermal pressure, malformed output or unsupported camera controls cannot discard that reference.
- Camera mutations are temporary and user-selected state is restored.
- Total physical captures default to three per shutter event, including the reference.
- One VLM only: Qwen3-VL-2B-Instruct GGUF. Native camera APIs and deterministic quality metrics are not additional AI models.
- Model output is constrained JSON data, then semantic/tool validation occurs before Android APIs are touched.
- Unsupported capability is never fabricated.

See `PRODUCT_CONTRACT.md`, `PRODUCTION_REMEDIATION.md`, and `RELEASE_VALIDATION.md`.

## Model

Director downloads official Qwen GGUF files on first AI enablement:

- `Qwen3VL-2B-Instruct-Q4_K_M.gguf`
- `mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf`

Files are stored app-private and SHA-256 verified. The APK contains no model weights. Interrupted downloads can resume; verification is invalidated if native model loading fails so corrupted files can be rechecked/repaired.

## Camera and image behavior

- Android 10 / API 29+.
- arm64-v8a AI runtime.
- Preview + ImageCapture only; no continuous VLM frame analysis.
- Real CameraX zoom/EV/focus/flash capabilities drive both UI and AI tool contracts.
- OEM HDR/Night/Bokeh/Face Retouch are capability-gated and transactionally isolated.
- Manual-sensor/RAW characteristics may be reported, but manual ISO/shutter/WB are intentionally not executable AI tools until physical-device qualification proves them safe.
- Natural/identity output copies the original full-resolution JPEG bytes.
- Creative processing uses an EXIF-correct, memory-bounded 4096px long-edge path by default; the untouched reference remains protected until save succeeds.
- Pending references live under no-backup private storage and are recovered after process death.

## Build stack

- Android Gradle Plugin `8.13.2`
- Gradle `8.13`
- Kotlin / Compose plugin `2.3.10`
- JDK `17`
- compile/target SDK `36`
- Build Tools `36.0.0`
- NDK `28.2.13676358`
- CMake `3.22.1`
- CameraX `1.5.0`

## Build resilience

Director treats CI caches as untrusted input.

`scripts/bootstrap_android_sdk.sh` provisions Android packages one at a time, accepts valid NDK symlinks/wrappers, validates `source.properties`, `android.toolchain.cmake`, Clang/Clang++, aapt2 and CMake functionally, and performs bounded repair attempts. If Codemagic's shared SDK cannot be repaired, it automatically switches to a clean private SDK under `$HOME/.director/android-sdk`.

The custom `gradlew` verifies Gradle's official SHA-256 sidecar and ZIP integrity before extraction, rejects broken restored wrapper caches and cleanly redownloads them.

`scripts/fetch_llama_cpp.sh` verifies the pinned archive SHA-256 and the required CMake/llama/ggml/mtmd source files before a restored cache can be reused.

`scripts/ci_preflight.sh` rejects unsupported JDKs, missing host commands, incomplete SDK/NDK/CMake installations and dangerously low disk space before compilation starts.

`scripts/run_gradle_ci.sh` performs one recovery retry only for recognized dependency-cache/network corruption. Real Kotlin/Java/C++ compiler errors, lint errors and test failures are never hidden behind retries.

## Local build

```bash
cd Director
chmod +x gradlew scripts/*.sh scripts/*.py
./scripts/bootstrap_android_sdk.sh
./scripts/ci_preflight.sh
./scripts/fetch_llama_cpp.sh
./scripts/run_gradle_ci.sh /tmp/director-assemble.log assembleDebug
```

## Codemagic

Use **Director Android Debug V7** from a **new build** on the latest `main` commit. Do not use Re-run on an older V6/V5/V4 failure because Codemagic re-runs that job's old revision/configuration.

Debug V7:

1. identifies the exact commit/workflow and reports free disk;
2. self-heals or clean-room provisions the Android SDK/toolchain;
3. runs the deterministic JDK/SDK/NDK/CMake/Gradle/disk/product-contract preflight;
4. fetches and validates checksum-pinned llama.cpp source;
5. compiles the debug APK with targeted recovery for cache/network corruption only;
6. verifies the APK exists and is non-empty;
7. runs unit tests with the same targeted infrastructure recovery;
8. produces diagnostic lint reports while keeping the debug APK installable.

Release **Director Android Release V7** remains strict: APK/AAB, tests and lint must pass, and release signing must be configured before distribution.

## Production status

The implementation contains production-oriented fallbacks and validation gates, but no source review or CI configuration can honestly guarantee that every external service, network transfer, or future dependency will never fail. V7 is designed to automatically recover from the infrastructure failure classes encountered so far and to fail early with precise diagnostics for non-recoverable conditions. Physical-device qualification in `RELEASE_VALIDATION.md` remains a release requirement, especially for OEM extensions, native VLM latency, memory, thermal behavior and quality-threshold calibration.
