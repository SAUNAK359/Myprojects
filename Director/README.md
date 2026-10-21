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

## Android SDK bootstrap resilience

`scripts/bootstrap_android_sdk.sh` verifies the actual SDK files/binaries instead of trusting package directories. It provisions packages one at a time and performs bounded retries. If Codemagic's shared SDK is incomplete or its downloaded archive is corrupt, Director purges partial package/temp/cache state and retries. If shared-SDK repair still fails, it automatically switches to a clean user-owned SDK under `$HOME/.director/android-sdk` and verifies the complete toolchain there before Gradle is allowed to run.

Required verified components are:

- `platforms;android-36`
- `build-tools;36.0.0`
- `ndk;28.2.13676358`
- `cmake;3.22.1`

## Local build

```bash
cd Director
chmod +x gradlew scripts/*.sh scripts/*.py
./scripts/bootstrap_android_sdk.sh
SDK_ROOT="$(cat ../.director-android-sdk-root 2>/dev/null || cat .director-android-sdk-root)"
printf 'sdk.dir=%s\n' "$SDK_ROOT" > local.properties
./scripts/fetch_llama_cpp.sh
./scripts/validate_project.sh
```

`validate_project.sh` compiles Skills IR v2, validates XML/JSON and enforces the product contract. With an Android SDK present it also runs unit tests, lint and debug assembly.

## Codemagic

Use **Director Android Debug V6** from a **new build** on the latest `main` commit. Do not use Re-run on an older V5/V4 failure because Codemagic re-runs that job's old revision/configuration.

Debug V6:

1. identifies the exact commit/workflow;
2. self-heals or clean-room provisions the Android SDK/toolchain;
3. verifies Android 36, Build Tools 36.0.0, NDK 28.2 and CMake 3.22.1 binaries;
4. compiles Skills IR v2 and enforces the shutter product contract;
5. fetches checksum-pinned llama.cpp;
6. compiles and verifies a non-empty debug APK;
7. runs unit tests and diagnostic lint.

Release **Director Android Release V6** remains strict: APK/AAB, tests and lint must pass, and release signing must be configured before distribution.

## Production status

The implementation contains production-oriented fallbacks and validation gates, but it is not truthful to call any Android camera app production-certified solely from source review/CI. Physical-device qualification in `RELEASE_VALIDATION.md` is a release requirement, especially for OEM extensions, native VLM latency, memory, thermal behavior and quality-threshold calibration.
