# Director

Director is an offline-first AI-native Android camera. It behaves like a normal camera first, then layers one on-device VLM over the native camera pipeline to analyze a representative frame, select photography skills, produce validated tool calls, capture, evaluate the result, and retry up to three times.

## Core invariants

- Camera availability is more important than AI availability.
- Manual shutter always wins over autonomous work.
- One VLM only: Qwen3-VL-2B-Instruct GGUF.
- Model output is data, never executable code.
- Every tool call is validated against the active camera capability registry.
- Preview frames are sampled; the VLM never receives the live frame stream.
- Full-resolution capture and AI-resolution images are separate paths.
- AI is suspended before the camera under thermal or memory pressure.

## Model

Director downloads the official Apache-2.0 Qwen GGUF files on first AI startup:

- `Qwen3VL-2B-Instruct-Q4_K_M.gguf` (~1.11 GB)
- `mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf` (~445 MB)

Both files are SHA-256 checked before activation. The APK therefore does not contain model weights.

## Supported device baseline

- Android 10 / API 29 or newer.
- `arm64-v8a` device for the on-device AI runtime.
- The native runtime deliberately avoids a global `-march` requirement and enables llama.cpp KleidiAI runtime dispatch.
- The normal CameraX camera remains usable if the model is missing, unsupported, thermally suspended, or fails to load.

## Build stack

Pinned build stack:

- Android Gradle Plugin `8.13.2`
- Gradle `8.13`
- Kotlin / Compose compiler plugin `2.3.10`
- JDK `17`
- compile/target SDK `36`
- Android Build Tools `35.0.0`
- NDK `28.2.13676358`
- CMake `3.22.1`

## Build locally

The repository includes `scripts/bootstrap_android_sdk.sh`. It first looks for a usable Android SDK and `sdkmanager`. If the command-line tools are missing, it downloads the current pinned Google command-line-tools archive for the CI host architecture, verifies SHA-256, installs the exact platform/build-tools/NDK/CMake versions, and writes the resolved SDK path to `.director-android-sdk-root`.

```bash
cd Director
chmod +x gradlew scripts/*.sh scripts/*.py
./scripts/bootstrap_android_sdk.sh
SDK_ROOT="$(cat ../.director-android-sdk-root 2>/dev/null || cat .director-android-sdk-root)"
printf 'sdk.dir=%s\n' "$SDK_ROOT" > local.properties
./scripts/fetch_llama_cpp.sh
python3 scripts/compile_skills.py
./gradlew testDebugUnitTest lintDebug assembleDebug --stacktrace --no-daemon
```

## Codemagic

`codemagic.yaml` is stored at the repository root because Codemagic discovers YAML workflows from there.

Use **Director Android Debug V2** (`director-android-debug-v2`). Do not use an old build's **Re-run** button because a re-run is tied to the commit/configuration that originally failed. Start a **new build from the latest `main` commit**.

The V2 workflow:

1. prints its workflow name, Git commit, branch, host, and Java version;
2. discovers or bootstraps `sdkmanager` instead of assuming it is on `PATH`;
3. installs/verifies SDK 36, Build Tools 35.0.0, NDK 28.2, and CMake 3.22.1;
4. compiles and validates the hierarchical skills asset;
5. fetches a checksum-pinned llama.cpp source revision and rejects stale cache contents;
6. runs unit tests;
7. runs Android lint;
8. builds the debug APK and verifies that an APK was produced.

The release workflow is **Director Android Release V2**. Release signing should be configured through Codemagic environment variables/keystore before store distribution.

## Production note

No software-only implementation can guarantee identical camera controls on every Android phone. Director discovers device capabilities and degrades to safe CameraX auto behavior when a requested feature is unavailable. Real-device validation remains required for supported device families, especially for memory, thermal behavior, camera extension support, and actual on-device VLM latency.
