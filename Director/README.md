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

The current native runtime build targets `arm64-v8a`. It deliberately avoids a global `-march` requirement and enables llama.cpp KleidiAI runtime dispatch so older Arm64 devices do not execute unsupported premium-core instructions. The normal CameraX camera remains usable if the AI runtime cannot load. Qualify real launch devices before store release.

## Build locally

Requirements: Android SDK 36, JDK 17, NDK `28.2.13676358`, CMake `3.22.1`, curl, unzip, Python 3.

```bash
chmod +x gradlew scripts/*.sh
./scripts/fetch_llama_cpp.sh
echo "sdk.dir=$ANDROID_SDK_ROOT" > local.properties
./gradlew testDebugUnitTest lintDebug assembleDebug
```

## Codemagic

This project lives in the `Director/` folder of the GitHub repository. The repository-root `codemagic.yaml` changes into that folder, installs the exact NDK/CMake versions, fetches the pinned llama.cpp commit, runs tests/lint, and builds the APK/AAB.

Choose workflow `director-android-debug` first. It does not require signing. For release signing, add a Codemagic Android signing identity before Play Store distribution.

## Production note

No software-only implementation can guarantee identical camera controls on every Android phone. Director therefore discovers device capabilities and degrades to safe CameraX auto behavior when a requested feature is unavailable. Real-device validation is required for every supported device family.
