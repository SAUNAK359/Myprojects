# Director Mode Android

Offline autonomous photography director for Android. The runtime does not request Internet permission and ships its vision models inside the APK.

## What it does

- CameraX preview, analysis, native still capture, tap focus/metering, zoom, torch, EV and Camera2 manual sensor controls where the OEM exposes them.
- Capability-gated native 4K still capture. The app never upscales a lower-resolution image and labels it 4K.
- Fully local MediaPipe pose, face and object perception.
- Data-driven shot-template engine; scene-specific behavior is described in JSON assets rather than hard-coded camera modes.
- Retargeted ghost skeletons, object-relative staging, alignment scoring, hysteretic pose lock, technical/moment gates and local best-frame ranking.
- Thermal governor and latest-frame backpressure to keep preview responsive.
- Dark professional studio UI.

## Android compatibility

`minSdk 24` (Android 7.0+) and `targetSdk 37`. Older Android releases cannot run the selected current LiteRT/MediaPipe stack. Hardware-dependent controls are capability-gated rather than emulated.

## Offline contract

The app manifest deliberately has **no `android.permission.INTERNET`**. The CI build downloads the official open-source model assets before compilation and packages them into `app/src/main/assets/`; the installed APK needs no model download.

Bundled build-time model sources:

- MediaPipe Pose Landmarker Lite
- MediaPipe Face Landmarker
- MediaPipe EfficientDet-Lite0 Object Detector

See `app/src/main/assets/MODEL_SOURCES.md`.

## Build

Requirements:

- JDK 17
- Gradle 9.6+
- Android SDK Platform 37

The repository workflow performs the reproducible build and uploads an installable debug APK artifact.

```bash
gradle :app:testDebugUnitTest :app:assembleDebug
```

A production Play Store release must be signed with the publisher's long-lived release key / Play App Signing. Do not distribute an ephemeral CI-signed package as a permanent production release.

## Architecture

```text
Camera + IMU
   -> FrameBroker (KEEP_ONLY_LATEST)
      -> pose / face / object perception
      -> deterministic image-quality analysis
      -> temporal SceneState
         -> DirectorBrain
            -> feasibility + template ranker
            -> PoseSynthesizer
               -> retargeted ghost overlay
                  -> AlignmentEngine
                     -> LOCKED + moment/technical gate
                        -> native high-resolution micro-burst
                           -> local best-frame ranker
                              -> MediaStore
```

## No-hardcoded-scene policy

Photographic scenarios and tunable Director behavior live in assets:

- `shot_templates.json`
- `director_config.json`

Kotlin implements generic geometry, inference, ranking, safety/capability gates and state machines. Adding a new shot family should normally mean adding/training data, not adding a scene-specific `if/else` block.

## Hardware truth

Android cannot guarantee identical camera controls on every device. Camera HALs differ. Director Mode discovers capabilities at runtime and enables only what the device actually exposes. Native 4K capture is additionally verified against the dimensions of the produced JPEG before save.
