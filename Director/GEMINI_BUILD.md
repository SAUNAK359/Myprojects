# Director — Gemini Camera Build

This branch implements the Gemini-first Director flow. No Gemini API key is stored in GitHub, Gradle, Codemagic, resources, or source code. The user enters the key inside the app Settings UI and it is encrypted with Android Keystore.

## Product contract

1. App opens directly to a normal live camera preview.
2. CameraX/Camera2 continue normal AF/AE/AWB and device image processing.
3. The shutter is **white** before Director prepares a shot.
4. White press captures a private analysis frame. It is not written to Gallery.
5. Director sends Gemini the analysis frame, current camera state, generated device `config.md`, photography skills, and relevant approved recipe hints.
6. Gemini returns structured camera/look actions.
7. The local `ToolValidator` validates every action against live `CapabilityRegistry` ranges and supported modes before `ToolExecutor` can mutate the Director camera session.
8. Director takes a private intermediate shot and Gemini critiques it.
9. Director may apply corrective actions and repeat, with a maximum of three evaluations / three physical preparation captures.
10. The entire preparation transaction has a hard seven-second deadline. Gemini HTTP calls are coroutine-cancellable.
11. When approved, the live camera remains in the approved Director state and the shutter turns **yellow**.
12. Yellow press captures the real final frame and writes the final image to `Pictures/Director` through MediaStore.
13. The final corrected recipe is saved locally for later reuse/adaptation.

## Device capability onboarding

After camera permission is granted and the camera binds, Director immediately scans the phone cameras and generates:

`<app files>/director/config.md`

The generated skill contains the phone camera inventory and the active Director session's exact executable ranges, including supported zoom and exposure-compensation ranges, Camera2 ISO/exposure-time ranges, focal lengths, apertures, focus capabilities, RAW/manual flags, stabilization, flash, Camera Extensions and Director tool rules. The file is regenerated when the active camera changes.

`config.md` is advisory context for Gemini. It is **not** trusted for execution. The backend still validates every returned action against the live CameraX/Camera2 capability registry immediately before execution.

## Safety boundary

Gemini never talks directly to the Android camera API. The path is:

`Gemini JSON -> VlmContractValidator -> ToolValidator -> ToolExecutor -> CameraEngine -> CameraX/Camera2`

Director only mutates its own camera session. It does not patch, configure, write to, or persist changes into the OEM camera application, Camera HAL, firmware, system settings or another application.

## Included user features

- Full-screen CameraX preview
- White prepare / yellow final-capture interaction
- Tap-to-focus
- Pinch zoom
- Supported optical/digital zoom presets
- Exposure compensation control
- Torch
- Still flash OFF/AUTO/ON
- Front/rear camera switching
- Rule-of-thirds grid
- Composition guide overlay
- Gallery thumbnail / gallery opening
- Gemini connection status
- AI preparation stage/status feedback
- Quality score feedback
- Encrypted Gemini key management
- Device-specific camera skill generation
- Scene/context/subject/lighting/depth/composition/aesthetic reasoning prompts
- Critic/refinement loop
- Three-attempt ceiling
- Seven-second hard preparation deadline
- Thermal policy support
- Private intermediate files
- Crash-safe pending-photo recovery inherited from the Director storage layer
- Professional look pipeline and Camera Extension support where available
- Approved recipe memory and future-scene adaptation

## Gemini key setup

Install the app, grant Camera permission, open the gear icon, paste the Gemini API key, and tap **Connect Gemini**. The key is stored encrypted by Android Keystore. Removing the key from Settings disables Director AI but leaves the camera interface usable.

## Codemagic

The repository root contains `codemagic.yaml`. Use workflow:

`director-android-debug`

for the installable debug APK. The workflow runs unit tests, Android lint and `assembleDebug`, then publishes the APK as an artifact.

The release workflow is:

`director-android-release`

and builds both APK and AAB after release unit tests and lint. Signing is only used if the standard Codemagic keystore environment variables are present.

This Gemini version intentionally removes the old llama.cpp/CMake build from Gradle and Codemagic. Old source/reference files may remain in the repository but are not part of the Gemini runtime path.

## Validation note

The branch is configured for a clean Codemagic build, but an actual Codemagic run or physical Android device run is still the authoritative validation for OEM-specific camera behavior. Camera Extensions and some low-level sensor capabilities vary by manufacturer and model, so unsupported controls are always rejected locally rather than assumed.
