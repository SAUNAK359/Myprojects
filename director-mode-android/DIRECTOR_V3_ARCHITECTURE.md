# Director V3 — Autonomous Computational Photographer

Director V3 is not a pose-overlay camera. It is an offline autonomous photography system whose job is to decide and execute the strongest photograph that the active Android camera can genuinely produce.

## Product contract

1. Camera preview and manual shutter are always available.
2. The AI observes the current scene and the actual camera capability graph.
3. The AI creates a complete `PhotoPlan`, not a generic pose.
4. Deterministic validation clamps every plan to real hardware support.
5. The camera applies supported controls continuously.
6. A deterministic readiness gate verifies technical stability.
7. White shutter means the treatment is still being designed/refined.
8. Yellow shutter means the treatment is locked and the frame is technically ready.
9. If an OEM extension is selected, Director temporarily enters the vendor capture pipeline, captures, then restores the AI analysis session.
10. No ghost skeleton is rendered. Pose/face perception is internal evidence only.

## End-to-end runtime

```text
CAMERA OPEN
   |
   v
Preview + ImageAnalysis + device sensors
   |
   +--> Human/face/object perception
   +--> Image physics (luma, contrast, clipping, sharpness, motion)
   +--> Camera capability graph
   +--> OEM extension availability
   |
   v
Scene evidence + 512px current visual snapshot
   |
   v
On-device photography VLM
   |
   v
PhotoPlan
   |
   v
Deterministic Plan Validator
   |
   v
Camera Executor
   |   zoom / EV / AF / AE / AWB / flash / aperture / scene mode / hardware effect
   |
   v
Readiness Evaluator
   |
   +--> not ready: continue observing and refining
   |
   +--> ready: shutter turns YELLOW
   |
   v
SHUTTER
   |
   +--> OEM extension capture if Bokeh/HDR/Night/Face Retouch/Auto is selected and supported
   |        |
   |        +--> restore normal AI analysis after capture
   |
   +--> otherwise native CameraX still capture / short burst
   |
   v
Local best-frame selection
   |
   v
SAVE
```

## What the autonomous VLM must judge

The VLM is a creative decision engine, not a low-level driver. It considers:

### Subject and story
- dominant subject and secondary subjects
- human / object / environment relationship
- portrait vs environmental portrait vs landscape vs action vs macro intent
- subject prominence
- facial presentation and eye emphasis
- whether context should be preserved or background separation should dominate

### Composition
- thirds
- golden-ratio placement
- center composition
- symmetry
- diagonal energy
- negative space
- leading lines
- depth layers
- frame-within-frame opportunities
- horizon placement and roll
- crop and headroom
- edge tension
- distracting objects and bright regions

### Perspective and lens
- current focal-length character
- perspective distortion risk
- subject distance
- whether zoom improves framing or only destroys quality
- whether wide environmental context is important
- whether a longer lens gives better facial perspective / compression

### Lighting
- overall luminance
- key-light direction
- front / side / back light
- softness vs harshness
- highlight clipping
- blocked shadows
- mixed-light risk
- whether HDR is justified
- whether Night is justified
- whether fill/flash is useful or destructive
- skin exposure priority when a face is present

### Exposure and motion
- exposure compensation intent
- clipping balance
- camera motion
- moving-subject risk
- whether burst capture is valuable
- whether low-light multi-frame processing is appropriate

### Focus and depth
- face priority
- subject priority
- center / macro / infinity intent
- metering point
- physical aperture choice when the camera exposes multiple apertures
- OEM Bokeh preference when subject separation is appropriate

### Color and rendering
- white-balance intent
- natural vs warm/cool cinematic rendering
- monochrome / sepia / other hardware-supported effects only when artistically justified
- saturation / contrast intent
- retouch should remain restrained
- no beautification simply because a human is present

### Computational capture
- standard capture
- OEM Auto extension
- OEM Bokeh extension
- OEM HDR extension
- OEM Night extension
- OEM Face Retouch extension
- RAW+JPEG intent when supported (capture implementation remains capability-gated)
- Ultra HDR intent when supported by the active output path

## PhotoPlan contract

`PhotoPlan` contains:

- theme
- confidence
- capture program
- composition guide
- OEM extension preference
- focus intent
- flash intent
- white-balance intent
- color style
- hardware effect
- output intent
- zoom
- exposure bias
- aperture intent
- focus point
- contrast/saturation/temperature intent
- vignette intent
- retouch strength
- background blur intent
- face priority
- burst count
- concise subject/camera guidance
- rationale

The VLM never supplies raw Camera2 constants.

## Deterministic hardware authority

`CameraEngine` maps high-level intent to the active camera only when supported.

Examples:

- requested zoom is clamped to CameraX's current zoom range
- EV is converted through the camera's compensation step/range
- AWB mode is checked against `CONTROL_AWB_AVAILABLE_MODES`
- effect is checked against `CONTROL_AVAILABLE_EFFECTS`
- scene mode is checked against `CONTROL_AVAILABLE_SCENE_MODES`
- AF mode is checked against `CONTROL_AF_AVAILABLE_MODES`
- aperture is snapped to `LENS_INFO_AVAILABLE_APERTURES`
- RAW intent falls back if the camera does not advertise RAW capability
- OEM extensions are used only when `ExtensionsManager` confirms availability

No device model-name table is allowed.

## OEM computational-photography bridge

CameraX Extensions can expose manufacturer implementations of:

- Auto
- Bokeh
- HDR
- Night
- Face Retouch

Extension sessions support still preview/capture but not ImageAnalysis. Director therefore uses a two-session strategy:

1. normal Director session = Preview + ImageAnalysis + ImageCapture
2. VLM reaches a stable plan
3. if an OEM extension is selected, temporarily bind Preview + ImageCapture through the OEM extension
4. capture
5. restore normal Director analysis session

This is deliberately capture-time only so semantic AI planning remains responsive.

## Readiness gate

The VLM does not decide READY by itself.

Readiness combines:

- plan confidence
- exposure quality
- sharpness
- camera stability
- clipping
- subject acquisition when the plan requires a face/subject

This prevents a confident model answer from turning the shutter yellow on a blurred or badly exposed frame.

## User interface

The old ghost skeleton is removed.

The live HUD shows only useful photographic information:

- AI-selected grid
- horizon/roll aid
- subject/face focus box when relevant
- readiness ring
- current theme/treatment
- white/yellow shutter state

Human pose landmarks remain internal and may be used later for subject framing, expression/moment analysis and face tracking, but they are not rendered as a target body.

## Model policy

The runtime is model-provider agnostic.

### Universal bundled tier
- SmolVLM2-500M LiteRT-LM
- Apache-2.0 family
- small enough to be realistically bundled compared with multi-gigabyte alternatives
- photography-specialized prompt and measured scene evidence reduce the amount of generic reasoning required

### Pro model tier (future build flavor)
- stronger permissively licensed VLM, selected by DirectorBench
- can be a larger SmolVLM2-class model if APK/RAM/thermal targets permit
- must remain optional rather than making every installation multi-gigabyte

### Model rejection policy
A model is not shipped only because it benchmarks fast. Distribution/commercial licensing must also be compatible with the product.

## Real-time policy

- camera preview: continuous
- pose/face/image physics: high frequency
- planning snapshot: latest-only, about every 650 ms
- VLM creative refresh: roughly every 1.2 s or immediately when material scene evidence changes
- no inference queue: stale visual decisions are not useful
- thermal pressure reduces semantic/VLM cadence before degrading preview or shutter responsiveness

## Features that require dedicated capture pipelines

The architecture has enums for future capture programs, but Director must not pretend they already work through a normal single JPEG capture.

### Panorama
Requires guided camera motion, frame overlap estimation, exposure/WB locking, homography/warping, seam finding, blending and final large-image output. It is intentionally excluded from autonomous VLM selection until that pipeline exists.

### Long exposure / computational motion blur
Requires a dedicated multi-frame RAW/YUV capture and alignment pipeline. It must not be simulated through a VLM setting.

### Custom cinematic color grading
`PhotoPlan` already carries rendering intent. Production-grade full-resolution grading should be implemented through a GPU capture/processing pipeline rather than decoding huge JPEGs into multiple full-size Java bitmaps.

### Custom portrait segmentation blur
Prefer OEM Bokeh when available. A Director-owned blur path requires high-quality segmentation/depth, edge refinement and lens-like blur rendering; it should not ship as a crude Gaussian mask.

## Benchmark gates

Features must beat the simpler path before shipping.

- manual shutter blocked by AI: 0
- fake upscaling: 0
- unsupported Camera2 control sent intentionally: 0
- VLM plan accepted without validation: 0
- stale VLM queue growth: 0
- capture crash target: <1 / 10,000
- real-time preview must remain responsive while VLM runs
- OEM extension output must be compared against non-extension output per scene class
- photo quality must be A/B tested against the stock camera, not only evaluated by code metrics

## Product north star

Director wins if an ordinary user can open it, point at a scene, wait for a yellow shutter, press once, and consistently obtain a photograph whose composition, camera treatment and capture mode would otherwise require photography knowledge.

The competitive goal is not to claim that third-party software can override every proprietary ISP decision on every Android device. The goal is to become the best decision layer above the hardware: understand the photograph, choose the best supported pipeline, configure it correctly, and capture at the right moment.
