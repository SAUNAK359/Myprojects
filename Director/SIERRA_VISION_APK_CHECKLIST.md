# Sierra Vision APK — Production Implementation Checklist

This checklist is the implementation contract for the customer-facing Android APK. A checkbox may only be marked complete when the feature is wired to real runtime state, has a safe failure path, and is exercised by build/tests where practical. Demo-only/hardcoded product behavior is not acceptable.

## 0. Release invariants
- [ ] App builds from a clean checkout with `./gradlew :app:testDebugUnitTest :app:assembleDebug :app:lintDebug`.
- [ ] CircleCI/GitHub status is green on the exact final commit.
- [ ] No provider secrets, passwords, tokens, demo accounts, fake balances, or customer PII are embedded in the APK.
- [ ] Camera remains usable when Sierra AI/server/providers are unavailable.
- [ ] No credit can be consumed by Sierra Beta/local-only capture.
- [ ] Cloud credit settlement remains server-authoritative when backend is introduced; APK contains only the client contract/state representation.
- [ ] No final full-resolution photo is uploaded for ordinary Sierra analysis.
- [ ] UI never exposes provider/VLM internals to normal customers.

## 1. Product identity and visual system
- [ ] Product-facing name is **Sierra Vision**.
- [ ] Cloud intelligence is **Sierra AI**.
- [ ] On-device intelligence is **Sierra β**.
- [ ] Obsidian/graphite/ivory/Sierra-gold design tokens are centralized.
- [ ] White shutter = prepare; Sierra gold shutter = ready/capture.
- [ ] Edge-to-edge layout respects status/navigation/cutout insets.
- [ ] Motion/haptics are short, purposeful, and disabled/reduced when the platform requests reduced animation where possible.

## 2. Camera-first UX
- [ ] Launches directly into camera after onboarding/permission.
- [ ] Live Preview remains interactive during Sierra preparation.
- [ ] Tap-to-focus, pinch zoom, lens switch, exposure, flash/torch remain functional.
- [ ] Rule-of-thirds and level overlays are local and instant.
- [ ] Preparation states use photography language (reading scene, composing, locking exposure, finishing), never backend terminology.
- [ ] Error states never strand the user; local capture remains available.

## 3. Sierra β local engine
- [ ] Local quality telemetry is available before/without cloud: exposure, clipping, sharpness/detail, motion proxy, orientation/crop safety where supported.
- [ ] Local planner produces a safe plan using the same validated camera/look tool surface as Sierra AI.
- [ ] Local scene heuristics cover at least portrait-like, landscape-like, low-light, bright/high-dynamic-range, and general scenes.
- [ ] Local finish selection is deterministic, bounded, and adaptive to clipping/quality.
- [ ] Sierra β can prepare and capture an image with zero network/provider configuration.
- [ ] Sierra β path never consumes/requests a cloud credit.

## 4. Sierra AI + Sierra β cooperation
- [ ] Sierra AI never directly executes arbitrary Camera2 operations.
- [ ] Sierra AI outputs a canonical plan; Sierra β validators/capability registry decide what may execute.
- [ ] Cloud plan is enriched/validated with local quality telemetry before execution.
- [ ] Cloud failure/timeout automatically falls back to Sierra β instead of surfacing a dead end.
- [ ] Only one normal semantic cloud analysis image is uploaded per prepared shot.
- [ ] Final technical QA is local; no routine second VLM critique upload.

## 5. Aspect ratio and composition
- [ ] Aspect ratio has a user preference: Sierra Auto or manual ratio override.
- [ ] Sierra-selected ratio is represented in runtime state and UI.
- [ ] Crop overlay animates without rebinding the camera session.
- [ ] Final crop is subject/crop-safe and rejects impossible/overly destructive crops.
- [ ] Manual ratio override constrains Sierra composition rather than disabling intelligence.
- [ ] Supported output ratios are explicitly validated (4:3, 3:2, 4:5, 1:1, 16:9, 9:16, cinematic where feasible).

## 6. Capture/save pipeline
- [ ] Final still capture uses the quality path, not the analysis frame.
- [ ] Final output saves under `Pictures/Sierra Vision`.
- [ ] MediaStore pending/commit semantics prevent corrupt gallery entries.
- [ ] EXIF copied only from an allowlist; orientation normalized.
- [ ] Processing is memory-bounded on high-megapixel phones.
- [ ] Optional keep-original preference is implemented without risking final-save loss.
- [ ] Last saved URI/thumbnail opens safely with URI grants.
- [ ] Failed saves clean temporary files and do not report success.

## 7. Gallery/view/share/edit surface
- [ ] In-app Sierra gallery lists Sierra Vision images without requiring broad storage permission.
- [ ] Photo viewer supports share/delete/favourite/info where implemented.
- [ ] Sierra decision metadata is customer-readable, not raw JSON.
- [ ] Local social derivatives can be created without cloud/credits where implemented.
- [ ] Editing is non-destructive or explicitly saves a new derivative.

## 8. Settings/tools
- [ ] Customer settings expose photography controls, not provider keys.
- [ ] Aspect ratio, finish, timer/grid/level, save-original, privacy/local-only are persisted.
- [ ] Provider/API configuration is moved behind developer/internal configuration until laptop backend replaces it.
- [ ] Private Photography forces Sierra β and prevents cloud requests.

## 9. Server-ready contracts (APK side only in this phase)
- [ ] API base URL is configuration-driven; no fake production URL.
- [ ] Auth/session contract is isolated behind an interface.
- [ ] Credit balance is never hardcoded as authoritative product state.
- [ ] AI job model supports create/reserve/analyze/ready/complete/cancel/expire semantics for future backend wiring.
- [ ] One credit = one successfully completed Sierra AI final picture.
- [ ] Backend absence produces a clean Sierra β experience.

## 10. Performance/reliability
- [ ] ImageAnalysis uses latest-frame semantics and closes frames safely.
- [ ] Network/image preparation never blocks the main thread.
- [ ] Repeated shutter/user camera changes cancel stale sessions.
- [ ] Provider/local preparation has bounded deadlines.
- [ ] Camera controls settle by state/deadline rather than long blind sleeps.
- [ ] Processing avoids unnecessary full-resolution bitmap duplication.
- [ ] Lifecycle stop/foreground, camera rebind, rotation, memory pressure and thermal constraints have safe paths.

## 11. Security/privacy
- [ ] Existing development provider credentials remain encrypted and are not surfaced to consumers.
- [ ] No plaintext user passwords are ever stored by APK code.
- [ ] Sensitive logs never include API keys, full image payloads, auth tokens, or raw PII.
- [ ] Privacy/local-only mode is enforceable at the runtime boundary, not merely a UI toggle.
- [ ] Analysis frames are ephemeral unless a future explicit user feature requires retention.

## 12. Tests
- [ ] Provider response parser JVM tests remain green.
- [ ] VLM contract/tool validator tests remain green.
- [ ] Sierra local planner has deterministic unit tests.
- [ ] Aspect-ratio/crop policy has unit tests.
- [ ] Credit/job state model has unit tests without pretending to implement server authority.
- [ ] Save preference/metadata pure policy logic is unit-tested where possible.
- [ ] Existing regression tests are preserved.

## 13. Final release gate
- [ ] Search repository for stale customer-facing `Director` branding and intentionally classify remaining internal package/class names.
- [ ] Search repository for hardcoded user balances/accounts/server URLs/secrets.
- [ ] Verify exact branch head after all writes.
- [ ] Verify CI status for exact branch head.
- [ ] If CI fails, fix from the actual failing log before calling the APK ready.
- [ ] Document what is APK-complete versus what deliberately awaits the laptop server (auth, authoritative credits, payments, cloud job gateway/admin control plane).
