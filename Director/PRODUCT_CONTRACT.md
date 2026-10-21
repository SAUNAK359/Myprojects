# Director Product Contract — Shutter-Triggered AI Photography

This file supersedes any earlier Director design text that implies autonomous VLM-triggered photography from sampled preview frames.

## Non-negotiable user contract

1. The live preview is native camera UI only. The VLM never autonomously starts a photo session from preview frames.
2. Pressing the white shutter is the only normal entry point into a Director photo session.
3. The shutter first captures a full-quality reference JPEG. That exact captured photograph is the first image provided to the VLM.
4. The reference image is kept privately until the session is finalized. If AI is unavailable, cancelled, times out, overheats, runs out of memory, crashes, produces invalid output, or cannot safely execute a plan, the reference photo is saved rather than lost.
5. The VLM may propose a photography objective and typed tool calls. It never executes Android APIs or arbitrary text.
6. Every tool call is schema-validated, range-clamped, conflict-resolved, and checked against the active device capability registry before execution.
7. AI camera mutations are temporary. User-selected zoom/exposure/torch/lens state is restored after a session or cancellation.
8. Director may recapture only when a validated plan materially requires a new camera state. Post-processing-only changes do not force a physical recapture.
9. The same VLM evaluates corrective captures. Total physical captures are bounded to 3 per shutter event by default: the reference plus at most 2 corrective recaptures.
10. Director saves the first passing result or the best technically valid result. The reference is always eligible as the safety fallback.
11. Temporary/reference/retry files never pollute the gallery and are cleaned after finalization. Pending references are recoverable after process death.
12. The user can press the shutter again or change a camera control while AI is working. New user intent cancels stale AI work; stale work cannot later mutate the camera.
13. Camera availability is more important than AI availability. Thermal or memory pressure disables AI before it disables the camera.
14. One VLM supplies all learned intelligence. Native camera APIs and deterministic image statistics are allowed and are not additional AI models.
15. Skills are compiled into root/routing/branch/evaluation/tool sections. The planning pass receives root rules plus a compact routing index, not every full branch. Evaluation/correction receives only the selected branch.

## Default capture budget

Selected production default: **3 total physical captures** (reference + up to 2 recaptures).

Alternative A: reference + 3 recaptures (4 total) for higher quality at the cost of latency, heat and user surprise.
Alternative B: reference + 1 recapture for low-end devices or severe thermal state.

The thermal policy may reduce the budget but never increase it beyond the configured maximum.
