#!/usr/bin/env python3
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "app/src/main/java"

def fail(msg: str):
    print(f"PRODUCT CONTRACT VIOLATION: {msg}", file=sys.stderr)
    raise SystemExit(1)

required = [
    ROOT / "PRODUCT_CONTRACT.md",
    ROOT / "PRODUCTION_REMEDIATION.md",
    ROOT / "app/src/main/assets/skills.json",
]
for p in required:
    if not p.is_file():
        fail(f"missing {p.relative_to(ROOT)}")

all_code = "\n".join(p.read_text(encoding="utf-8") for p in SRC.rglob("*.kt"))
for forbidden in [
    "ImageAnalysis",
    "FrameSampler",
    "SceneChangeMonitor",
    "onPreviewFrame",
    "setFrameListener",
    "manualCapture",
    "captureToGallery",
]:
    if forbidden in all_code:
        fail(f"preview-autonomy/bypass symbol reintroduced: {forbidden}")

vm = (SRC / "com/saunak/director/ui/DirectorViewModel.kt").read_text(encoding="utf-8")
ui = (SRC / "com/saunak/director/ui/DirectorApp.kt").read_text(encoding="utf-8")
orch = (SRC / "com/saunak/director/director/DirectorOrchestrator.kt").read_text(encoding="utf-8")
policy = (SRC / "com/saunak/director/director/CapturePolicy.kt").read_text(encoding="utf-8")

if "fun shutterCapture()" not in vm or "orchestrator.onShutterPressed()" not in vm:
    fail("white shutter is not wired to DirectorOrchestrator.onShutterPressed")
if "ShutterButton" not in ui or "vm.shutterCapture()" not in ui:
    fail("UI shutter does not invoke vm.shutterCapture")

capture_i = orch.find("camera.capturePrivate(ref)")
plan_i = orch.find("vlm.plan(referenceBitmap")
if capture_i < 0 or plan_i < 0 or capture_i >= plan_i:
    fail("reference capture must occur before VLM planning")
if "storage.newPendingReference(session)" not in orch:
    fail("shutter reference is not stored in protected pending storage")
if "storage.saveRawJpeg(ref)" not in orch:
    fail("raw reference fallback is missing")
if "CONFIGURED_MAX_PHYSICAL_CAPTURES = 3" not in policy:
    fail("configured physical capture ceiling changed without product-contract review")

skills = json.loads((ROOT / "app/src/main/assets/skills.json").read_text(encoding="utf-8"))
if skills.get("version") != 2:
    fail("skills asset must use IR version 2")
if not skills.get("routing") or "general" not in skills.get("branches", {}):
    fail("skills routing/general fallback is incomplete")

print("Director product-contract validation passed")
