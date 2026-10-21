#!/usr/bin/env python3
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"
OUT = ROOT / "app/src/main/assets/skills.json"

def parse(path: pathlib.Path):
    text = path.read_text(encoding="utf-8").strip()
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, flags=re.S)
    if not m:
        raise SystemExit(f"Invalid skill front matter: {path}")
    meta = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            raise SystemExit(f"Invalid metadata line in {path}: {line}")
        k, v = line.split(":", 1)
        meta[k.strip()] = v.strip()
    body = " ".join(line.strip() for line in m.group(2).splitlines() if line.strip())
    if not body or not meta.get("id") or not meta.get("kind"):
        raise SystemExit(f"Invalid skill: {path}")
    return meta, body

root = None
branches = {}
evaluation = []
tools = []
seen = set()
for path in sorted(SKILLS.rglob("*.SKILL.md")):
    meta, body = parse(path)
    sid = meta["id"]
    if sid in seen:
        raise SystemExit(f"Duplicate skill id: {sid}")
    seen.add(sid)
    kind = meta["kind"]
    if kind == "root":
        if root is not None: raise SystemExit("Only one root skill is allowed")
        root = body
    elif kind == "scene": branches[sid] = body
    elif kind == "evaluation": evaluation.append(body)
    elif kind == "tool": tools.append(body)
    else: raise SystemExit(f"Unknown skill kind {kind}: {path}")
if root is None or "general" not in branches:
    raise SystemExit("Skills require one root and a general scene branch")
payload = {"version":1,"root":root,"branches":branches,"evaluation":" ".join(evaluation),"tools":" ".join(tools)}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
print(f"Compiled {len(seen)} skills -> {OUT}")
