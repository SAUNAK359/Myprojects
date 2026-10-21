#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "gradlew", *sorted((ROOT / "scripts").glob("*.sh"))]

# Director's Codemagic macOS runner invokes /bin/bash, which is commonly Bash 3.2.
# Reject syntax/features introduced in Bash 4+ that would otherwise fail only at runtime.
FORBIDDEN = [
    (re.compile(r"\$\{[^}\n]+,,\}"), "Bash 4 lowercase parameter expansion (${var,,})"),
    (re.compile(r"\$\{[^}\n]+\^\^\}"), "Bash 4 uppercase parameter expansion (${var^^})"),
    (re.compile(r"\bmapfile\b"), "Bash 4 mapfile builtin"),
    (re.compile(r"\breadarray\b"), "Bash 4 readarray builtin"),
    (re.compile(r"\bdeclare\s+-A\b"), "Bash 4 associative arrays"),
    (re.compile(r"\blocal\s+-A\b"), "Bash 4 associative arrays"),
    (re.compile(r"\bglobstar\b"), "Bash 4 globstar shell option"),
    (re.compile(r"&>>"), "Bash 4 combined append redirection (&>>)")
]

errors: list[str] = []
for path in FILES:
    if not path.is_file():
        continue
    text = path.read_text(encoding="utf-8")
    if not text.startswith("#!/usr/bin/env bash") and not text.startswith("#!/bin/bash"):
        errors.append(f"{path.relative_to(ROOT)}: shell helper must declare bash explicitly")
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern, reason in FORBIDDEN:
            if pattern.search(line):
                errors.append(f"{path.relative_to(ROOT)}:{line_no}: {reason}: {stripped}")

if errors:
    print("Director shell compatibility check FAILED", file=sys.stderr)
    for error in errors:
        print(f" - {error}", file=sys.stderr)
    sys.exit(1)

print(f"Director shell compatibility check passed for {len(FILES)} Bash scripts (Bash 3.2 baseline)")
