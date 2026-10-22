#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "gradlew", *sorted((ROOT / "scripts").glob("*.sh"))]

# Director's Codemagic macOS runner invokes /bin/bash 3.2.
# Reject syntax/features introduced in Bash 4+ and a subtle Bash 3.2 + `set -u`
# hazard where a later variable in a multi-assignment `local` statement expands an
# earlier variable before that earlier assignment has taken effect.
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

LOCAL_ASSIGN_RE = re.compile(r"^\s*local\s+(.+)$")
ASSIGN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=")
PARAM_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)")

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

        local_match = LOCAL_ASSIGN_RE.match(line)
        if local_match:
            payload = local_match.group(1)
            assignments = list(ASSIGN_RE.finditer(payload))
            for index, assignment in enumerate(assignments):
                name = assignment.group(1)
                # Only references appearing after another assignment in the same `local`
                # statement are dangerous. Bash 3.2 expands RHS expressions before those
                # earlier local assignments are visible when `set -u` is active.
                if index == 0:
                    continue
                prefix_names = {m.group(1) for m in assignments[:index]}
                rhs_start = assignment.end()
                rhs_end = assignments[index + 1].start() if index + 1 < len(assignments) else len(payload)
                rhs = payload[rhs_start:rhs_end]
                refs = {m.group(1) for m in PARAM_RE.finditer(rhs)}
                risky = sorted(prefix_names & refs)
                if risky:
                    errors.append(
                        f"{path.relative_to(ROOT)}:{line_no}: Bash 3.2 local expansion hazard under set -u "
                        f"({name} references earlier local(s): {', '.join(risky)}): {stripped}"
                    )

if errors:
    print("Director shell compatibility check FAILED", file=sys.stderr)
    for error in errors:
        print(f" - {error}", file=sys.stderr)
    sys.exit(1)

print(f"Director shell compatibility check passed for {len(FILES)} Bash scripts (Bash 3.2 baseline)")
