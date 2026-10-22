#!/usr/bin/env python3
from __future__ import annotations
import os, signal, subprocess, sys, time

def main() -> int:
    if len(sys.argv) < 3:
        print('Usage: run_with_timeout.py <seconds> <command> [args...]', file=sys.stderr)
        return 2
    try:
        timeout_seconds = int(sys.argv[1])
    except ValueError:
        print(f'Invalid timeout: {sys.argv[1]}', file=sys.stderr)
        return 2
    if timeout_seconds <= 0:
        print('Timeout must be > 0 seconds', file=sys.stderr)
        return 2
    command = sys.argv[2:]
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        elapsed = int(time.monotonic() - started)
        print(f"[Director timeout] Command exceeded {timeout_seconds}s (elapsed={elapsed}s): {' '.join(command)}", file=sys.stderr, flush=True)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        return 124

if __name__ == '__main__':
    raise SystemExit(main())
