#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: run_gradle_ci.sh <log-file> <gradle args...>" >&2
  exit 2
fi

LOG="$1"
shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMEOUT_RUNNER="$ROOT/scripts/run_with_timeout.py"
cd "$ROOT"
mkdir -p "$(dirname "$LOG")"

GRADLE_ARGS=("$@")
TOTAL_BUDGET="${DIRECTOR_GRADLE_TIMEOUT_SEC:-3600}"
START_EPOCH="$(date +%s)"

remaining_budget() {
  local now elapsed remaining
  now="$(date +%s)"
  elapsed=$((now - START_EPOCH))
  remaining=$((TOTAL_BUDGET - elapsed))
  (( remaining > 0 )) || remaining=0
  printf '%s' "$remaining"
}

run_once() {
  local refresh="$1" remaining status
  remaining="$(remaining_budget)"
  if (( remaining < 60 )); then
    echo "[Director Gradle] Less than 60s remains in the Gradle budget; refusing another attempt." | tee -a "$LOG"
    return 124
  fi

  set +e
  if [[ "$refresh" == "1" ]]; then
    python3 "$TIMEOUT_RUNNER" "$remaining" ./gradlew "${GRADLE_ARGS[@]}" --refresh-dependencies --stacktrace --console=plain --no-daemon 2>&1 | tee "$LOG"
  else
    python3 "$TIMEOUT_RUNNER" "$remaining" ./gradlew "${GRADLE_ARGS[@]}" --stacktrace --console=plain --no-daemon 2>&1 | tee "$LOG"
  fi
  status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

set +e
run_once 0
FIRST_STATUS=$?
set -e
if [[ "$FIRST_STATUS" -eq 0 ]]; then exit 0; fi
if [[ "$FIRST_STATUS" -eq 124 ]]; then
  echo "[Director Gradle] Hard timeout reached after ${TOTAL_BUDGET}s." | tee -a "$LOG"
  exit 124
fi

INFRA_PATTERN='zip END header not found|invalid LOC header|error in opening zip file|Could not read workspace metadata|Could not load workspace metadata|cache.*corrupt|corrupt.*cache|Could not GET|Could not HEAD|Read timed out|Connection reset|Connection refused|Temporary failure in name resolution|UnknownHostException|SocketTimeoutException|Premature EOF|unexpected end of stream|TLS handshake|Remote host terminated|No space left on device'

if ! grep -Eiq "$INFRA_PATTERN" "$LOG"; then exit "$FIRST_STATUS"; fi
if (( $(remaining_budget) < 180 )); then
  echo "[Director Gradle] Transient failure detected, but insufficient budget remains for a safe retry." | tee -a "$LOG"
  exit "$FIRST_STATUS"
fi

echo "[Director Gradle] Transient infrastructure/cache failure detected. Performing one bounded recovery retry." | tee -a "$LOG"

GRADLE_HOME="${GRADLE_USER_HOME:-$HOME/.gradle}"
rm -rf \
  "$GRADLE_HOME/caches/modules-2" \
  "$GRADLE_HOME/caches/transforms-3" \
  "$GRADLE_HOME/caches/transforms-4" \
  "$GRADLE_HOME/caches/jars-9" \
  "$ROOT/.gradle" 2>/dev/null || true

sleep 3
run_once 1
