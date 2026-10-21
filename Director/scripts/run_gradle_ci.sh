#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: run_gradle_ci.sh <log-file> <gradle args...>" >&2
  exit 2
fi

LOG="$1"
shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p "$(dirname "$LOG")"

GRADLE_ARGS=("$@")

run_once() {
  local refresh="$1"
  set +e
  if [[ "$refresh" == "1" ]]; then
    ./gradlew "${GRADLE_ARGS[@]}" --refresh-dependencies --stacktrace --console=plain --no-daemon 2>&1 | tee "$LOG"
  else
    ./gradlew "${GRADLE_ARGS[@]}" --stacktrace --console=plain --no-daemon 2>&1 | tee "$LOG"
  fi
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

if run_once 0; then
  exit 0
fi

# Retry only known infrastructure/cache/download failure classes. Never retry ordinary
# Kotlin/Java/C++ compiler errors, lint errors, test failures, or application exceptions.
INFRA_PATTERN='zip END header not found|invalid LOC header|error in opening zip file|Could not read workspace metadata|Could not load workspace metadata|cache.*corrupt|corrupt.*cache|Could not resolve all files|Could not GET|Could not HEAD|Read timed out|Connection reset|Connection refused|Temporary failure in name resolution|UnknownHostException|SocketTimeoutException|Premature EOF|unexpected end of stream|TLS handshake|Remote host terminated|No space left on device'

if ! grep -Eiq "$INFRA_PATTERN" "$LOG"; then
  exit 1
fi

echo "[Director Gradle] Recognized transient/corrupt infrastructure failure. Performing one isolated recovery retry." | tee -a "$LOG"

# Preserve the verified Director Gradle distribution cache, but discard Gradle's mutable
# dependency/transforms metadata which is the usual source of CI cache corruption.
GRADLE_HOME="${GRADLE_USER_HOME:-$HOME/.gradle}"
rm -rf \
  "$GRADLE_HOME/caches/modules-2" \
  "$GRADLE_HOME/caches/transforms-3" \
  "$GRADLE_HOME/caches/transforms-4" \
  "$GRADLE_HOME/caches/jars-9" \
  "$GRADLE_HOME/caches/build-cache-1" \
  "$ROOT/.gradle" \
  "$ROOT/app/.cxx" 2>/dev/null || true

sleep 3
run_once 1
