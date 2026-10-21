#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log() { printf '[Director Preflight] %s\n' "$*"; }
fail() { printf '[Director Preflight] ERROR: %s\n' "$*" >&2; exit 1; }

for cmd in bash java python3 curl unzip tar git awk sed grep find; do
  command -v "$cmd" >/dev/null 2>&1 || fail "Required host command missing: $cmd"
done

JAVA_MAJOR="$(java -version 2>&1 | awk -F'[\".]' '/version/ {print $2; exit}')"
[[ "$JAVA_MAJOR" == "17" ]] || fail "Director requires JDK 17; detected Java major ${JAVA_MAJOR:-unknown}"
log "JDK 17 verified"

STATE_FILE="${CM_BUILD_DIR:-${GITHUB_WORKSPACE:-$(cd "$ROOT/.." && pwd)}}/.director-android-sdk-root"
[[ -s "$STATE_FILE" ]] || fail "Android SDK state file missing: $STATE_FILE"
SDK_ROOT="$(cat "$STATE_FILE")"
[[ -d "$SDK_ROOT" ]] || fail "Resolved Android SDK root does not exist: $SDK_ROOT"

[[ -r "$SDK_ROOT/platforms/android-36/android.jar" ]] || fail "android-36 platform is incomplete"
[[ -x "$SDK_ROOT/build-tools/36.0.0/aapt2" ]] || fail "Build Tools 36.0.0 aapt2 missing"
[[ -x "$SDK_ROOT/cmake/3.22.1/bin/cmake" ]] || fail "CMake 3.22.1 missing"
[[ -r "$SDK_ROOT/ndk/28.2.13676358/build/cmake/android.toolchain.cmake" ]] || fail "NDK Android CMake toolchain missing"

NDK_BIN=""
for candidate in "$SDK_ROOT"/ndk/28.2.13676358/toolchains/llvm/prebuilt/*/bin; do
  [[ -d "$candidate" ]] || continue
  if [[ -x "$candidate/clang" && -x "$candidate/clang++" ]]; then NDK_BIN="$candidate"; break; fi
done
[[ -n "$NDK_BIN" ]] || fail "NDK LLVM host toolchain not found"
"$NDK_BIN/clang" --version >/dev/null || fail "NDK clang cannot execute"
"$SDK_ROOT/cmake/3.22.1/bin/cmake" --version >/dev/null || fail "CMake cannot execute"
"$SDK_ROOT/build-tools/36.0.0/aapt2" version >/dev/null || fail "aapt2 cannot execute"
log "Android SDK/NDK/CMake executables verified"

# Director downloads a ~1.6 GB VLM later and native/Gradle builds can consume several GB.
# CI should fail immediately with a useful diagnostic instead of corrupting archives mid-build.
CHECK_PATH="${CM_BUILD_DIR:-$ROOT}"
AVAILABLE_KB="$(df -Pk "$CHECK_PATH" | awk 'NR==2 {print $4}')"
MIN_KB=$((10 * 1024 * 1024))
if [[ -n "$AVAILABLE_KB" && "$AVAILABLE_KB" -lt "$MIN_KB" ]]; then
  fail "Less than 10 GiB free disk space (${AVAILABLE_KB} KiB available)"
fi
log "Disk-space preflight passed"

chmod +x gradlew scripts/*.sh scripts/*.py
./gradlew --version >/dev/null
log "Verified Gradle 8.13 runtime"

python3 scripts/compile_skills.py
python3 -m json.tool app/src/main/assets/skills.json >/dev/null
python3 scripts/validate_product_contract.py
log "Skills IR and product contract verified"

printf 'sdk.dir=%s\n' "$SDK_ROOT" > local.properties
log "local.properties written for $SDK_ROOT"
log "Director CI preflight passed"
