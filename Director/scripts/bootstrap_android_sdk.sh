#!/usr/bin/env bash
set -euo pipefail

PLATFORM="android-36"
BUILD_TOOLS="36.0.0"
NDK="28.2.13676358"
CMAKE="3.22.1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMEOUT_RUNNER="$ROOT/scripts/run_with_timeout.py"
PRIVATE_SDK_ROOT="${DIRECTOR_ANDROID_SDK_ROOT:-$HOME/.director/android-sdk}"

log() { printf '[Director SDK] %s\n' "$*"; }
fail() { printf '[Director SDK] ERROR: %s\n' "$*" >&2; exit 1; }

find_sdk_root() {
  local candidate
  for candidate in "${ANDROID_SDK_ROOT:-}" "${ANDROID_HOME:-}" "/usr/local/share/android-sdk" "$HOME/Library/Android/sdk" "/usr/local/lib/android/sdk" "/opt/android-sdk"; do
    [[ -n "$candidate" && -d "$candidate" ]] || continue
    printf '%s' "$candidate"
    return 0
  done
  return 1
}

find_sdkmanager() {
  local root="$1" candidate
  candidate="$root/cmdline-tools/latest/bin/sdkmanager"
  if [[ -x "$candidate" ]]; then printf '%s' "$candidate"; return 0; fi
  candidate="$(find "$root/cmdline-tools" -maxdepth 4 -name sdkmanager -perm -111 -print -quit 2>/dev/null || true)"
  [[ -n "$candidate" && -x "$candidate" ]] || return 1
  printf '%s' "$candidate"
}

find_ndk_bin() {
  local root="$1" candidate
  for candidate in "$root/ndk/$NDK"/toolchains/llvm/prebuilt/*/bin; do
    [[ -d "$candidate" ]] || continue
    if [[ -x "$candidate/clang" && -x "$candidate/clang++" && -x "$candidate/llvm-ar" ]]; then printf '%s' "$candidate"; return 0; fi
  done
  return 1
}

package_ok() {
  local package="$1" sdk_root="$2" ndk_bin
  case "$package" in
    "platforms;$PLATFORM") [[ -s "$sdk_root/platforms/$PLATFORM/android.jar" ]] ;;
    "build-tools;$BUILD_TOOLS") [[ -x "$sdk_root/build-tools/$BUILD_TOOLS/aapt2" && -x "$sdk_root/build-tools/$BUILD_TOOLS/zipalign" && -f "$sdk_root/build-tools/$BUILD_TOOLS/apksigner" ]] ;;
    "ndk;$NDK")
      [[ -s "$sdk_root/ndk/$NDK/source.properties" && -s "$sdk_root/ndk/$NDK/build/cmake/android.toolchain.cmake" ]] || return 1
      grep -Eq "^Pkg.Revision[[:space:]]*=[[:space:]]*$NDK([[:space:]]*)$" "$sdk_root/ndk/$NDK/source.properties" || return 1
      ndk_bin="$(find_ndk_bin "$sdk_root" || true)"; [[ -n "$ndk_bin" ]]
      ;;
    "cmake;$CMAKE") [[ -x "$sdk_root/cmake/$CMAKE/bin/cmake" ]] ;;
    *) return 1 ;;
  esac
}

missing_packages() {
  local sdk_root="$1" package
  for package in "platforms;$PLATFORM" "build-tools;$BUILD_TOOLS" "ndk;$NDK" "cmake;$CMAKE"; do
    package_ok "$package" "$sdk_root" || printf '%s\n' "$package"
  done
}

collect_missing() {
  local sdk_root="$1" package
  MISSING=()
  while IFS= read -r package; do [[ -n "$package" ]] && MISSING+=("$package"); done < <(missing_packages "$sdk_root")
}

install_missing_once() {
  local sdk_root="$1" sdkmanager="$2"; shift 2
  local packages=("$@") license_status
  [[ "${#packages[@]}" -gt 0 ]] || return 0
  log "Missing packages: ${packages[*]}"
  log "Accepting licenses with a 90 second hard timeout"
  set +e
  yes | python3 "$TIMEOUT_RUNNER" 90 "$sdkmanager" --sdk_root="$sdk_root" --licenses >/dev/null 2>&1
  license_status=${PIPESTATUS[1]}
  set -e
  if [[ "$license_status" -ne 0 && "$license_status" -ne 124 ]]; then log "License command returned $license_status; continuing to bounded install"; fi
  log "Installing only missing packages once (20 minute hard timeout)"
  python3 "$TIMEOUT_RUNNER" 1200 "$sdkmanager" --sdk_root="$sdk_root" --channel=0 --install "${packages[@]}"
}

functional_preflight() {
  local sdk_root="$1" ndk_bin
  ndk_bin="$(find_ndk_bin "$sdk_root")"
  log "Running functional toolchain preflight"
  python3 "$TIMEOUT_RUNNER" 30 "$sdk_root/build-tools/$BUILD_TOOLS/aapt2" version
  python3 "$TIMEOUT_RUNNER" 30 "$sdk_root/cmake/$CMAKE/bin/cmake" --version | head -n 1
  python3 "$TIMEOUT_RUNNER" 30 "$ndk_bin/clang" --version | head -n 1
  python3 "$TIMEOUT_RUNNER" 30 "$ndk_bin/clang++" --version | head -n 1
  python3 "$TIMEOUT_RUNNER" 30 "$ndk_bin/llvm-ar" --version | head -n 1
  log "Android SDK/NDK/CMake preflight passed"
}

[[ -f "$TIMEOUT_RUNNER" ]] || fail "Timeout helper missing: $TIMEOUT_RUNNER"
SDK_ROOT="$(find_sdk_root || true)"
[[ -n "$SDK_ROOT" ]] || fail "No Android SDK root found. Expected ANDROID_SDK_ROOT/ANDROID_HOME or Codemagic's preinstalled SDK."
SDKMANAGER="$(find_sdkmanager "$SDK_ROOT" || true)"
[[ -n "$SDKMANAGER" ]] || fail "sdkmanager not found under $SDK_ROOT"

log "Using discovered Android SDK: $SDK_ROOT"
log "sdkmanager: $SDKMANAGER"
python3 "$TIMEOUT_RUNNER" 30 "$SDKMANAGER" --version

collect_missing "$SDK_ROOT"
if [[ "${#MISSING[@]}" -gt 0 ]]; then
  if [[ -w "$SDK_ROOT" ]]; then
    install_missing_once "$SDK_ROOT" "$SDKMANAGER" "${MISSING[@]}"
  else
    log "SDK is not writable; provisioning private SDK only because required packages are missing"
    SDK_ROOT="$PRIVATE_SDK_ROOT"
    mkdir -p "$SDK_ROOT"
    install_missing_once "$SDK_ROOT" "$SDKMANAGER" "platforms;$PLATFORM" "build-tools;$BUILD_TOOLS" "ndk;$NDK" "cmake;$CMAKE"
  fi
fi

collect_missing "$SDK_ROOT"
if [[ "${#MISSING[@]}" -gt 0 ]]; then
  printf '[Director SDK] ERROR: Required packages still missing after bounded provisioning:\n' >&2
  printf '  - %s\n' "${MISSING[@]}" >&2
  exit 4
fi

functional_preflight "$SDK_ROOT"
STATE_FILE="${CM_BUILD_DIR:-${GITHUB_WORKSPACE:-$(cd "$ROOT/.." && pwd)}}/.director-android-sdk-root"
printf '%s\n' "$SDK_ROOT" > "$STATE_FILE"
log "Verified Android SDK ready: $SDK_ROOT"
log "State file: $STATE_FILE"
