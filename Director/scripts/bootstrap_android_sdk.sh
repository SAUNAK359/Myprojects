#!/usr/bin/env bash
set -euo pipefail

PLATFORM="android-36"
BUILD_TOOLS="36.0.0"
NDK="28.2.13676358"
CMAKE="3.22.1"
CMDLINE_TOOLS_REV="15859902"
MAC_ARM_SHA256="835b62a26162b229b441d1f6d4680383815a270809eb33522c0d480fa5002c4e"
MAC_X64_SHA256="c5a6378ab5cf7e0d5701921405115befff13e9ff7417fb588389338f8bd050f3"
LINUX_SHA256="4e4c464f145a7512b57d088ac6c278c03c9eea610886b35a5e0804e74eedf583"
PRIVATE_SDK_ROOT="${DIRECTOR_ANDROID_SDK_ROOT:-$HOME/.director/android-sdk}"

SDK_ROOT=""
SDKMANAGER=""
DISCOVERED_SDK_ROOT=""

log() { printf '[Director SDK] %s\n' "$*"; }

find_sdkmanager() {
  local roots=()
  [[ -n "${ANDROID_SDK_ROOT:-}" ]] && roots+=("$ANDROID_SDK_ROOT")
  [[ -n "${ANDROID_HOME:-}" ]] && roots+=("$ANDROID_HOME")
  roots+=("$HOME/Library/Android/sdk" "/usr/local/share/android-sdk" "/usr/local/lib/android/sdk" "/opt/android-sdk")

  local root candidate
  for root in "${roots[@]}"; do
    [[ -d "$root" ]] || continue
    candidate="$root/cmdline-tools/latest/bin/sdkmanager"
    if [[ -x "$candidate" ]]; then
      DISCOVERED_SDK_ROOT="$root"
      SDKMANAGER="$candidate"
      return 0
    fi
    candidate="$(find "$root/cmdline-tools" -maxdepth 4 -type f -name sdkmanager -perm -111 2>/dev/null | head -n 1 || true)"
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      DISCOVERED_SDK_ROOT="$root"
      SDKMANAGER="$candidate"
      return 0
    fi
  done
  return 1
}

install_commandline_tools() {
  local os arch package sha tmp zip extracted
  os="$(uname -s)"
  arch="$(uname -m)"
  DISCOVERED_SDK_ROOT="$PRIVATE_SDK_ROOT"
  mkdir -p "$DISCOVERED_SDK_ROOT/cmdline-tools"

  case "$os/$arch" in
    Darwin/arm64|Darwin/aarch64)
      package="commandlinetools-mac_arm64-${CMDLINE_TOOLS_REV}_latest.zip"
      sha="$MAC_ARM_SHA256"
      ;;
    Darwin/x86_64)
      package="commandlinetools-mac_x86_64-${CMDLINE_TOOLS_REV}_latest.zip"
      sha="$MAC_X64_SHA256"
      ;;
    Linux/x86_64|Linux/amd64)
      package="commandlinetools-linux-${CMDLINE_TOOLS_REV}_latest.zip"
      sha="$LINUX_SHA256"
      ;;
    *)
      echo "Unsupported CI host for Android command-line tools: $os/$arch" >&2
      exit 2
      ;;
  esac

  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN
  zip="$tmp/tools.zip"
  log "sdkmanager not found; bootstrapping $package"
  curl --fail --location --retry 6 --retry-all-errors --retry-delay 3 --connect-timeout 30 \
    "https://dl.google.com/android/repository/$package" -o "$zip"

  if command -v shasum >/dev/null 2>&1; then
    echo "$sha  $zip" | shasum -a 256 -c -
  else
    echo "$sha  $zip" | sha256sum -c -
  fi

  extracted="$tmp/extracted"
  mkdir -p "$extracted"
  unzip -tq "$zip" >/dev/null
  unzip -q "$zip" -d "$extracted"
  rm -rf "$DISCOVERED_SDK_ROOT/cmdline-tools/latest"
  mv "$extracted/cmdline-tools" "$DISCOVERED_SDK_ROOT/cmdline-tools/latest"
  SDKMANAGER="$DISCOVERED_SDK_ROOT/cmdline-tools/latest/bin/sdkmanager"
  trap - RETURN
  rm -rf "$tmp"
}

package_target() {
  case "$1" in
    "platforms;$PLATFORM") printf '%s/platforms/%s' "$2" "$PLATFORM" ;;
    "build-tools;$BUILD_TOOLS") printf '%s/build-tools/%s' "$2" "$BUILD_TOOLS" ;;
    "ndk;$NDK") printf '%s/ndk/%s' "$2" "$NDK" ;;
    "cmake;$CMAKE") printf '%s/cmake/%s' "$2" "$CMAKE" ;;
    *) return 2 ;;
  esac
}

package_ok() {
  local package="$1" root="$2" target
  target="$(package_target "$package" "$root")"
  case "$package" in
    "platforms;$PLATFORM")
      [[ -s "$target/android.jar" ]]
      ;;
    "build-tools;$BUILD_TOOLS")
      [[ -x "$target/aapt2" && -x "$target/zipalign" && -f "$target/apksigner" ]]
      ;;
    "ndk;$NDK")
      [[ -s "$target/source.properties" ]] && find "$target/toolchains/llvm/prebuilt" -type f -name clang++ -perm -111 -print -quit 2>/dev/null | grep -q .
      ;;
    "cmake;$CMAKE")
      [[ -x "$target/bin/cmake" ]]
      ;;
    *)
      return 1
      ;;
  esac
}

purge_failed_package() {
  local package="$1" root="$2" target
  target="$(package_target "$package" "$root")"
  log "Purging partial state for $package from $root"
  rm -rf "$target" 2>/dev/null || true
  rm -rf "$root/.temp" "$root/temp" 2>/dev/null || true
  rm -rf "$HOME/.android/cache" 2>/dev/null || true
}

accept_licenses() {
  local root="$1" tmp="$HOME/.director/sdk-tmp/licenses"
  mkdir -p "$root" "$tmp"
  yes | env TMPDIR="$tmp" "$SDKMANAGER" --sdk_root="$root" --licenses >/dev/null 2>&1 || true
}

install_one() {
  local package="$1" root="$2" max_attempts="$3"
  local attempt status tmp

  if package_ok "$package" "$root"; then
    log "$package already verified in $root"
    return 0
  fi

  for ((attempt=1; attempt<=max_attempts; attempt++)); do
    purge_failed_package "$package" "$root"
    tmp="$HOME/.director/sdk-tmp/install-${attempt}"
    rm -rf "$tmp" && mkdir -p "$tmp" "$root"
    log "Installing $package into $root (attempt $attempt/$max_attempts)"

    set +e
    env TMPDIR="$tmp" "$SDKMANAGER" --sdk_root="$root" --channel=0 --install "$package"
    status=$?
    set -e

    rm -rf "$tmp"
    if [[ "$status" -eq 0 ]] && package_ok "$package" "$root"; then
      log "$package installed and verified"
      return 0
    fi

    log "$package install attempt $attempt failed (status=$status or verification failed)"
    df -h "$root" 2>/dev/null || true
    sleep $((attempt * 3))
  done
  return 1
}

ensure_toolchain() {
  local root="$1" attempts="$2" package
  mkdir -p "$root"
  accept_licenses "$root"
  for package in \
    "platforms;$PLATFORM" \
    "build-tools;$BUILD_TOOLS" \
    "ndk;$NDK" \
    "cmake;$CMAKE"; do
    install_one "$package" "$root" "$attempts" || return 1
  done
  return 0
}

toolchain_ok() {
  local root="$1" package
  for package in \
    "platforms;$PLATFORM" \
    "build-tools;$BUILD_TOOLS" \
    "ndk;$NDK" \
    "cmake;$CMAKE"; do
    package_ok "$package" "$root" || return 1
  done
  return 0
}

if ! find_sdkmanager; then
  install_commandline_tools
fi
[[ -x "$SDKMANAGER" ]] || { echo "sdkmanager bootstrap failed" >&2; exit 3; }

log "sdkmanager: $SDKMANAGER"
"$SDKMANAGER" --version

# Prefer the preinstalled SDK only when it is already complete. If it is incomplete,
# first give it two self-healing attempts; then abandon it and use a clean private SDK.
SDK_ROOT="$DISCOVERED_SDK_ROOT"
if toolchain_ok "$SDK_ROOT"; then
  log "Preinstalled Android SDK is complete and verified: $SDK_ROOT"
else
  log "Preinstalled Android SDK is incomplete; repairing with isolated temp/cache state"
  if [[ -w "$SDK_ROOT" ]] && ensure_toolchain "$SDK_ROOT" 2; then
    log "Shared Android SDK repaired successfully"
  else
    log "Shared Android SDK repair failed or root is not writable. Switching to private clean-room SDK: $PRIVATE_SDK_ROOT"
    SDK_ROOT="$PRIVATE_SDK_ROOT"
    mkdir -p "$SDK_ROOT"
    if ! ensure_toolchain "$SDK_ROOT" 4; then
      echo "Unable to provision a verified Android toolchain after shared and private SDK recovery." >&2
      echo "sdkmanager: $SDKMANAGER" >&2
      echo "private root: $SDK_ROOT" >&2
      df -h "$SDK_ROOT" 2>/dev/null || true
      exit 4
    fi
  fi
fi

# Final hard verification. A directory alone is not sufficient: verify the binaries/files
# Gradle/CMake will actually use.
toolchain_ok "$SDK_ROOT" || { echo "Android toolchain verification failed for $SDK_ROOT" >&2; exit 5; }

export ANDROID_HOME="$SDK_ROOT"
export ANDROID_SDK_ROOT="$SDK_ROOT"

log "Using verified Android SDK root: $SDK_ROOT"
log "Build Tools: $BUILD_TOOLS"
log "NDK: $NDK"
log "CMake: $CMAKE"
"$SDK_ROOT/build-tools/$BUILD_TOOLS/aapt2" version || true
"$SDK_ROOT/cmake/$CMAKE/bin/cmake" --version | head -n 1

STATE_FILE="${CM_BUILD_DIR:-${GITHUB_WORKSPACE:-$(pwd)}}/.director-android-sdk-root"
printf '%s\n' "$SDK_ROOT" > "$STATE_FILE"
log "Android toolchain ready. State file: $STATE_FILE"
