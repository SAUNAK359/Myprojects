#!/usr/bin/env bash
set -euo pipefail

# Director Android SDK bootstrap.
# Works on Codemagic macOS Apple Silicon/Intel and Linux CI.
# Never assumes sdkmanager is already on PATH.

PLATFORM="android-36"
BUILD_TOOLS="36.0.0"
NDK="28.2.13676358"
CMAKE="3.22.1"
CMDLINE_TOOLS_REV="15859902"
MAC_ARM_SHA256="835b62a26162b229b441d1f6d4680383815a270809eb33522c0d480fa5002c4e"
MAC_X64_SHA256="c5a6378ab5cf7e0d5701921405115befff13e9ff7417fb588389338f8bd050f3"
LINUX_SHA256="4e4c464f145a7512b57d088ac6c278c03c9eea610886b35a5e0804e74eedf583"

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
      printf '%s\n%s\n' "$root" "$candidate"
      return 0
    fi
    candidate="$(find "$root/cmdline-tools" -maxdepth 4 -type f -name sdkmanager -perm -111 2>/dev/null | head -n 1 || true)"
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n%s\n' "$root" "$candidate"
      return 0
    fi
  done
  return 1
}

install_commandline_tools() {
  local os arch package sha sdk_root tmp zip extracted
  os="$(uname -s)"
  arch="$(uname -m)"
  sdk_root="${DIRECTOR_ANDROID_SDK_ROOT:-$HOME/.director/android-sdk}"
  mkdir -p "$sdk_root/cmdline-tools"

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
  echo "Android sdkmanager not found; bootstrapping $package"
  curl --fail --location --retry 4 --retry-all-errors --connect-timeout 30 \
    "https://dl.google.com/android/repository/$package" -o "$zip"

  if command -v shasum >/dev/null 2>&1; then
    echo "$sha  $zip" | shasum -a 256 -c -
  else
    echo "$sha  $zip" | sha256sum -c -
  fi

  extracted="$tmp/extracted"
  mkdir -p "$extracted"
  unzip -q "$zip" -d "$extracted"
  rm -rf "$sdk_root/cmdline-tools/latest"
  mv "$extracted/cmdline-tools" "$sdk_root/cmdline-tools/latest"
  printf '%s\n%s\n' "$sdk_root" "$sdk_root/cmdline-tools/latest/bin/sdkmanager"
}

resolved="$(find_sdkmanager || install_commandline_tools)"
SDK_ROOT="$(printf '%s\n' "$resolved" | sed -n '1p')"
SDKMANAGER="$(printf '%s\n' "$resolved" | sed -n '2p')"

[[ -x "$SDKMANAGER" ]] || { echo "sdkmanager bootstrap failed" >&2; exit 3; }

echo "Using Android SDK root: $SDK_ROOT"
echo "Using sdkmanager: $SDKMANAGER"
"$SDKMANAGER" --version

yes | "$SDKMANAGER" --sdk_root="$SDK_ROOT" --licenses >/dev/null 2>&1 || true
"$SDKMANAGER" --sdk_root="$SDK_ROOT" --install \
  "platforms;$PLATFORM" \
  "build-tools;$BUILD_TOOLS" \
  "ndk;$NDK" \
  "cmake;$CMAKE"

for path in \
  "$SDK_ROOT/platforms/$PLATFORM" \
  "$SDK_ROOT/build-tools/$BUILD_TOOLS" \
  "$SDK_ROOT/ndk/$NDK" \
  "$SDK_ROOT/cmake/$CMAKE"; do
  [[ -d "$path" ]] || { echo "Required Android component missing after install: $path" >&2; exit 4; }
done

# Persist for later CI steps without relying on shell PATH inheritance.
STATE_FILE="${CM_BUILD_DIR:-${GITHUB_WORKSPACE:-$(pwd)}}/.director-android-sdk-root"
printf '%s\n' "$SDK_ROOT" > "$STATE_FILE"
echo "Android toolchain ready. State file: $STATE_FILE"
