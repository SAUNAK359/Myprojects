#!/usr/bin/env bash
set -euo pipefail

VERSION="v1.24.0"
EXPECTED_MD5="2f02ebe29573d45813e671eb304f2a00"
URL="https://github.com/ARM-software/kleidiai/releases/download/${VERSION}/kleidiai-${VERSION}-src.tar.gz"
DEST="${DIRECTOR_KLEIDIAI_SOURCE_DIR:-$HOME/.director/kleidiai/$VERSION}"
MARKER="$DEST/.director-kleidiai-version"

cache_ok() {
  [[ -s "$DEST/CMakeLists.txt" && -f "$MARKER" && "$(cat "$MARKER" 2>/dev/null || true)" == "$VERSION" ]]
}

if cache_ok; then
  echo "Using verified cached KleidiAI $VERSION"
  exit 0
fi

rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
archive="$(mktemp -t kleidiai.XXXXXX.tar.gz)"
staging="$(mktemp -d -t kleidiai.XXXXXX)"
trap 'rm -rf "$archive" "$staging"' EXIT

curl --fail --location \
  --retry 4 --retry-all-errors --retry-delay 2 \
  --connect-timeout 20 --max-time 300 \
  --speed-time 30 --speed-limit 1024 \
  "$URL" -o "$archive"

if command -v md5 >/dev/null 2>&1; then
  actual="$(md5 -q "$archive")"
elif command -v md5sum >/dev/null 2>&1; then
  actual="$(md5sum "$archive" | awk '{print $1}')"
else
  echo "No MD5 tool available to verify KleidiAI archive" >&2
  exit 1
fi
[[ "$actual" == "$EXPECTED_MD5" ]] || { echo "KleidiAI checksum mismatch: expected=$EXPECTED_MD5 actual=$actual" >&2; exit 1; }

tar -tzf "$archive" >/dev/null
mkdir -p "$staging/src"
tar -xzf "$archive" --strip-components=1 -C "$staging/src"
[[ -s "$staging/src/CMakeLists.txt" ]] || { echo "KleidiAI archive missing CMakeLists.txt" >&2; exit 1; }

mv "$staging/src" "$DEST"
printf '%s\n' "$VERSION" > "$MARKER"
cache_ok || { echo "Activated KleidiAI cache failed integrity verification" >&2; rm -rf "$DEST"; exit 1; }
echo "Pinned KleidiAI $VERSION ready and verified at $DEST"
