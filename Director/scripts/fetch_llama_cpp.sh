#!/usr/bin/env bash
set -euo pipefail

COMMIT="56381e407c0ccfb3a6f71e668a27a901001d22ce"
SHA256="7fd19c03e7d7bb02c07e64c47e0539aeabe54f366a59be822f550b5f012a8751"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/llama.cpp"
MARKER="$DEST/.director-llama-commit"

if [[ -f "$DEST/CMakeLists.txt" && -f "$MARKER" && "$(cat "$MARKER")" == "$COMMIT" ]]; then
  echo "Using cached llama.cpp $COMMIT"
  exit 0
fi

if [[ -e "$DEST" ]]; then
  echo "Discarding stale/incomplete llama.cpp cache"
  rm -rf "$DEST"
fi

archive="$(mktemp -t llama-cpp.XXXXXX.tar.gz)"
staging="$(mktemp -d -t llama-cpp.XXXXXX)"
trap 'rm -rf "$archive" "$staging"' EXIT

curl --fail --location --retry 4 --retry-all-errors --connect-timeout 30 \
  "https://github.com/ggml-org/llama.cpp/archive/${COMMIT}.tar.gz" -o "$archive"

if command -v shasum >/dev/null 2>&1; then
  actual="$(shasum -a 256 "$archive" | awk '{print $1}')"
else
  actual="$(sha256sum "$archive" | awk '{print $1}')"
fi
[[ "$actual" == "$SHA256" ]] || { echo "llama.cpp checksum mismatch" >&2; exit 1; }

tar -xzf "$archive" --strip-components=1 -C "$staging"
test -f "$staging/CMakeLists.txt"
mkdir -p "$(dirname "$DEST")"
mv "$staging" "$DEST"
printf '%s\n' "$COMMIT" > "$MARKER"
echo "Pinned llama.cpp $COMMIT ready"
