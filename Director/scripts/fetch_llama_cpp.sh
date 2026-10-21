#!/usr/bin/env bash
set -euo pipefail
COMMIT="56381e407c0ccfb3a6f71e668a27a901001d22ce"
SHA256="7fd19c03e7d7bb02c07e64c47e0539aeabe54f366a59be822f550b5f012a8751"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/llama.cpp"
if [[ -f "$DEST/CMakeLists.txt" ]]; then exit 0; fi
archive="$(mktemp -t llama-cpp.XXXXXX.tar.gz)"
staging="$(mktemp -d -t llama-cpp.XXXXXX)"
trap 'rm -rf "$archive" "$staging"' EXIT
curl --fail --location --retry 3 "https://github.com/ggml-org/llama.cpp/archive/${COMMIT}.tar.gz" -o "$archive"
actual="$(shasum -a 256 "$archive" | awk '{print $1}')"
[[ "$actual" == "$SHA256" ]] || { echo "llama.cpp checksum mismatch" >&2; exit 1; }
tar -xzf "$archive" --strip-components=1 -C "$staging"
mkdir -p "$(dirname "$DEST")"
mv "$staging" "$DEST"
