#!/usr/bin/env bash
set -euo pipefail

COMMIT="56381e407c0ccfb3a6f71e668a27a901001d22ce"
SHA256="7fd19c03e7d7bb02c07e64c47e0539aeabe54f366a59be822f550b5f012a8751"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/llama.cpp"
MARKER="$DEST/.director-llama-commit"
PRUNER="$ROOT/scripts/prune_llama_for_director.py"
PRUNE_MARKER="$DEST/.director-qwen3vl-pruned-v3"

cache_ok() {
  [[ -f "$MARKER" && "$(cat "$MARKER" 2>/dev/null || true)" == "$COMMIT" ]] || return 1
  local required
  for required in CMakeLists.txt include/llama.h ggml/include/ggml.h tools/mtmd/mtmd.h tools/mtmd/mtmd-helper.h tools/mtmd/models/qwen2vl.cpp tools/mtmd/models/qwen3vl.cpp; do
    [[ -s "$DEST/$required" ]] || return 1
  done
}

activate_director_profile() {
  python3 "$PRUNER" "$DEST"
  [[ -s "$PRUNE_MARKER" ]] || { echo "Director llama.cpp pruning v3 marker missing" >&2; return 1; }
}

if cache_ok; then
  activate_director_profile
  echo "Using verified cached llama.cpp $COMMIT (Director Qwen3-VL profile v3)"
  exit 0
fi
if [[ -e "$DEST" ]]; then echo "Discarding stale/incomplete llama.cpp cache"; rm -rf "$DEST"; fi

archive="$(mktemp -t llama-cpp.XXXXXX.tar.gz)"
staging="$(mktemp -d -t llama-cpp.XXXXXX)"
trap 'rm -rf "$archive" "$staging"' EXIT

curl --fail --location \
  --retry 4 --retry-all-errors --retry-delay 2 \
  --connect-timeout 20 --max-time 300 \
  --speed-time 30 --speed-limit 1024 \
  "https://github.com/ggml-org/llama.cpp/archive/${COMMIT}.tar.gz" -o "$archive"

if command -v shasum >/dev/null 2>&1; then actual="$(shasum -a 256 "$archive" | awk '{print $1}')"; else actual="$(sha256sum "$archive" | awk '{print $1}')"; fi
[[ "$actual" == "$SHA256" ]] || { echo "llama.cpp checksum mismatch" >&2; exit 1; }

tar -tzf "$archive" >/dev/null
tar -xzf "$archive" --strip-components=1 -C "$staging"
for required in CMakeLists.txt include/llama.h ggml/include/ggml.h tools/mtmd/mtmd.h tools/mtmd/mtmd-helper.h tools/mtmd/models/qwen2vl.cpp tools/mtmd/models/qwen3vl.cpp; do
  [[ -s "$staging/$required" ]] || { echo "llama.cpp archive missing required file: $required" >&2; exit 1; }
done
mkdir -p "$(dirname "$DEST")"
mv "$staging" "$DEST"
printf '%s\n' "$COMMIT" > "$MARKER"
activate_director_profile
cache_ok || { echo "Activated llama.cpp cache failed integrity verification" >&2; rm -rf "$DEST"; exit 1; }
echo "Pinned llama.cpp $COMMIT ready with Director Qwen3-VL build profile v3"
