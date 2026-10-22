#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("Usage: prune_llama_for_director.py <llama.cpp-root>")

root = Path(sys.argv[1]).resolve()
cmake_path = root / "tools/mtmd/CMakeLists.txt"
clip_path = root / "tools/mtmd/clip.cpp"
marker = root / ".director-qwen3vl-pruned-v1"

for required in (cmake_path, clip_path, root / "tools/mtmd/models/qwen2vl.cpp", root / "tools/mtmd/models/qwen3vl.cpp"):
    if not required.is_file():
        raise SystemExit(f"Missing pinned llama.cpp source required by Director: {required}")

if marker.is_file():
    cmake = cmake_path.read_text(encoding="utf-8")
    if "models/qwen2vl.cpp" in cmake and "models/qwen3vl.cpp" in cmake and "models/gemma4v.cpp" not in cmake:
        print("Director Qwen3-VL mtmd pruning already applied")
        raise SystemExit(0)

cmake = cmake_path.read_text(encoding="utf-8")
start = cmake.find("add_library(mtmd\n")
end_token = "\n\nset_target_properties(mtmd PROPERTIES"
end = cmake.find(end_token, start)
if start < 0 or end < 0:
    raise SystemExit("Pinned mtmd CMake layout changed; refusing unsafe pruning")

lean_mtmd = '''add_library(mtmd
            mtmd.cpp
            mtmd-audio.cpp
            mtmd-image.cpp
            mtmd.h
            mtmd-internal.h
            mtmd-helper.cpp
            mtmd-helper-gen.cpp
            mtmd-helper-common.h
            mtmd-helper.h
            clip.cpp
            clip.h
            clip-impl.h
            clip-model.h
            clip-graph.h
            models/models.h
            models/qwen2vl.cpp
            models/qwen3vl.cpp
            )'''
cmake = cmake[:start] + lean_mtmd + cmake[end:]
cmake_path.write_text(cmake, encoding="utf-8")

clip = clip_path.read_text(encoding="utf-8")
func_pos = clip.find("static std::unique_ptr<clip_graph> clip_get_graph_builder")
if func_pos < 0:
    raise SystemExit("Pinned clip graph builder not found; refusing unsafe pruning")
switch_pos = clip.find("switch (ctx->proj_type()) {", func_pos)
if switch_pos < 0:
    raise SystemExit("Pinned clip graph switch not found; refusing unsafe pruning")
brace_pos = clip.find("{", switch_pos)
depth = 0
switch_end = -1
for i in range(brace_pos, len(clip)):
    ch = clip[i]
    if ch == "{":
        depth += 1
    elif ch == "}":
        depth -= 1
        if depth == 0:
            switch_end = i + 1
            break
if switch_end < 0:
    raise SystemExit("Could not locate end of pinned clip graph switch")

lean_switch = '''switch (ctx->proj_type()) {
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
            {
                builder = std::make_unique<clip_graph_qwen2vl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_QWEN3VL:
            {
                builder = std::make_unique<clip_graph_qwen3vl>(ctx, img);
            } break;
        default:
            LOG_ERR("%s: projector type is not enabled in the Director Qwen3-VL build\\n", __func__);
            return nullptr;
    }'''
clip = clip[:switch_pos] + lean_switch + clip[switch_end:]
clip_path.write_text(clip, encoding="utf-8")

verification = cmake_path.read_text(encoding="utf-8")
for required in ("models/qwen2vl.cpp", "models/qwen3vl.cpp"):
    if required not in verification:
        raise SystemExit(f"Director mtmd pruning lost required source: {required}")
for forbidden in ("models/gemma4v.cpp", "models/llava.cpp", "models/deepseekocr.cpp", "models/whisper-enc.cpp", "models/qwen3tts-gen.cpp"):
    if forbidden in verification:
        raise SystemExit(f"Director mtmd pruning left forbidden model source: {forbidden}")

marker.write_text("qwen3vl-only-v1\n", encoding="utf-8")
print("Pruned llama.cpp mtmd build to Qwen2/2.5-VL base + Qwen3-VL")
