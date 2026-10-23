#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("Usage: prune_llama_for_director.py <llama.cpp-root>")

root = Path(sys.argv[1]).resolve()
cmake_path = root / "tools/mtmd/CMakeLists.txt"
clip_path = root / "tools/mtmd/clip.cpp"
marker = root / ".director-qwen3vl-pruned-v3"

for required in (cmake_path, clip_path, root / "tools/mtmd/models/qwen2vl.cpp", root / "tools/mtmd/models/qwen3vl.cpp"):
    if not required.is_file():
        raise SystemExit(f"Missing pinned llama.cpp source required by Director: {required}")


def mtmd_target_block(cmake: str) -> str:
    start = cmake.find("add_library(mtmd\n")
    end = cmake.find("\n\nset_target_properties(mtmd PROPERTIES", start)
    if start < 0 or end < 0:
        raise SystemExit("Pinned mtmd CMake layout changed; refusing unsafe pruning")
    return cmake[start:end]


def profile_ok() -> bool:
    cmake = cmake_path.read_text(encoding="utf-8")
    clip = clip_path.read_text(encoding="utf-8")
    target = mtmd_target_block(cmake)
    return (
        "models/qwen2vl.cpp" in target
        and "models/qwen3vl.cpp" in target
        and "mtmd-helper.cpp" in target
        and "models/gemma4v.cpp" not in target
        and "models/qwen3tts-gen.cpp" not in target
        and "models/pockettts-gen.cpp" not in target
        and "mtmd-helper-gen.cpp" not in target
        and "list_c2w_state_slots(hparams, model)" not in clip
        and "list_pockettts_state_slots(hparams, model)" not in clip
    )


if marker.is_file() and profile_ok():
    print("Director Qwen3-VL mtmd pruning v3 already applied")
    raise SystemExit(0)

cmake = cmake_path.read_text(encoding="utf-8")
start = cmake.find("add_library(mtmd\n")
end_token = "\n\nset_target_properties(mtmd PROPERTIES"
end = cmake.find(end_token, start)
if start < 0 or end < 0:
    raise SystemExit("Pinned mtmd CMake layout changed; refusing unsafe pruning")

# Director consumes only still-image Qwen3-VL inference. Keep the generic MTMD
# image/runtime core plus Qwen2/2.5-VL (Qwen3-VL inherits shared input helpers)
# and Qwen3-VL. Audio generation and unrelated projector implementations are not
# reachable from Director and are intentionally excluded from the target.
lean_mtmd = '''add_library(mtmd
            mtmd.cpp
            mtmd-audio.cpp
            mtmd-image.cpp
            mtmd.h
            mtmd-internal.h
            mtmd-helper.cpp
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

# Upstream also names mtmd-helper-gen.cpp in source-property metadata. Once that
# translation unit is removed from add_library(mtmd), leaving it in this metadata
# list is unnecessary and previously caused a false-positive whole-file verifier.
old_props = '''set_source_files_properties(
    mtmd-helper.cpp
    mtmd-helper-gen.cpp
    PROPERTIES SKIP_PRECOMPILE_HEADERS ON
)'''
new_props = '''set_source_files_properties(
    mtmd-helper.cpp
    PROPERTIES SKIP_PRECOMPILE_HEADERS ON
)'''
if old_props in cmake:
    cmake = cmake.replace(old_props, new_props, 1)
elif "mtmd-helper-gen.cpp" in mtmd_target_block(cmake):
    raise SystemExit("mtmd-helper-gen.cpp unexpectedly remains in the MTMD target")
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

# Generic clip.cpp audio-generation state handling references Qwen3-TTS and
# PocketTTS symbols. Director cannot select those projector types, so remove the
# unreachable references rather than compiling their large generation modules.
audio_func = "static std::vector<c2w_state_slot> list_gen_state_slots"
audio_pos = clip.find(audio_func)
if audio_pos < 0:
    raise SystemExit("Pinned generic audio-state helper not found; refusing unsafe pruning")
audio_brace = clip.find("{", audio_pos)
depth = 0
audio_end = -1
for i in range(audio_brace, len(clip)):
    ch = clip[i]
    if ch == "{":
        depth += 1
    elif ch == "}":
        depth -= 1
        if depth == 0:
            audio_end = i + 1
            break
if audio_end < 0:
    raise SystemExit("Could not locate end of pinned generic audio-state helper")

lean_audio = '''static std::vector<c2w_state_slot> list_gen_state_slots(const clip_hparams & hparams, const clip_model & model) {
    (void) hparams;
    (void) model;
    return {};
}'''
clip = clip[:audio_pos] + lean_audio + clip[audio_end:]
clip_path.write_text(clip, encoding="utf-8")

verification = cmake_path.read_text(encoding="utf-8")
target = mtmd_target_block(verification)
for required in ("models/qwen2vl.cpp", "models/qwen3vl.cpp", "mtmd-helper.cpp"):
    if required not in target:
        raise SystemExit(f"Director mtmd pruning lost required target source: {required}")
for forbidden in (
    "models/gemma4v.cpp",
    "models/llava.cpp",
    "models/deepseekocr.cpp",
    "models/whisper-enc.cpp",
    "models/qwen3tts-gen.cpp",
    "models/pockettts-gen.cpp",
    "mtmd-helper-gen.cpp",
):
    if forbidden in target:
        raise SystemExit(f"Director mtmd pruning left forbidden target source: {forbidden}")

clip_verification = clip_path.read_text(encoding="utf-8")
for forbidden_ref in ("list_c2w_state_slots(hparams, model)", "list_pockettts_state_slots(hparams, model)"):
    if forbidden_ref in clip_verification:
        raise SystemExit(f"Director mtmd pruning left unreachable audio symbol reference: {forbidden_ref}")

if not profile_ok():
    raise SystemExit("Director Qwen3-VL pruning v3 verification failed")
marker.write_text("qwen3vl-only-v3\n", encoding="utf-8")
print("Pruned llama.cpp mtmd to the Director Qwen3-VL still-image profile v3")
