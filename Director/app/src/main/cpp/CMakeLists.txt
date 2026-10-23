cmake_minimum_required(VERSION 3.22.1)
project("director-vlm" LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Codemagic builds the Android debug variant with CMAKE_BUILD_TYPE=Debug. The
# default Debug flags add large DWARF sections to the statically-linked
# llama.cpp/MTMD library, after which AGP spends a long time stripping them.
# Director's CI debug APK does not need native source-level debugging, so remove
# debug metadata only. This does not change generated code, optimization level,
# model behavior, sampling, camera behavior, or image quality.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS_DEBUG "-g0" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-g0" CACHE STRING "" FORCE)
    set(CMAKE_ASM_FLAGS_DEBUG "-g0" CACHE STRING "" FORCE)
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_SUBPROCESS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)

if(ANDROID_ABI STREQUAL "arm64-v8a")
    # Director uses llama.cpp's standard ARM/NEON CPU backend. KleidiAI remains
    # disabled because it is an optional performance layer with a very large
    # Android cross-compilation cost; model behavior and output are unchanged.
    set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
else()
    message(FATAL_ERROR "Director AI currently supports arm64-v8a only")
endif()

set(LLAMA_SRC "${CMAKE_CURRENT_LIST_DIR}/../../../../third_party/llama.cpp")
if(NOT EXISTS "${LLAMA_SRC}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp is missing. Run scripts/fetch_llama_cpp.sh first.")
endif()
if(NOT EXISTS "${LLAMA_SRC}/.director-qwen3vl-pruned-v3")
    message(FATAL_ERROR "llama.cpp is not prepared for Director. Run scripts/fetch_llama_cpp.sh first.")
endif()

add_subdirectory("${LLAMA_SRC}" "${CMAKE_BINARY_DIR}/llama-build")
add_library(director-vlm SHARED director_vlm.cpp)

target_include_directories(director-vlm PRIVATE
    "${LLAMA_SRC}/include"
    "${LLAMA_SRC}/ggml/include"
    "${LLAMA_SRC}/tools/mtmd"
)

# The NDK already emits function/data sections. Drop only sections that are not
# reachable from Director's JNI exports. This keeps every reachable llama.cpp,
# MTMD and Qwen3-VL code path byte-for-byte compiled with the same -O3 settings.
target_link_options(director-vlm PRIVATE "-Wl,--gc-sections")

target_link_libraries(director-vlm PRIVATE llama mtmd android log)
