cmake_minimum_required(VERSION 3.18)
project(imagegen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# QNN_SDK_ROOT is passed from Gradle (-D arg). Empty string means "not configured".
# Phase 1 does not link QNN, so we tolerate absence here; Phase 4 will require it.
if(DEFINED QNN_SDK_ROOT AND NOT QNN_SDK_ROOT STREQUAL "")
    message(STATUS "imagegen: QNN_SDK_ROOT = ${QNN_SDK_ROOT}")
    set(IMAGEGEN_HAS_QNN ON)
else()
    message(WARNING "imagegen: QNN_SDK_ROOT not set. libimagegen.so will build, but "
                    "Phase 4+ (QNN runtime) cannot run. Set qnn.sdk.root in "
                    "local.properties or export QNN_SDK_ROOT before building.")
    set(IMAGEGEN_HAS_QNN OFF)
endif()

# Alibaba MNN — text encoder runtime (xororz ships CLIP as clip_v2.mnn).
# Configured to build statically into libimagegen.so with OpenCL backend.
# Override defaults BEFORE add_subdirectory so MNN picks them up.
set(MNN_BUILD_SHARED_LIBS  OFF CACHE INTERNAL "" FORCE)
set(MNN_SEP_BUILD          OFF CACHE INTERNAL "" FORCE)
set(MNN_BUILD_TOOLS        OFF CACHE INTERNAL "" FORCE)
set(MNN_BUILD_QUANTOOLS    OFF CACHE INTERNAL "" FORCE)
set(MNN_BUILD_CONVERTER    OFF CACHE INTERNAL "" FORCE)
set(MNN_BUILD_PROTOBUFFER  OFF CACHE INTERNAL "" FORCE)
set(MNN_BUILD_DEMO         OFF CACHE INTERNAL "" FORCE)
set(MNN_BUILD_TRAIN        OFF CACHE INTERNAL "" FORCE)
set(MNN_BUILD_BENCHMARK    OFF CACHE INTERNAL "" FORCE)
set(MNN_BUILD_TEST         OFF CACHE INTERNAL "" FORCE)
set(MNN_USE_THREAD_POOL    ON  CACHE INTERNAL "" FORCE)
set(MNN_OPENCL             ON  CACHE INTERNAL "" FORCE)
set(MNN_ARM82              ON  CACHE INTERNAL "" FORCE)
set(MNN_USE_LOGCAT         ON  CACHE INTERNAL "" FORCE)
# KleidiAI uses relative paths in its CMakeLists that break under
# add_subdirectory from a parent project. Disable.
set(MNN_KLEIDIAI           OFF CACHE INTERNAL "" FORCE)
add_subdirectory(3rdparty/MNN EXCLUDE_FROM_ALL)

add_library(imagegen SHARED
    imagegen.cpp
    scheduler.cpp
    bundle_loader.cpp
    tokenizer.cpp
    qnn_session.cpp
    mnn_session.cpp
    diffusion.cpp
    segformer.cpp
    mask_ops.cpp
    outfit_swap.cpp
    png_encode.cpp
)

target_include_directories(imagegen PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/stb
)

# stb_image_write is a single-header public-domain library; suppress the
# noisy warnings it triggers under -Wall -Wextra -Wpedantic.
set_source_files_properties(png_encode.cpp PROPERTIES COMPILE_FLAGS
    "-Wno-cast-qual -Wno-missing-field-initializers -Wno-unused-function -Wno-pedantic -Wno-sign-compare")

# Pull in QNN headers + flip on the real qnn_session implementation when QAIRT
# is configured. Without it, qnn_session.cpp falls back to a stub that reports
# the missing configuration cleanly.
if(IMAGEGEN_HAS_QNN)
    target_include_directories(imagegen PRIVATE
        ${QNN_SDK_ROOT}/include/QNN
    )
    target_compile_definitions(imagegen PRIVATE IMAGEGEN_HAS_QNN=1)
endif()

target_compile_options(imagegen PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -fvisibility=hidden
)

target_link_libraries(imagegen PRIVATE
    MNN
    log
    android
    dl
    m
    GLESv2
    EGL
)

target_include_directories(imagegen PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/MNN/include
)

# ============================================================================
# llama.cpp engine (libllmcpp.so) — second LLM backend beside Kotlin LiteRT-LM.
# Hosts Gemma 4 12B QAT Q4_0 on the CPU (the LiteRT-LM GPU runtime hits the
# Adreno 1 GB logits-alloc cap; see ModelId.kt / LiteRT-LM #2461).
#
# llama + ggml are built STATIC and linked into our single SHARED libllmcpp.so
# (same self-contained pattern as MNN -> imagegen above). CPU-only for now.
# ============================================================================
set(BUILD_SHARED_LIBS      OFF CACHE INTERNAL "" FORCE)   # static llama+ggml
set(LLAMA_BUILD_COMMON     OFF CACHE INTERNAL "" FORCE)   # hand-roll Gemma prompt + raw sampler; skip common (curl dep)
set(LLAMA_BUILD_TESTS      OFF CACHE INTERNAL "" FORCE)
set(LLAMA_BUILD_TOOLS      OFF CACHE INTERNAL "" FORCE)
set(LLAMA_BUILD_EXAMPLES   OFF CACHE INTERNAL "" FORCE)
set(LLAMA_BUILD_SERVER     OFF CACHE INTERNAL "" FORCE)
set(LLAMA_BUILD_APP        OFF CACHE INTERNAL "" FORCE)
set(LLAMA_BUILD_HTML       OFF CACHE INTERNAL "" FORCE)
set(LLAMA_BUILD_UI         OFF CACHE INTERNAL "" FORCE)
set(LLAMA_CURL             OFF CACHE INTERNAL "" FORCE)   # no libcurl on device
set(GGML_OPENMP            OFF CACHE INTERNAL "" FORCE)   # NDK ships no OpenMP
set(GGML_LLAMAFILE         OFF CACHE INTERNAL "" FORCE)   # incompatible on Android
set(GGML_OPENCL            OFF CACHE INTERNAL "" FORCE)   # CPU-only milestone
set(GGML_HEXAGON           OFF CACHE INTERNAL "" FORCE)   # NPU path deferred
set(GGML_CPU_REPACK        ON  CACHE INTERNAL "" FORCE)   # Q4_0 -> Q4_X_X for i8mm
# Default NDK arm64-v8a is baseline armv8-a: ggml's dotprod/i8mm compile-checks
# fail, so Q4_0 matmul falls back to plain NEON (~4x slower — measured <1 tok/s
# for 12B). Pin the arch so the int8 matmul kernels (i8mm = vmmlaq_s32) build.
# Safe here because the 12B model is RAM-gated to flagship SoCs (SM8xxx) that
# all implement dotprod + i8mm.
set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+i8mm+fp16" CACHE INTERNAL "" FORCE)
add_subdirectory(3rdparty/llama.cpp EXCLUDE_FROM_ALL)

add_library(llmcpp SHARED
    llamacpp/llama_bridge.cpp
)

target_link_libraries(llmcpp PRIVATE
    llama
    log
    android
)

target_include_directories(llmcpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/ggml/include
)

target_compile_options(llmcpp PRIVATE -fvisibility=hidden)
