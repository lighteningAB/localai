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
