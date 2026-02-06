cmake_minimum_required(VERSION 3.20)

# Wrapper to use llama.cpp's ggml implementation directly.
find_package(Threads REQUIRED)

if (NOT DEFINED GGML_SCHED_MAX_COPIES)
    set(GGML_SCHED_MAX_COPIES 4)
endif()

if (NOT DEFINED GGML_SCHED_MAX_BACKENDS)
    set(GGML_SCHED_MAX_BACKENDS 16)
endif()

add_compile_definitions(
    GGML_SCHED_MAX_COPIES=${GGML_SCHED_MAX_COPIES}
    GGML_SCHED_MAX_BACKENDS=${GGML_SCHED_MAX_BACKENDS}
)
set(LLAMA_GGML_DIR "${PROJECT_SOURCE_DIR}/third_party/llama.cpp/ggml")

add_subdirectory(
    "${LLAMA_GGML_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/llama-ggml"
)

# Preserve Liquid target names expected across the build.
if (TARGET ggml-base AND NOT TARGET lfg_ggml_base)
    add_library(lfg_ggml_base ALIAS ggml-base)
endif()

if (TARGET ggml AND NOT TARGET lfg_ggml)
    add_library(lfg_ggml ALIAS ggml)
endif()

if (TARGET ggml AND NOT TARGET lfg_ggml::ggml)
    add_library(lfg_ggml::ggml ALIAS ggml)
endif()
