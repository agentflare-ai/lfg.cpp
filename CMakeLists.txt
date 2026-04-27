cmake_minimum_required(VERSION 3.20)
project(lfg VERSION 1.1 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LFG_ENABLE_SANITIZERS "Enable sanitizers for robustness testing" OFF)
set(LFG_SANITIZERS "address,undefined" CACHE STRING "Sanitizers to enable when LFG_ENABLE_SANITIZERS is ON")

option(LFG_BUILD_FUZZERS "Build libFuzzer targets" OFF)
set(LFG_FUZZ_SANITIZERS "address,undefined" CACHE STRING "Sanitizers to enable for fuzzers")

option(LFG_BUILD_DEMO "Build the macOS ImGui demo application" OFF)

include(GNUInstallDirs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(FetchContent)

# Use spdlog in header-only mode to avoid shared library dependency at runtime
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.12.0
)
FetchContent_MakeAvailable(spdlog)

if(LFG_BUILD_DEMO)
    if(NOT APPLE)
        message(FATAL_ERROR "LFG_BUILD_DEMO is only supported on macOS.")
    endif()

    enable_language(OBJC)

    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        glfw
        URL https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(glfw)
endif()

# Optimization and architecture flags (apply globally)
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(LFG_ENABLE_SANITIZERS)
        add_compile_options(-O1 -g -fno-omit-frame-pointer)
        add_compile_options(-fsanitize=${LFG_SANITIZERS})
        add_link_options(-fsanitize=${LFG_SANITIZERS})
        add_compile_options(-fno-sanitize-recover=all)
        add_link_options(-fno-sanitize-recover=all)
    else()
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
            add_compile_options(-O3 -DNDEBUG -march=armv8.2-a+dotprod -mtune=native)
        else()
            add_compile_options(-O3 -DNDEBUG -march=native -mtune=native)
        endif()
    endif()
    # Suppress warnings about system headers using reserved suffixes (macOS SDK issue)
    add_compile_options(-Wno-reserved-user-defined-literal)
endif()

# Strict warning flags for our own source code (not third-party)
set(LFG_STRICT_WARNINGS
    -Wall
    -Wextra
    -Wpedantic
)

# Help FindOpenMP on macOS with Homebrew
if(APPLE)
    if(EXISTS "/opt/homebrew/opt/libomp")
        list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp")
        set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
    elseif(EXISTS "/usr/local/opt/libomp")
        list(APPEND CMAKE_PREFIX_PATH "/usr/local/opt/libomp")
        set(OpenMP_ROOT "/usr/local/opt/libomp")
    endif()
endif()

add_compile_definitions(GGML_VERSION="1.0" GGML_COMMIT="unknown")

# Includes
include_directories(third_party/llama.cpp/ggml/include)
include_directories(src/inference)
include_directories(src/loader)

# Third-party includes with relaxed warnings
include_directories(SYSTEM
    third_party/llama.cpp/src
    third_party/llama.cpp/include
    third_party/llama.cpp/common
)



# Enable llama.cpp options
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Enable GGML backends
set(GGML_CPU ON CACHE BOOL "" FORCE)
if(APPLE)
    option(GGML_METAL "Enable Metal backend" ON)
endif()

# GGML / llama.cpp
# Note: third_party/llama.cpp provides its own ggml.
# We use that instead of src/ggml to avoid target collisions and ensure compatibility with liquid-audio.
# add_subdirectory(third_party/llama.cpp)
add_subdirectory(src/ggml)

# Build llama.cpp core against the same ggml we use for lfg_core.
file(GLOB LLAMA_CORE_SOURCES
    "third_party/llama.cpp/src/*.cpp"
    "third_party/llama.cpp/src/models/*.cpp"
)
add_library(llama_core_local STATIC ${LLAMA_CORE_SOURCES})
target_include_directories(llama_core_local PRIVATE
    third_party/llama.cpp/src
    third_party/llama.cpp/include
    third_party/llama.cpp/ggml/include
)
target_link_libraries(llama_core_local PUBLIC lfg_ggml)

# Inference Core (LFM) Source
file(GLOB INFERENCE_SOURCES 
    "src/inference/*.cpp"
    "src/inference/models/*.cpp"
)

# Model Loader Source
# Model Loader Source
file(GLOB LOADER_SOURCES "src/loader/*.cpp")

set(LFG_COMMON_SOURCES
    third_party/llama.cpp/common/peg-parser.cpp
    third_party/llama.cpp/common/unicode.cpp
)

# New Vision Sources (Refactored)
set(VISION_SOURCES
    "src/vision/clip.cpp"
    "src/vision/siglip.cpp"
)

# Allow STB image inclusion
include_directories(SYSTEM "third_party/llama.cpp/vendor")

# Allow common headers inclusion (e.g. common/common.h for audio-decoder)
include_directories(SYSTEM "third_party/llama.cpp")

# Vision Library (Refactored)
add_library(lfg_vision STATIC
    src/vision/clip.cpp
    src/vision/siglip.cpp
)
# Relax warnings for MTMD/Vision due to legacy code
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(lfg_vision PRIVATE -w) 
endif()
target_link_libraries(lfg_vision PUBLIC lfg_ggml spdlog::spdlog_header_only)

# Library
add_library(lfg_core STATIC
    ${INFERENCE_SOURCES}
    ${LOADER_SOURCES}
    ${LFG_COMMON_SOURCES}
)

# Apply strict warnings only to our code
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(lfg_core PRIVATE ${LFG_STRICT_WARNINGS})
endif()

# Link against ggml from llama.cpp
# Also link lfg_mtmd
target_link_libraries(lfg_core PUBLIC lfg_ggml lfg_vision spdlog::spdlog_header_only pthread dl)

# Main Executable (Placeholder for now, or CLI later)
add_executable(lfg_cli src/main.cpp)
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(lfg_cli PRIVATE ${LFG_STRICT_WARNINGS})
endif()
target_link_libraries(lfg_cli PRIVATE lfg_core spdlog::spdlog_header_only)

if(LFG_BUILD_DEMO)
    add_library(lfg_imgui STATIC
        third_party/imgui/imgui.cpp
        third_party/imgui/imgui_demo.cpp
        third_party/imgui/imgui_draw.cpp
        third_party/imgui/imgui_tables.cpp
        third_party/imgui/imgui_widgets.cpp
        third_party/imgui/backends/imgui_impl_glfw.cpp
        third_party/imgui/backends/imgui_impl_opengl3.cpp
    )
    target_include_directories(lfg_imgui PUBLIC
        third_party/imgui
        third_party/imgui/backends
        ${glfw_SOURCE_DIR}/include
    )
    if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(lfg_imgui PRIVATE -w)
    endif()
    target_link_libraries(lfg_imgui PUBLIC glfw)

    find_library(APPKIT_FRAMEWORK AppKit REQUIRED)
    find_library(UNIFORM_TYPE_IDENTIFIERS_FRAMEWORK UniformTypeIdentifiers REQUIRED)
    find_library(OPENGL_FRAMEWORK OpenGL REQUIRED)

    add_executable(lfg_demo
        tools/demo/main.cpp
        tools/demo/file_dialog.m
    )
    set_target_properties(lfg_demo PROPERTIES OUTPUT_NAME lfg-demo)
    set_source_files_properties(
        tools/demo/file_dialog.m
        PROPERTIES COMPILE_OPTIONS "-fno-objc-arc;-Wno-deprecated-declarations"
    )
    target_include_directories(lfg_demo PRIVATE
        tools/demo
        third_party/imgui
        third_party/imgui/backends
        ${glfw_SOURCE_DIR}/include
    )
    if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(lfg_demo PRIVATE ${LFG_STRICT_WARNINGS})
    endif()
    target_link_libraries(lfg_demo PRIVATE
        lfg_core
        lfg_imgui
        glfw
        ${APPKIT_FRAMEWORK}
        ${UNIFORM_TYPE_IDENTIFIERS_FRAMEWORK}
        ${OPENGL_FRAMEWORK}
    )
endif()


add_subdirectory(src/eval)
add_subdirectory(src/tests)
add_subdirectory(src/benchmarks)

if(LFG_BUILD_FUZZERS)
    add_subdirectory(src/fuzz)
endif()
