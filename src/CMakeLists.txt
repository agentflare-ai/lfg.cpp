cmake_minimum_required(VERSION 3.12)
project(lfg)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GGML_CPU ON CACHE BOOL "Enable CPU backend" FORCE)

# Add subdirectories
add_subdirectory(ggml)
add_subdirectory(inference)
add_subdirectory(eval)

# Loader Library
add_library(loader loader/model_loader.cpp)
target_include_directories(loader PUBLIC loader)
target_link_libraries(loader PUBLIC liquid)

# Main Executable
add_executable(lfg main.cpp)
target_link_libraries(lfg PRIVATE loader)

# Tests
add_subdirectory(tests)

# Benchmarks
add_subdirectory(benchmarks)
