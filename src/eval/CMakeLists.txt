# lfg-eval: The harness app (using LFG)
find_package(Threads REQUIRED)

add_executable(lfg-eval
    eval.cpp
)
target_include_directories(lfg-eval PRIVATE ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor)
target_include_directories(lfg-eval PRIVATE ${CMAKE_SOURCE_DIR}/third_party/llama.cpp)
target_include_directories(lfg-eval PRIVATE ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include)
target_include_directories(lfg-eval PRIVATE ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include)
# Vision headers are now local or part of lfg_vision interface
target_link_libraries(lfg-eval PRIVATE lfm_core lfm_ggml Threads::Threads)

# llama-compare: Standalone llama.cpp comparison tool
# Links against in-tree llama.cpp sources with the same ggml build as liquid
add_executable(llama-compare
    llama_compare.cpp
)
target_include_directories(llama-compare PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
)
target_link_libraries(llama-compare PRIVATE
    llama_core_local
    Threads::Threads
)

# lfg-compare: Minimal lfg comparison tool (mirrors llama-compare exactly)
add_executable(lfg-compare
    lfg_compare.cpp
)
target_link_libraries(lfg-compare PRIVATE lfm_core lfm_ggml Threads::Threads)

# lfg-structured-compare: LFG structured decoding + healing/checkpointing
add_executable(lfg-structured-compare
    lfg_structured_compare.cpp
)
target_link_libraries(lfg-structured-compare PRIVATE lfm_core lfm_ggml Threads::Threads)

# llama-structured-compare: llama.cpp structured decoding (no healing/checkpoint)
add_executable(llama-structured-compare
    llama_structured_compare.cpp
    ../inference/json_schema_to_grammar.cpp
)
target_include_directories(llama-structured-compare PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
)
target_link_libraries(llama-structured-compare PRIVATE
    llama_core_local
    spdlog::spdlog_header_only
    Threads::Threads
)