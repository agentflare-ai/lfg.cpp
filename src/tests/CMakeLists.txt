add_executable(test_loader test_loader.cpp)
target_link_libraries(test_loader PRIVATE lfm_core)
target_include_directories(test_loader PRIVATE ../loader)


add_executable(test_inference test_inference.cpp)
target_link_libraries(test_inference PRIVATE lfm_core)
target_include_directories(test_inference PRIVATE ../inference ../loader)

add_executable(test_c_api_errors test_c_api_errors.cpp)
target_link_libraries(test_c_api_errors PRIVATE lfm_core)
target_include_directories(test_c_api_errors PRIVATE ../inference)
add_test(NAME test_c_api_errors COMMAND test_c_api_errors)

# JSON Sampling Test
add_executable(test_json_sampling test_json_sampling.cpp)
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(test_json_sampling PRIVATE ${LFG_STRICT_WARNINGS})
endif()
target_link_libraries(test_json_sampling PRIVATE lfm_core)
target_include_directories(test_json_sampling PRIVATE ../inference ../loader)

# Tool Calling Test
add_executable(test_tool_calling test_tool_calling.cpp)
target_link_libraries(test_tool_calling PRIVATE lfm_core)
target_include_directories(test_tool_calling PRIVATE ../inference ../loader)
add_test(NAME test_tool_calling COMMAND test_tool_calling)

# Exhaustive 350M Tooling Test
add_executable(test_350m_exhaustive_tooling test_350m_exhaustive_tooling.cpp)
target_link_libraries(test_350m_exhaustive_tooling PRIVATE lfm_core)
target_include_directories(test_350m_exhaustive_tooling PRIVATE ../inference ../loader)
add_test(NAME test_350m_exhaustive_tooling COMMAND test_350m_exhaustive_tooling)

# Exhaustive LFM2.5-1.2B Thinking Tooling Test
add_executable(test_lfm25_thinking_exhaustive_tooling test_lfm25_thinking_exhaustive_tooling.cpp)
target_link_libraries(test_lfm25_thinking_exhaustive_tooling PRIVATE lfm_core)
target_include_directories(test_lfm25_thinking_exhaustive_tooling PRIVATE ../inference ../loader)
add_test(NAME test_lfm25_thinking_exhaustive_tooling COMMAND test_lfm25_thinking_exhaustive_tooling)

# LFM2.5-1.2B Thinking Chat Template Tooling Test
add_executable(test_lfm25_thinking_chat_tooling test_lfm25_thinking_chat_tooling.cpp)
target_link_libraries(test_lfm25_thinking_chat_tooling PRIVATE lfm_core)
target_include_directories(test_lfm25_thinking_chat_tooling PRIVATE ../inference ../loader)
add_test(NAME test_lfm25_thinking_chat_tooling COMMAND test_lfm25_thinking_chat_tooling)

# Checkpointing Test
add_executable(test_checkpointing test_checkpointing.cpp)
target_link_libraries(test_checkpointing PRIVATE lfm_core)
target_include_directories(test_checkpointing PRIVATE ../loader ../inference ../ggml src)
add_test(NAME test_checkpointing COMMAND test_checkpointing)

# Leap Bundle Test
add_executable(test_leap_bundle test_leap_bundle.cpp)
target_link_libraries(test_leap_bundle PRIVATE lfm_core)
target_include_directories(test_leap_bundle PRIVATE ../loader ../inference ../ggml src)
add_test(NAME test_leap_bundle COMMAND test_leap_bundle ${CMAKE_SOURCE_DIR}/models/lfm2-350M.gguf)

# Token Healing Test
add_executable(test_token_healing test_token_healing.cpp)
target_link_libraries(test_token_healing PRIVATE lfm_core)
target_include_directories(test_token_healing PRIVATE ../inference ../loader ../ggml src)
add_test(NAME test_token_healing COMMAND test_token_healing)

# Token Healing Integration Test
add_executable(test_healing_integration test_healing_integration.cpp)
target_link_libraries(test_healing_integration PRIVATE lfm_core)
target_include_directories(test_healing_integration PRIVATE ../inference ../loader ../ggml src)
add_test(NAME test_healing_integration COMMAND test_healing_integration)

# Structured Checkpointing Test
add_executable(test_structured_checkpointing test_structured_checkpointing.cpp)
target_link_libraries(test_structured_checkpointing PRIVATE lfm_core)
target_include_directories(test_structured_checkpointing PRIVATE ../inference ../loader ../ggml src)
add_test(NAME test_structured_checkpointing COMMAND test_structured_checkpointing)

# Structured Decoding Double-Accept Regression Test
add_executable(test_structured_double_accept test_structured_double_accept.cpp)
target_link_libraries(test_structured_double_accept PRIVATE lfm_core)
target_include_directories(test_structured_double_accept PRIVATE ../inference ../loader ../ggml src)
add_test(NAME test_structured_double_accept COMMAND test_structured_double_accept)

# Reasoning Healing Integration Test
add_executable(test_reasoning_healing_integration test_reasoning_healing_integration.cpp)
target_link_libraries(test_reasoning_healing_integration PRIVATE lfm_core lfm_ggml)
target_compile_features(test_reasoning_healing_integration PRIVATE cxx_std_17)
add_test(NAME test_reasoning_healing_integration COMMAND test_reasoning_healing_integration)

add_executable(test_complex_reasoning_healing test_complex_reasoning_healing.cpp)

target_link_libraries(test_complex_reasoning_healing PRIVATE lfm_core lfm_ggml)

target_compile_features(test_complex_reasoning_healing PRIVATE cxx_std_17)

add_test(NAME test_complex_reasoning_healing COMMAND test_complex_reasoning_healing)



# Model Capabilities Test (Parameterized)



add_executable(test_model_capabilities test_model_capabilities.cpp)



target_link_libraries(test_model_capabilities PRIVATE lfm_core lfm_vision)



target_include_directories(test_model_capabilities PRIVATE ../inference ../loader ../vision ../ggml src)
add_test(NAME test_model_capabilities COMMAND test_model_capabilities --model ${CMAKE_SOURCE_DIR}/models/lfm2-350M.gguf)

add_test(NAME test_loader COMMAND test_loader)
add_test(NAME test_inference COMMAND test_inference)
add_test(NAME test_json_sampling COMMAND test_json_sampling ${CMAKE_SOURCE_DIR}/models/lfm2-350M.gguf)
