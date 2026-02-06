add_library(liquid
    lfm_adapter.cpp
    lfm_arch.cpp
    lfm_batch.cpp
    lfm_chat.cpp
    lfm_context.cpp
    lfm_cparams.cpp
    lfm_grammar.cpp
    json_schema_to_grammar.cpp
    lfm_graph.cpp
    lfm_hparams.cpp
    lfm_impl.cpp
    lfm_inference.cpp
    lfm_io.cpp
    lfm_kv_cache.cpp
    lfm_kv_cache_iswa.cpp
    lfm_memory.cpp
    lfm_memory_hybrid.cpp
    lfm_memory_hybrid_iswa.cpp
    lfm_memory_recurrent.cpp
    lfm_mmap.cpp
    lfm_model.cpp
    lfm_model_loader.cpp
    lfm_model_saver.cpp
    lfm_quant.cpp
    lfm_sampling.cpp
    lfm_sampler_prefix.cpp
    lfm_vocab.cpp
    unicode-data.cpp
    unicode.cpp
    models/lfm2.cpp
    inference_core.cpp
    structured_decoding_engine.cpp
)



target_include_directories(liquid PRIVATE .)
target_include_directories(liquid PUBLIC 
    . 
    ../../third_party/llama.cpp/vendor
)

target_link_libraries(liquid PUBLIC lfm_ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(liquid PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(liquid PRIVATE LFM_BUILD)
    target_compile_definitions(liquid PUBLIC  LFM_SHARED)
endif()

# Strict warnings
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Werror)
endif()
