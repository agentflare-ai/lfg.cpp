add_library(liquid
    lfg_adapter.cpp
    lfg_arch.cpp
    lfg_batch.cpp
    lfg_chat.cpp
    lfg_context.cpp
    lfg_cparams.cpp
    lfg_grammar.cpp
    json_schema_to_grammar.cpp
    lfg_graph.cpp
    lfg_hparams.cpp
    lfg_impl.cpp
    lfg_inference.cpp
    lfg_io.cpp
    lfg_kv_cache.cpp
    lfg_kv_cache_iswa.cpp
    lfg_memory.cpp
    lfg_memory_hybrid.cpp
    lfg_memory_hybrid_iswa.cpp
    lfg_memory_recurrent.cpp
    lfg_mmap.cpp
    lfg_model.cpp
    lfg_model_loader.cpp
    lfg_model_saver.cpp
    lfg_quant.cpp
    lfg_sampling.cpp
    lfg_sampler_prefix.cpp
    lfg_vocab.cpp
    unicode-data.cpp
    unicode.cpp
    models/lfm2.cpp
    lfg_api.cpp
)



target_include_directories(liquid PRIVATE .)
target_include_directories(liquid PUBLIC 
    . 
    ../../third_party/llama.cpp/vendor
)

target_link_libraries(liquid PUBLIC lfg_ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(liquid PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(liquid PRIVATE LFG_BUILD)
    target_compile_definitions(liquid PUBLIC  LFG_SHARED)
endif()

# Strict warnings
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Werror)
endif()
