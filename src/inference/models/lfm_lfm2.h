#pragma once

#include "../lfm_graph.h"
#include "../lfm_model.h"
#include <type_traits>

template <bool iswa>
struct lfm_build_lfm2 : public llm_graph_context {
    const lfm_model & model;

    lfm_build_lfm2(const lfm_model & model, const llm_graph_params & params);
    
    ggml_tensor * build_moe_feed_forward(ggml_tensor * cur, int il) const;
    ggml_tensor * build_dense_feed_forward(ggml_tensor * cur, int il) const;
    ggml_tensor * build_attn_block(ggml_tensor * cur, ggml_tensor * inp_pos, std::conditional_t<iswa, llm_graph_input_attn_kv_iswa, llm_graph_input_attn_kv> * inp_attn, int il) const;
    ggml_tensor * build_shortconv_block(ggml_tensor * cur, llm_graph_input_rs * inp_recr, int il);
};
