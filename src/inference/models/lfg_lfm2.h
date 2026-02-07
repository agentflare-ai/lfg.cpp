#pragma once

#include "../lfg_graph.h"
#include "../lfg_model.h"
#include <type_traits>

template <bool iswa>
struct lfg_build_lfm2 : public lfg_graph_context {
    const lfg_model & model;

    lfg_build_lfm2(const lfg_model & model, const lfg_graph_params & params);
    
    ggml_tensor * build_moe_feed_forward(ggml_tensor * cur, int il) const;
    ggml_tensor * build_dense_feed_forward(ggml_tensor * cur, int il) const;
    ggml_tensor * build_attn_block(ggml_tensor * cur, ggml_tensor * inp_pos, std::conditional_t<iswa, lfg_graph_input_attn_kv_iswa, lfg_graph_input_attn_kv> * inp_attn, int il) const;
    ggml_tensor * build_shortconv_block(ggml_tensor * cur, lfg_graph_input_rs * inp_recr, int il);
};
