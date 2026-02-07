#pragma once

#include "lfg_inference.h"
#include "lfg_arch.h"
#include "lfg_graph.h"
#include "lfg_hparams.h"
#include "lfg_memory.h"
#include "lfg_vocab.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct lfg_cparams;
struct lfg_ubatch;
struct lfg_model_loader;

// available models
enum lfg_type_enum {
    LFG_TYPE_UNKNOWN,
    LFG_TYPE_14M,
    LFG_TYPE_17M,
    LFG_TYPE_22M,
    LFG_TYPE_33M,
    LFG_TYPE_47M,
    LFG_TYPE_60M,
    LFG_TYPE_70M,
    LFG_TYPE_80M,
    LFG_TYPE_109M,
    LFG_TYPE_137M,
    LFG_TYPE_140M,
    LFG_TYPE_149M,
    LFG_TYPE_160M,
    LFG_TYPE_190M,
    LFG_TYPE_220M,
    LFG_TYPE_250M,
    LFG_TYPE_256M,
    LFG_TYPE_270M,
    LFG_TYPE_335M,
    LFG_TYPE_350M,
    LFG_TYPE_360M,
    LFG_TYPE_395M,
    LFG_TYPE_410M,
    LFG_TYPE_450M,
    LFG_TYPE_475M,
    LFG_TYPE_558M,
    LFG_TYPE_700M,
    LFG_TYPE_770M,
    LFG_TYPE_780M,
    LFG_TYPE_950M,
    LFG_TYPE_0_3B,
    LFG_TYPE_0_5B,
    LFG_TYPE_0_6B,
    LFG_TYPE_1B,
    LFG_TYPE_1_2B,
    LFG_TYPE_1_3B,
    LFG_TYPE_1_4B,
    LFG_TYPE_1_5B,
    LFG_TYPE_1_6B,
    LFG_TYPE_1_7B,
    LFG_TYPE_1_8B,
    LFG_TYPE_2B,
    LFG_TYPE_2_6B,
    LFG_TYPE_2_8B,
    LFG_TYPE_2_9B,
    LFG_TYPE_3B,
    LFG_TYPE_4B,
    LFG_TYPE_6B,
    LFG_TYPE_6_9B,
    LFG_TYPE_7B,
    LFG_TYPE_8B,
    LFG_TYPE_9B,
    LFG_TYPE_11B,
    LFG_TYPE_12B,
    LFG_TYPE_13B,
    LFG_TYPE_14B,
    LFG_TYPE_15B,
    LFG_TYPE_16B,
    LFG_TYPE_20B,
    LFG_TYPE_26B,
    LFG_TYPE_27B,
    LFG_TYPE_30B,
    LFG_TYPE_32B,
    LFG_TYPE_34B,
    LFG_TYPE_35B,
    LFG_TYPE_36B,
    LFG_TYPE_40B,
    LFG_TYPE_65B,
    LFG_TYPE_70B,
    LFG_TYPE_120B,
    LFG_TYPE_142B,
    LFG_TYPE_236B,
    LFG_TYPE_290B,
    LFG_TYPE_314B,
    LFG_TYPE_405B,
    LFG_TYPE_671B,
    LFG_TYPE_SMALL,
    LFG_TYPE_MEDIUM,
    LFG_TYPE_LARGE,
    LFG_TYPE_XL,
    LFG_TYPE_A1_7B,
    LFG_TYPE_A2_7B,
    LFG_TYPE_8x7B,
    LFG_TYPE_8x22B,
    LFG_TYPE_16x12B,
    LFG_TYPE_16x3_8B,
    LFG_TYPE_10B_128x3_66B,
    LFG_TYPE_57B_A14B,
    LFG_TYPE_17B_16E, // liquid4 Scout
    LFG_TYPE_17B_128E, // liquid4 Maverick
    LFG_TYPE_A13B,
    LFG_TYPE_7B_A1B,
    LFG_TYPE_8B_A1B, // lfm2moe
    LFG_TYPE_16B_A1B,
    LFG_TYPE_21B_A3B, // Ernie MoE small
    LFG_TYPE_30B_A3B,
    LFG_TYPE_31B_A3_5B,
    LFG_TYPE_80B_A3B, // Qwen3 Next
    LFG_TYPE_100B_A6B,
    LFG_TYPE_102B_A12B, // Solar-Open
    LFG_TYPE_106B_A12B, // GLM-4.5-Air
    LFG_TYPE_230B_A10B, // Minimax M2
    LFG_TYPE_235B_A22B,
    LFG_TYPE_300B_A47B, // Ernie MoE big
    LFG_TYPE_310B_A15B, // /MiMo-V2-Flash
    LFG_TYPE_355B_A32B, // GLM-4.5
    LFG_TYPE_E2B,
    LFG_TYPE_E4B,
};

std::string lfg_rope_scaling_type_name(lfg_rope_scaling_type rope_scaling_type);

struct lfg_layer_posnet {
    // resnet
    struct ggml_tensor * norm1   = nullptr;
    struct ggml_tensor * norm1_b = nullptr;

    struct ggml_tensor * conv1   = nullptr;
    struct ggml_tensor * conv1_b = nullptr;

    struct ggml_tensor * norm2   = nullptr;
    struct ggml_tensor * norm2_b = nullptr;

    struct ggml_tensor * conv2   = nullptr;
    struct ggml_tensor * conv2_b = nullptr;

    // attention
    struct ggml_tensor * attn_norm   = nullptr;
    struct ggml_tensor * attn_norm_b = nullptr;

    struct ggml_tensor * attn_q   = nullptr;
    struct ggml_tensor * attn_q_b = nullptr;

    struct ggml_tensor * attn_k   = nullptr;
    struct ggml_tensor * attn_k_b = nullptr;

    struct ggml_tensor * attn_v   = nullptr;
    struct ggml_tensor * attn_v_b = nullptr;

    struct ggml_tensor * attn_o   = nullptr;
    struct ggml_tensor * attn_o_b = nullptr;

    // normalize
    struct ggml_tensor * norm   = nullptr;
    struct ggml_tensor * norm_b = nullptr;
};

struct lfg_layer_convnext {
    struct ggml_tensor * dw   = nullptr;
    struct ggml_tensor * dw_b = nullptr;

    struct ggml_tensor * norm   = nullptr;
    struct ggml_tensor * norm_b = nullptr;

    struct ggml_tensor * pw1   = nullptr;
    struct ggml_tensor * pw1_b = nullptr;

    struct ggml_tensor * pw2   = nullptr;
    struct ggml_tensor * pw2_b = nullptr;

    struct ggml_tensor * gamma = nullptr;
};

struct lfg_layer_shortconv {
    struct ggml_tensor * in_proj  = nullptr;
    struct ggml_tensor * conv     = nullptr;
    struct ggml_tensor * out_proj = nullptr;
};

struct lfg_layer_nextn {
    struct ggml_tensor * eh_proj          = nullptr;
    struct ggml_tensor * embed_tokens     = nullptr;
    struct ggml_tensor * enorm            = nullptr;
    struct ggml_tensor * hnorm            = nullptr;
    struct ggml_tensor * shared_head_head = nullptr;
    struct ggml_tensor * shared_head_norm = nullptr;
};

struct lfg_layer {
    // normalization
    struct ggml_tensor * attn_norm       = nullptr;
    struct ggml_tensor * attn_norm_b     = nullptr;
    struct ggml_tensor * attn_norm_2     = nullptr;
    struct ggml_tensor * attn_norm_2_b   = nullptr;
    struct ggml_tensor * attn_q_norm     = nullptr;
    struct ggml_tensor * attn_q_norm_b   = nullptr;
    struct ggml_tensor * attn_k_norm     = nullptr;
    struct ggml_tensor * attn_k_norm_b   = nullptr;
    struct ggml_tensor * attn_out_norm   = nullptr;
    struct ggml_tensor * attn_out_norm_b = nullptr;
    struct ggml_tensor * attn_q_a_norm   = nullptr;
    struct ggml_tensor * attn_kv_a_norm  = nullptr;
    struct ggml_tensor * attn_sub_norm   = nullptr;
    struct ggml_tensor * attn_post_norm  = nullptr;
    struct ggml_tensor * ffn_sub_norm    = nullptr;
    struct ggml_tensor * attn_norm_cross = nullptr;
    struct ggml_tensor * attn_norm_enc   = nullptr;
    struct ggml_tensor * ssm_norm        = nullptr;
    struct ggml_tensor * ssm_dt_norm     = nullptr;
    struct ggml_tensor * ssm_b_norm      = nullptr;
    struct ggml_tensor * ssm_c_norm      = nullptr;

    // attention
    struct ggml_tensor * wq        = nullptr;
    struct ggml_tensor * wk        = nullptr;
    struct ggml_tensor * wv        = nullptr;
    struct ggml_tensor * wo        = nullptr;
    struct ggml_tensor * wqkv      = nullptr;
    struct ggml_tensor * wq_a      = nullptr;
    struct ggml_tensor * wq_b      = nullptr;
    struct ggml_tensor * wkv_a_mqa = nullptr;
    struct ggml_tensor * wkv_b     = nullptr;
    struct ggml_tensor * wk_b      = nullptr;
    struct ggml_tensor * wv_b      = nullptr;
    struct ggml_tensor * wq_cross  = nullptr;
    struct ggml_tensor * wk_cross  = nullptr;
    struct ggml_tensor * wv_cross  = nullptr;
    struct ggml_tensor * wo_cross  = nullptr;
    struct ggml_tensor * wq_enc    = nullptr;
    struct ggml_tensor * wk_enc    = nullptr;
    struct ggml_tensor * wv_enc    = nullptr;
    struct ggml_tensor * wo_enc    = nullptr;
    struct ggml_tensor * wqkv_gate = nullptr;

    // attention bias
    struct ggml_tensor * bq   = nullptr;
    struct ggml_tensor * bk   = nullptr;
    struct ggml_tensor * bv   = nullptr;
    struct ggml_tensor * bo   = nullptr;
    struct ggml_tensor * bqkv = nullptr;

    // relative position bias
    struct ggml_tensor * attn_rel_b       = nullptr;
    struct ggml_tensor * attn_rel_b_enc   = nullptr;
    struct ggml_tensor * attn_rel_b_cross = nullptr;

    // normalization
    struct ggml_tensor * ffn_norm         = nullptr;
    struct ggml_tensor * ffn_norm_b       = nullptr;
    struct ggml_tensor * ffn_post_norm    = nullptr;
    struct ggml_tensor * layer_out_norm   = nullptr;
    struct ggml_tensor * layer_out_norm_b = nullptr;
    struct ggml_tensor * ffn_norm_exps    = nullptr;
    struct ggml_tensor * ffn_norm_enc     = nullptr;

    // ff
    struct ggml_tensor * ffn_gate     = nullptr; // w1
    struct ggml_tensor * ffn_down     = nullptr; // w2
    struct ggml_tensor * ffn_up       = nullptr; // w3
    struct ggml_tensor * ffn_gate_enc = nullptr;
    struct ggml_tensor * ffn_down_enc = nullptr;
    struct ggml_tensor * ffn_up_enc   = nullptr;

    // ff MoE
    struct ggml_tensor * ffn_gate_inp    = nullptr;
    struct ggml_tensor * ffn_gate_exps   = nullptr;
    struct ggml_tensor * ffn_down_exps   = nullptr;
    struct ggml_tensor * ffn_up_exps     = nullptr;
    struct ggml_tensor * ffn_gate_inp_b  = nullptr;
    struct ggml_tensor * ffn_gate_exps_b = nullptr;
    struct ggml_tensor * ffn_down_exps_b = nullptr;
    struct ggml_tensor * ffn_up_exps_b   = nullptr;

    // ff shared expert (shexp)
    struct ggml_tensor * ffn_gate_inp_shexp = nullptr;
    struct ggml_tensor * ffn_gate_shexp     = nullptr;
    struct ggml_tensor * ffn_down_shexp     = nullptr;
    struct ggml_tensor * ffn_up_shexp       = nullptr;

    // ff adjugate experts (chexps)
    struct ggml_tensor * ffn_gate_chexps     = nullptr;
    struct ggml_tensor * ffn_down_chexps     = nullptr;
    struct ggml_tensor * ffn_up_chexps       = nullptr;

    // ff bias
    struct ggml_tensor * ffn_gate_b = nullptr;
    struct ggml_tensor * ffn_down_b = nullptr; // b2
    struct ggml_tensor * ffn_up_b   = nullptr; // b3
    struct ggml_tensor * ffn_act    = nullptr;
    struct ggml_tensor * ffn_exp_probs_b = nullptr;

    // mamba proj
    struct ggml_tensor * ssm_in  = nullptr;
    struct ggml_tensor * ssm_x   = nullptr;
    struct ggml_tensor * ssm_dt  = nullptr;
    struct ggml_tensor * ssm_out = nullptr;

    // mamba
    struct ggml_tensor * ssm_conv1d = nullptr;
    struct ggml_tensor * ssm_a      = nullptr;
    struct ggml_tensor * ssm_d      = nullptr;

    // mamba bias
    struct ggml_tensor * ssm_conv1d_b = nullptr;
    struct ggml_tensor * ssm_dt_b     = nullptr;

    // qwen3next
    struct ggml_tensor * ssm_beta_alpha = nullptr;

    // rwkv
    struct ggml_tensor * time_mix_w1         = nullptr;
    struct ggml_tensor * time_mix_w2         = nullptr;
    struct ggml_tensor * time_mix_lerp_x     = nullptr;
    struct ggml_tensor * time_mix_lerp_w     = nullptr;
    struct ggml_tensor * time_mix_lerp_k     = nullptr;
    struct ggml_tensor * time_mix_lerp_v     = nullptr;
    struct ggml_tensor * time_mix_lerp_r     = nullptr;
    struct ggml_tensor * time_mix_lerp_g     = nullptr;
    struct ggml_tensor * time_mix_lerp_fused = nullptr;

    struct ggml_tensor * time_mix_first        = nullptr;
    struct ggml_tensor * time_mix_decay        = nullptr;
    struct ggml_tensor * time_mix_decay_w1     = nullptr;
    struct ggml_tensor * time_mix_decay_w2     = nullptr;
    struct ggml_tensor * time_mix_key          = nullptr;
    struct ggml_tensor * time_mix_key_b        = nullptr;
    struct ggml_tensor * time_mix_value        = nullptr;
    struct ggml_tensor * time_mix_value_b      = nullptr;
    struct ggml_tensor * time_mix_receptance   = nullptr;
    struct ggml_tensor * time_mix_receptance_b = nullptr;
    struct ggml_tensor * time_mix_gate         = nullptr;

    // rwkv7
    struct ggml_tensor * time_mix_w0         = nullptr;
    struct ggml_tensor * time_mix_a0         = nullptr;
    struct ggml_tensor * time_mix_a1         = nullptr;
    struct ggml_tensor * time_mix_a2         = nullptr;
    struct ggml_tensor * time_mix_v0         = nullptr;
    struct ggml_tensor * time_mix_v1         = nullptr;
    struct ggml_tensor * time_mix_v2         = nullptr;
    struct ggml_tensor * time_mix_g1         = nullptr;
    struct ggml_tensor * time_mix_g2         = nullptr;
    struct ggml_tensor * time_mix_k_k        = nullptr;
    struct ggml_tensor * time_mix_k_a        = nullptr;
    struct ggml_tensor * time_mix_r_k        = nullptr;

    struct ggml_tensor * time_mix_ln     = nullptr;
    struct ggml_tensor * time_mix_ln_b   = nullptr;
    struct ggml_tensor * time_mix_output = nullptr;

    struct ggml_tensor * channel_mix_lerp_k = nullptr;
    struct ggml_tensor * channel_mix_lerp_r = nullptr;

    struct ggml_tensor * channel_mix_key        = nullptr;
    struct ggml_tensor * channel_mix_receptance = nullptr;
    struct ggml_tensor * channel_mix_value      = nullptr;

    // long rope factors
    struct ggml_tensor * rope_long  = nullptr;
    struct ggml_tensor * rope_short = nullptr;
    struct ggml_tensor * rope_freqs = nullptr;

    // bitnet scale
    struct ggml_tensor * wq_scale       = nullptr;
    struct ggml_tensor * wk_scale       = nullptr;
    struct ggml_tensor * wv_scale       = nullptr;
    struct ggml_tensor * wo_scale       = nullptr;
    struct ggml_tensor * ffn_gate_scale = nullptr;
    struct ggml_tensor * ffn_up_scale   = nullptr;
    struct ggml_tensor * ffn_down_scale = nullptr;

    // altup & laurel
    struct ggml_tensor * per_layer_inp_gate   = nullptr;
    struct ggml_tensor * per_layer_proj       = nullptr;
    struct ggml_tensor * per_layer_post_norm  = nullptr;
    struct ggml_tensor * altup_correct_coef   = nullptr;
    struct ggml_tensor * altup_correct_scale  = nullptr;
    struct ggml_tensor * altup_predict_coef   = nullptr;
    struct ggml_tensor * altup_router         = nullptr;
    struct ggml_tensor * altup_router_norm    = nullptr;
    struct ggml_tensor * laurel_l             = nullptr;
    struct ggml_tensor * laurel_r             = nullptr;
    struct ggml_tensor * laurel_post_norm     = nullptr;

    // openai-moe
    struct ggml_tensor * attn_sinks = nullptr;

    // cogvlm
    struct ggml_tensor * visexp_attn_wqkv = nullptr;
    struct ggml_tensor * visexp_attn_wo   = nullptr;
    struct ggml_tensor * visexp_ffn_gate  = nullptr;
    struct ggml_tensor * visexp_ffn_down  = nullptr;
    struct ggml_tensor * visexp_ffn_up    = nullptr;

    // xIELU activation parameters for Apertus
    struct ggml_tensor * ffn_act_alpha_n = nullptr;
    struct ggml_tensor * ffn_act_alpha_p = nullptr;
    struct ggml_tensor * ffn_act_beta    = nullptr;
    struct ggml_tensor * ffn_act_eps     = nullptr;

    struct lfg_layer_posnet posnet;

    struct lfg_layer_convnext convnext;

    struct lfg_layer_shortconv shortconv;

    struct lfg_layer_nextn nextn;
};

struct lfg_model {
    lfg_type_enum type = LFG_TYPE_UNKNOWN;
    lfg_arch_enum arch = LFG_ARCH_UNKNOWN;

    std::string name = "n/a";

    lfg_hparams hparams = {};
    lfg_vocab   vocab;

    // for classifier models
    std::vector<std::string> classifier_labels;

    struct ggml_tensor * tok_embd   = nullptr;
    struct ggml_tensor * type_embd  = nullptr;
    struct ggml_tensor * pos_embd   = nullptr;
    struct ggml_tensor * tok_norm   = nullptr;
    struct ggml_tensor * tok_norm_b = nullptr;

    struct ggml_tensor * output_norm     = nullptr;
    struct ggml_tensor * output_norm_b   = nullptr;
    struct ggml_tensor * output          = nullptr;
    struct ggml_tensor * output_b        = nullptr;
    struct ggml_tensor * output_norm_enc = nullptr;

    // classifier
    struct ggml_tensor * cls       = nullptr;
    struct ggml_tensor * cls_b     = nullptr;
    struct ggml_tensor * cls_out   = nullptr;
    struct ggml_tensor * cls_out_b = nullptr;

    struct ggml_tensor * conv1d   = nullptr;
    struct ggml_tensor * conv1d_b = nullptr;

    // gemma3n altup
    struct ggml_tensor * tok_embd_per_layer   = nullptr;
    struct ggml_tensor * altup_proj           = nullptr;
    struct ggml_tensor * altup_unembd_proj    = nullptr;
    struct ggml_tensor * per_layer_model_proj = nullptr;
    struct ggml_tensor * per_layer_proj_norm  = nullptr;

    std::vector<lfg_layer> layers;

    //Dense linear projections for SentenceTransformers models like embeddinggemma
    // For Sentence Transformers models structure see
    // https://sbert.net/docs/sentence_transformer/usage/custom_models.html#structure-of-sentence-transformer-models
    struct ggml_tensor * dense_2_out_layers   = nullptr;
    struct ggml_tensor * dense_2_out_layers_b = nullptr;
    struct ggml_tensor * dense_3_out_layers   = nullptr;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // list of devices used in this model
    std::vector<ggml_backend_dev_t> devices;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    // for keeping track of associated LoRA adapters
    std::unordered_set<lfg_adapter_lora *> loras;

    int64_t t_load_us  = 0;
    int64_t t_start_us = 0;

    explicit lfg_model(const struct lfg_model_params & params);
    ~lfg_model();

    void load_stats  (lfg_model_loader & ml);
    void load_arch   (lfg_model_loader & ml);
    void load_hparams(lfg_model_loader & ml);
    void load_vocab  (lfg_model_loader & ml);
    bool load_tensors(lfg_model_loader & ml); // returns false if cancelled by progress_callback

    std::string arch_name() const;
    std::string type_name() const;

    std::string desc() const;

    size_t size() const; // file size
    size_t n_tensors() const;
    size_t n_devices() const;

    uint32_t n_gpu_layers() const;
    lfg_split_mode split_mode() const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // total number of parameters in the model
    uint64_t n_elements() const;

    void print_info() const;

    ggml_backend_dev_t dev_layer(uint32_t il) const;
    ggml_backend_dev_t dev_output() const;

    ggml_backend_buffer_type_t select_buft(uint32_t il) const;

    bool has_tensor_overrides() const;

    const struct ggml_tensor * get_tensor(const char * name) const;

    float get_rope_freq_base (const lfg_cparams & cparams, uint32_t il) const;
    float get_rope_freq_scale(const lfg_cparams & cparams, uint32_t il) const;

    ggml_tensor * get_rope_factors(const lfg_cparams & cparams, uint32_t il) const;

    // TODO: move this to new lfg_arch_enum_model_i interface
    lfg_memory_i * create_memory(const lfg_memory_params & params, const lfg_cparams & cparams) const;

    // TODO: move this to new lfg_arch_enum_model_i interface
    ggml_cgraph * build_graph(const lfg_graph_params & params) const;

private:
    lfg_model_params params;

    struct impl;
    std::unique_ptr<impl> pimpl;
};

const char * lfg_type_name(lfg_type_enum type);

// For internal test use
// TODO: remove
const std::vector<std::pair<std::string, ggml_tensor *>> & lfg_internal_get_tensor_map(const lfg_model * model);
