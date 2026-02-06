#pragma once

#include "lfg_inference.h"

#ifdef __cplusplus
#include "ggml-cpp.h"

#include <string>
#include <unordered_map>
#include <vector>
#endif

// TODO: pimpl

//
// lfg_adapter_cvec
//

#ifdef __cplusplus
struct lfg_adapter_cvec {
    ggml_tensor * tensor_for(int il) const;

    ggml_tensor * apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const;

    bool apply(
            const lfg_model & model,
            const float * data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

private:
    bool init(const lfg_model & model);

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<ggml_tensor *> tensors; // per layer
};

//
// lfg_adapter_lora
//

struct lfg_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    lfg_adapter_lora_weight() = default;
    lfg_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b) : a(a), b(b) {}
};

struct lfg_adapter_lora {
    // map tensor name to lora_a_b
    std::unordered_map<std::string, lfg_adapter_lora_weight> ab_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // activated lora (aLoRA)
    std::vector<lfg_token> alora_invocation_tokens;

    lfg_adapter_lora() = default;
    ~lfg_adapter_lora() = default;

    lfg_adapter_lora_weight * get_weight(ggml_tensor * w);

    uint32_t get_n_nodes() const {
        return ab_map.size() * 6u; // a, b, scale, add, 2 x mul_mat
    }
};

using lfg_adapter_loras = std::unordered_map<lfg_adapter_lora *, float>;
#endif
