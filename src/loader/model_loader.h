#pragma once

#include <string>
#include <vector>
#include "lfm_inference.h" // Was liquid.h

namespace liquid {

class ModelLoader {
public:
    struct ModelConfig {
        std::string model_path;
        bool use_mmap = true;
        bool use_mlock = false;
        int n_gpu_layers = 0; // 0 for CPU-only
    };

    /**
     * Quantization Support:
     * Supported Types: Q4_K_M (Default), Q5_K_M, Q8_0, F16, etc. (Standard GGUF types)
     * Memory Impact:
     * - Q4_K_M: ~6-bit effective per weight. Low memory, good performance.
     * - Q5_K_M: ~7-bit effective. Higher memory, slight accuracy boost.
     * - F16: 16-bit. x2.5 memory of Q4. High fidelity.
     * 
     * Performance:
     * - CPU inference scales with bandwidth. Q4 is generally fastest.
     */

    struct ModelStats {
        uint64_t n_params;
        uint64_t size_bytes;
        int32_t n_vocab;
        int32_t n_ctx_train;
    };

    struct ContextConfig {
        uint32_t n_ctx = 0; // 0 = from model
        uint32_t n_batch = 512;
        int32_t n_threads = 1;
        bool flash_attn = false;
    };

    static lfm_model* LoadModel(const ModelConfig& config);
    static void FreeModel(lfm_model* model);

    static ModelStats GetModelStats(lfm_model* model);
    static std::string GetMetadata(lfm_model* model, const std::string& key);
    static lfm_context* CreateContext(lfm_model* model, const ContextConfig& config);
    static void FreeContext(lfm_context* ctx);
};

} // namespace liquid
