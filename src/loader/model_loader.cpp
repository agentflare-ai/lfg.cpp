#include "model_loader.h"
#include <spdlog/spdlog.h>
#include <vector>
#include <string>
#include "lfm_inference.h"

namespace liquid {

lfm_model* ModelLoader::LoadModel(const ModelConfig& config) {
    lfm_model_params params = lfm_model_default_params();
    
    params.use_mmap = config.use_mmap;
    params.use_mlock = config.use_mlock;
    params.n_gpu_layers = config.n_gpu_layers;

    lfm_model* model = lfm_model_load_from_file(config.model_path.c_str(), params);
    
    if (!model) {
        spdlog::error("Failed to load model from {}", config.model_path);
        return nullptr;
    }

    return model;
}

std::string ModelLoader::GetMetadata(lfm_model* model, const std::string& key) {
    if (!model) return "";
    std::vector<char> buf(256);
    int32_t len = lfm_model_meta_val_str(model, key.c_str(), buf.data(), buf.size());
    if (len < 0) {
        return "";
    }
    return std::string(buf.data());
}

void ModelLoader::FreeModel(lfm_model* model) {
    if (model) {
        lfm_model_free(model);
    }
}

ModelLoader::ModelStats ModelLoader::GetModelStats(lfm_model* model) {
    ModelStats stats = {};
    if (!model) return stats;

    stats.n_params = lfm_model_n_params(model);
    stats.size_bytes = lfm_model_size(model);
    stats.n_vocab = lfm_vocab_n_tokens(lfm_model_get_vocab(model));
    stats.n_ctx_train = lfm_model_n_ctx_train(model);
    return stats;
}

lfm_context* ModelLoader::CreateContext(lfm_model* model, const ContextConfig& config) {
    if (!model) return nullptr;

    lfm_context_params params = lfm_context_default_params();
    params.n_ctx = config.n_ctx;
    params.n_batch = config.n_batch;
    params.n_threads = config.n_threads;
    params.n_threads_batch = config.n_threads;
    
    if (config.flash_attn) {
        params.flash_attn_type = LFM_FLASH_ATTN_TYPE_ENABLED;
    }

    return lfm_init_from_model(model, params);
}

void ModelLoader::FreeContext(lfm_context* ctx) {
    if (ctx) {
        lfm_free(ctx);
    }
}

} // namespace liquid
