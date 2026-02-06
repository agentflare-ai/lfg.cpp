#include <spdlog/spdlog.h>
#include "inference/lfg_api.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        spdlog::error("Usage: {} <model_path>", argv[0]);
        return 1;
    }

    spdlog::info("LFG CLI v1.0");

    lfg_model_load_config config = lfg_model_load_default_config();
    config.model_path = argv[1];
    config.use_mmap = true;
    config.n_gpu_layers = 0; // Force CPU

    spdlog::info("Loading model: {}...", config.model_path);
    auto* model = lfg_load_model(&config);

    if (model) {
        spdlog::info("Model loaded successfully!");
        lfg_model_free(model);
        spdlog::info("Model freed.");
    } else {
        spdlog::error("Failed to load model.");
        return 1;
    }

    return 0;
}
