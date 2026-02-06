#include <spdlog/spdlog.h>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include "model_loader.h"
#include "inference_core.h"

// Simple JSON grammar for a list of items
const std::string JSON_GRAMMAR = R"({
  "type": "object",
  "properties": {
    "items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": { "type": "string" },
          "id": { "type": "integer" }
        },
        "required": ["name", "id"]
      }
    }
  },
  "required": ["items"]
})";

void run_benchmark(liquid::InferenceCore& core, const std::string& name, int n_tokens, bool structured) {
    if (structured) {
        core.ConfigureStructuredDecoding(JSON_GRAMMAR, "root");
    } else {
        core.ConfigureStructuredDecoding("", "");
    }

    core.Reset();

    // Warmup / Prompt
    std::string prompt = "Generate a JSON list of 5 items with 'name' and 'id'.";
    // NOTE: In a real app we'd tokenize this. For now, let's assume valid start state or just empty prompt.
    // If we can't easily tokenize without the vocab helper (which is hidden in model), 
    // we might just start generating from BOS.
    // However, InferenceCore doesn't expose Tokenize directly in header? 
    // It seems `IngestTokens` expects tokens.
    // Let's assume we can just start generation (empty context).
    
    // Actually, let's tokenize " {" to bias it if not structured? 
    // Or just run generation.
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Seed with a token (1 = BOS/Start) to generate logits
    std::vector<int32_t> start_tokens = {1}; 
    core.IngestTokens(start_tokens, false); // Don't update sampler for BOS, strictly follow grammar after

    int generated = 0;
    for (int i = 0; i < n_tokens; ++i) {
        // if (!core.Decode()) break; // Decode is empty/dummy
        auto token = core.Sample();
        std::vector<int32_t> tokens = {token};
        core.IngestTokens(tokens, false); // Sample() already accepted the token
        generated++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    double tps = generated / elapsed.count();
    
    spdlog::info("Benchmark [{}]: {} tokens in {:.2f}s ({} t/s)", 
              name, generated, elapsed.count(), tps);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        spdlog::error("Usage: {} <model_path>", argv[0]);
        return 1;
    }

    liquid::ModelLoader::ModelConfig config;
    config.model_path = argv[1];
    config.use_mmap = true;
    config.n_gpu_layers = 100; // Try GPU if available? Protocol says CPU benchmark.
    // Actually user protocol said "benchmark comparing ... on CPU".
    config.n_gpu_layers = 0; 
    // config.n_threads = 8; // Not in ModelConfig

    spdlog::info("Loading model: {}...", config.model_path);
    auto* model = liquid::ModelLoader::LoadModel(config);

    if (!model) {
        spdlog::error("Failed to load model.");
        return 1;
    }

    liquid::InferenceCore::Config core_config;
    core_config.n_threads = 8;
    core_config.n_ctx = 2048;
    
    liquid::InferenceCore core(model, core_config);

    spdlog::info("Starting Benchmark...");

    // Run Standard
    run_benchmark(core, "Standard (Run 1)", 100, false);
    run_benchmark(core, "Standard (Run 2)", 100, false);

    // Run Structured
    run_benchmark(core, "Structured (Run 1)", 100, true);
    run_benchmark(core, "Structured (Run 2)", 100, true);

    liquid::ModelLoader::FreeModel(model);
    return 0;
}
