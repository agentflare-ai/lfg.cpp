#include <spdlog/spdlog.h>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include "../inference/lfg_api.h"

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

void run_benchmark(lfg_session * session, const std::string& name, int n_tokens, bool structured) {
    if (structured) {
        lfg_session_configure_structured(session, JSON_GRAMMAR.c_str(), "root");
    } else {
        lfg_session_configure_structured(session, "", "");
    }

    lfg_session_reset(session);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Seed with a token (1 = BOS/Start) to generate logits
    lfg_token bos = 1;
    lfg_session_ingest_tokens(session, &bos, 1, false); // Don't update sampler for BOS

    int generated = 0;
    for (int i = 0; i < n_tokens; ++i) {
        auto token = lfg_session_sample(session);
        lfg_session_ingest_tokens(session, &token, 1, false); // Sample() already accepted the token
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

    lfg_model_load_config config = lfg_model_load_default_config();
    config.model_path = argv[1];
    config.use_mmap = true;
    config.n_gpu_layers = 0; // CPU benchmark

    spdlog::info("Loading model: {}...", config.model_path);
    auto* model = lfg_load_model(&config);

    if (!model) {
        spdlog::error("Failed to load model.");
        return 1;
    }

    lfg_session_config session_config = lfg_session_default_config();
    session_config.n_threads = 8;
    session_config.n_ctx = 2048;

    lfg_session * session = lfg_session_create(model, &session_config);

    spdlog::info("Starting Benchmark...");

    // Run Standard
    run_benchmark(session, "Standard (Run 1)", 100, false);
    run_benchmark(session, "Standard (Run 2)", 100, false);

    // Run Structured
    run_benchmark(session, "Structured (Run 1)", 100, true);
    run_benchmark(session, "Structured (Run 2)", 100, true);

    lfg_session_free(session);
    lfg_model_free(model);
    return 0;
}
