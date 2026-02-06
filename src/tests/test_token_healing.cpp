#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/inference_core.h"
#include "../loader/model_loader.h"
#include <spdlog/spdlog.h>

#include <chrono>

using namespace liquid;

TEST_CASE("Token Healing") {
    lfm_backend_init();

    std::string model_path = "models/lfm2-350M.gguf";
    ModelLoader::ModelConfig load_config;
    load_config.model_path = model_path;
    load_config.n_gpu_layers = 0; 

    lfm_model* model = ModelLoader::LoadModel(load_config);
    if (!model) {
        MESSAGE("Model not found, skipping test.");
        return;
    }

    InferenceCore::Config config;
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f; // Deterministic
    config.enable_healing = true; // Enable optimization
    InferenceCore core(model, config);

    SUBCASE("Benchmark: TPS Comparison") {
        std::string prompt = "The quick brown fox jumps over the lazy";
        const auto* vocab = lfm_model_get_vocab(model);
        std::vector<lfm_token> tokens(100);
        int n_tokens = lfm_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, false);
        tokens.resize(n_tokens);

        auto benchmark_run = [&](bool healing_enabled) -> double {
            core.Reset();
            // Hack to toggle healing without public API access to config
            // We assume 'core' was initialized with healing=true in this TEST_CASE context from previous turns?
            // Wait, InferenceCore config is const member? No, struct passed by value.
            // But we can't change it after construction.
            // So we need to construct a NEW core for each run.
            
            InferenceCore::Config cfg;
            cfg.n_ctx = 2048;
            cfg.enable_healing = healing_enabled;
            InferenceCore local_core(model, cfg);

            auto t0 = std::chrono::high_resolution_clock::now();
            
            // 1. Ingest
            local_core.IngestTokens(tokens);
            
            // 2. Heal (if enabled)
            if (healing_enabled) {
                local_core.HealLastToken();
            }

            auto t1 = std::chrono::high_resolution_clock::now(); // TTFT point

            // 3. Generate 20 tokens
            for (int i = 0; i < 20; ++i) {
                local_core.Decode();
                auto t = local_core.Sample();
                local_core.IngestTokens({t});
            }

            auto t2 = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double> total_time = t2 - t0;
            std::chrono::duration<double> decode_time = t2 - t1;
            
            double ttft_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double tps = 20.0 / decode_time.count();
            
            MESSAGE((healing_enabled ? "[With Healing]" : "[No Healing]") 
                    << " TTFT: " << ttft_ms << " ms"
                    << " | Decode TPS: " << tps);
            
            spdlog::info("{}: TTFT: {} ms | TPS: {}", 
                      (healing_enabled ? "With Healing:    " : "Without Healing: "),
                      ttft_ms, tps);
            
            return tps;
        };

        double tps_no_heal = benchmark_run(false);
        double tps_heal = benchmark_run(true);
        
        // Healing affects TTFT, but TPS should be roughly equal (decoding is identical)
        // Allowing 10% variance for noise
        CHECK(tps_heal >= tps_no_heal * 0.9);
    }

    lfm_model_free(model);
}
