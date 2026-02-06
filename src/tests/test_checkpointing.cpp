#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/inference_core.h"
#include "../loader/model_loader.h"
#include <chrono>
#include <fstream>
#include <spdlog/spdlog.h>

using namespace liquid;

TEST_CASE("InferenceCore Checkpointing") {
    lfm_backend_init();

    std::string model_path = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
    std::ifstream f(model_path);
    if (!f.good()) {
        MESSAGE("Skipping checkpoint test: Model not found at " << model_path);
        return;
    }

    ModelLoader::ModelConfig load_config;
    load_config.model_path = model_path;
    lfm_model* model = ModelLoader::LoadModel(load_config);
    REQUIRE(model != nullptr);

    InferenceCore::Config config;
    config.n_ctx = 512;
    // Fix seed for determinism (though greedy doesn't need it as much, it's good practice)
    config.sampling.seed = 42; 
    config.sampling.temp = 0.0f; // Greedy sampling for strict determinism
    InferenceCore core(model, config);

    // Initial prompt
    std::vector<lfm_token> tokens = {1, 15043}; // BOS, "Hello"
    core.IngestTokens(tokens);
    core.Decode();

    // Generate prefix
    for(int i=0; i<5; ++i) {
        lfm_token t = core.Sample();
        core.IngestTokens({t});
        core.Decode();
    }

    SUBCASE("Determinism Check") {
        // Create Checkpoint S1
        auto s1 = core.CreateCheckpoint();
        
        // Path A: Generate 5 tokens
        std::vector<lfm_token> path_a;
        for(int i=0; i<5; ++i) {
            lfm_token t = core.Sample();
            path_a.push_back(t);
            core.IngestTokens({t});
            core.Decode();
        }

        // Restore S1
        bool ok = core.RestoreCheckpoint(s1);
        CHECK(ok);

        // Path B: Generate 5 tokens
        std::vector<lfm_token> path_b;
        for(int i=0; i<5; ++i) {
            lfm_token t = core.Sample();
            path_b.push_back(t);
            core.IngestTokens({t});
            core.Decode();
        }

        // Verify A == B
        CHECK(path_a.size() == path_b.size());
        for(size_t i=0; i<path_a.size(); ++i) {
            CHECK(path_a[i] == path_b[i]);
        }
    }

    SUBCASE("Reasoning State Configuration") {
        // Just verify API doesn't crash
        std::vector<lfm_token> start = {32001};
        std::vector<lfm_token> end = {32002};
        core.ConfigureReasoning(start, end);
        CHECK(true);
    }
    
    // Note: Sampler state verification is implicit in Determinism Check
    // because samplers like Repetition Penalty depend on history.
    // If restore didn't re-ingest history, the penalty state would be different
    // and potentially lead to different sampling if logits were close.

    lfm_model_free(model);
}

TEST_CASE("Checkpointing Benchmark") {
    // Micro-benchmark for overhead
    lfm_backend_init();
    std::string model_path = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
    std::ifstream f(model_path);
    if (!f.good()) return;

    ModelLoader::ModelConfig load_config;
    load_config.model_path = model_path;
    lfm_model* model = ModelLoader::LoadModel(load_config);
    
    InferenceCore::Config config;
    config.n_ctx = 2048;
    InferenceCore core(model, config);
    
    // Fill up some context (e.g. 100 tokens)
    std::vector<lfm_token> prompt(100, 15043); // 100 "Hello"s
    core.IngestTokens(prompt);
    
    // Benchmark CreateCheckpoint
    auto start_create = std::chrono::high_resolution_clock::now();
    auto cp = core.CreateCheckpoint();
    auto end_create = std::chrono::high_resolution_clock::now();
    auto duration_create = std::chrono::duration_cast<std::chrono::microseconds>(end_create - start_create).count();
    
    MESSAGE("CreateCheckpoint overhead (100 tokens): " << duration_create << " us");
    CHECK(duration_create < 1000); // Should be < 1ms (likely < 10us)

    // Benchmark RestoreCheckpoint
    auto start_restore = std::chrono::high_resolution_clock::now();
    core.RestoreCheckpoint(cp);
    auto end_restore = std::chrono::high_resolution_clock::now();
    auto duration_restore = std::chrono::duration_cast<std::chrono::microseconds>(end_restore - start_restore).count();
    
    MESSAGE("RestoreCheckpoint overhead (100 tokens, truncate+re-ingest): " << duration_restore << " us");
    CHECK(duration_restore < 5000); // Strict: < 5ms. 
    
    lfm_model_free(model);
}
