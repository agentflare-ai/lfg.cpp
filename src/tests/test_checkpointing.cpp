#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/lfg_api.h"
#include <chrono>
#include <fstream>
#include <spdlog/spdlog.h>

TEST_CASE("InferenceCore Checkpointing") {
    lfg_backend_init();

    std::string model_path = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
    std::ifstream f(model_path);
    if (!f.good()) {
        MESSAGE("Skipping checkpoint test: Model not found at " << model_path);
        return;
    }

    lfg_model_load_config load_config = lfg_model_load_default_config();
    load_config.model_path = model_path.c_str();
    lfg_model* model = lfg_load_model(&load_config);
    REQUIRE(model != nullptr);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 512;
    // Fix seed for determinism (though greedy doesn't need it as much, it's good practice)
    config.sampling.seed = 42;
    config.sampling.temp = 0.0f; // Greedy sampling for strict determinism
    lfg_session *session = lfg_session_create(model, &config);

    // Initial prompt
    lfg_token initial_tokens[] = {1, 15043}; // BOS, "Hello"
    lfg_session_ingest_tokens(session, initial_tokens, 2, true);
    lfg_session_decode(session);

    // Generate prefix
    for(int i=0; i<5; ++i) {
        lfg_token t = lfg_session_sample(session);
        lfg_session_ingest_tokens(session, &t, 1, true);
        lfg_session_decode(session);
    }

    SUBCASE("Determinism Check") {
        // Create Checkpoint S1
        lfg_checkpoint *s1 = lfg_session_create_checkpoint(session);

        // Path A: Generate 5 tokens
        std::vector<lfg_token> path_a;
        for(int i=0; i<5; ++i) {
            lfg_token t = lfg_session_sample(session);
            path_a.push_back(t);
            lfg_session_ingest_tokens(session, &t, 1, true);
            lfg_session_decode(session);
        }

        // Restore S1
        bool ok = lfg_session_restore_checkpoint(session, s1);
        CHECK(ok);

        // Path B: Generate 5 tokens
        std::vector<lfg_token> path_b;
        for(int i=0; i<5; ++i) {
            lfg_token t = lfg_session_sample(session);
            path_b.push_back(t);
            lfg_session_ingest_tokens(session, &t, 1, true);
            lfg_session_decode(session);
        }

        // Verify A == B
        CHECK(path_a.size() == path_b.size());
        for(size_t i=0; i<path_a.size(); ++i) {
            CHECK(path_a[i] == path_b[i]);
        }

        lfg_checkpoint_free(s1);
    }

    SUBCASE("Reasoning State Configuration") {
        // Just verify API doesn't crash
        lfg_token start[] = {32001};
        lfg_token end[] = {32002};
        lfg_session_configure_reasoning(session, start, 1, end, 1);
        CHECK(true);
    }

    // Note: Sampler state verification is implicit in Determinism Check
    // because samplers like Repetition Penalty depend on history.
    // If restore didn't re-ingest history, the penalty state would be different
    // and potentially lead to different sampling if logits were close.

    lfg_session_free(session);
    lfg_model_free(model);
}

TEST_CASE("Checkpointing Benchmark") {
    // Micro-benchmark for overhead
    lfg_backend_init();
    std::string model_path = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
    std::ifstream f(model_path);
    if (!f.good()) return;

    lfg_model_load_config load_config = lfg_model_load_default_config();
    load_config.model_path = model_path.c_str();
    lfg_model* model = lfg_load_model(&load_config);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    lfg_session *session = lfg_session_create(model, &config);

    // Fill up some context (e.g. 100 tokens)
    std::vector<lfg_token> prompt(100, 15043); // 100 "Hello"s
    lfg_session_ingest_tokens(session, prompt.data(), prompt.size(), true);

    // Benchmark CreateCheckpoint
    auto start_create = std::chrono::high_resolution_clock::now();
    lfg_checkpoint *cp = lfg_session_create_checkpoint(session);
    auto end_create = std::chrono::high_resolution_clock::now();
    auto duration_create = std::chrono::duration_cast<std::chrono::microseconds>(end_create - start_create).count();

    MESSAGE("CreateCheckpoint overhead (100 tokens): " << duration_create << " us");
    CHECK(duration_create < 1000); // Should be < 1ms (likely < 10us)

    // Benchmark RestoreCheckpoint
    auto start_restore = std::chrono::high_resolution_clock::now();
    lfg_session_restore_checkpoint(session, cp);
    auto end_restore = std::chrono::high_resolution_clock::now();
    auto duration_restore = std::chrono::duration_cast<std::chrono::microseconds>(end_restore - start_restore).count();

    MESSAGE("RestoreCheckpoint overhead (100 tokens, truncate+re-ingest): " << duration_restore << " us");
    CHECK(duration_restore < 5000); // Strict: < 5ms.

    lfg_checkpoint_free(cp);
    lfg_session_free(session);
    lfg_model_free(model);
}
