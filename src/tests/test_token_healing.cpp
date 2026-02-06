#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/lfg_api.h"
#include <spdlog/spdlog.h>

#include <chrono>

TEST_CASE("Token Healing") {
    lfg_backend_init();

    std::string model_path = "models/lfm2-350M.gguf";
    lfg_model_load_config load_config = lfg_model_load_default_config();
    load_config.model_path = model_path.c_str();
    load_config.n_gpu_layers = 0;

    lfg_model* model = lfg_load_model(&load_config);
    if (!model) {
        MESSAGE("Model not found, skipping test.");
        return;
    }

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f; // Deterministic
    config.enable_healing = true; // Enable optimization
    lfg_session *session = lfg_session_create(model, &config);

    SUBCASE("Benchmark: TPS Comparison") {
        std::string prompt = "The quick brown fox jumps over the lazy";
        const auto* vocab = lfg_model_get_vocab(model);
        std::vector<lfg_token> tokens(100);
        int n_tokens = lfg_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, false);
        tokens.resize(n_tokens);

        auto benchmark_run = [&](bool healing_enabled) -> double {
            lfg_session_reset(session);
            // Hack to toggle healing without public API access to config
            // We assume 'session' was initialized with healing=true in this TEST_CASE context from previous turns?
            // But we can't change it after construction.
            // So we need to construct a NEW session for each run.

            lfg_session_config cfg = lfg_session_default_config();
            cfg.n_ctx = 2048;
            cfg.enable_healing = healing_enabled;
            lfg_session *local_session = lfg_session_create(model, &cfg);

            auto t0 = std::chrono::high_resolution_clock::now();

            // 1. Ingest
            lfg_session_ingest_tokens(local_session, tokens.data(), tokens.size(), true);

            // 2. Heal (if enabled)
            if (healing_enabled) {
                lfg_session_heal_last_token(local_session);
            }

            auto t1 = std::chrono::high_resolution_clock::now(); // TTFT point

            // 3. Generate 20 tokens
            for (int i = 0; i < 20; ++i) {
                lfg_session_decode(local_session);
                auto t = lfg_session_sample(local_session);
                lfg_session_ingest_tokens(local_session, &t, 1, true);
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

            lfg_session_free(local_session);
            return tps;
        };

        double tps_no_heal = benchmark_run(false);
        double tps_heal = benchmark_run(true);

        // Healing affects TTFT, but TPS should be roughly equal (decoding is identical)
        // Allowing 10% variance for noise
        CHECK(tps_heal >= tps_no_heal * 0.9);
    }

    lfg_session_free(session);
    lfg_model_free(model);
}
