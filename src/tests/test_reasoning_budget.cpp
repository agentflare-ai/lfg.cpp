#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/lfg_api.h"
#include <spdlog/spdlog.h>
#include <vector>
#include <string>
#include <fstream>

// Helper to detokenize a single token
std::string token_to_str(const lfg_vocab* vocab, lfg_token token) {
    char buf[256];
    int n = lfg_detokenize(const_cast<lfg_vocab*>(vocab), &token, 1, buf, sizeof(buf), false, false);
    if (n < 0) return "";
    return std::string(buf, n);
}

TEST_CASE("Reasoning Budget Enforcement") {
    lfg_backend_init();

    // 1. Load Model
    // We reuse the thinking model from other tests
    std::string model_path = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
    std::ifstream f(model_path);
    if (!f.good()) {
        MESSAGE("Skipping test: Model not found at " << model_path);
        return;
    }

    lfg_model_load_config load_config = lfg_model_load_default_config();
    load_config.model_path = model_path.c_str();
    load_config.n_gpu_layers = 0; // CPU for deterministic behavior

    lfg_model* model = lfg_load_model(&load_config);
    REQUIRE(model != nullptr);

    // 2. Configure session with Budget
    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f; // Deterministic
    config.reasoning_budget = 5; // Very short budget to force early exit
    config.enable_healing = true;

    lfg_session *session = lfg_session_create(model, &config);

    // 3. Configure Reasoning Tokens
    // <think> = 32001, </think> = 32002
    std::vector<lfg_token> start_tokens = {32001};
    std::vector<lfg_token> end_tokens = {32002};
    lfg_session_configure_reasoning(session, start_tokens.data(), start_tokens.size(), end_tokens.data(), end_tokens.size());

    // 4. Ingest Start Token to enter reasoning mode
    lfg_session_ingest_tokens(session, start_tokens.data(), start_tokens.size(), true);

    // 5. Generate and Monitor
    int tokens_generated = 0;
    bool exited_reasoning = false;

    // We expect the model to generate 5 tokens of "thought", then forced </think>
    // So total loop should be around 6-7 iterations
    for(int i=0; i<20; ++i) {
        if (!lfg_session_decode(session)) break;
        lfg_token t = lfg_session_sample(session);

        // Ingest back
        lfg_session_ingest_tokens(session, &t, 1, true);

        const auto* vocab = lfg_model_get_vocab(model);
        MESSAGE("Token " << i << ": " << t << " (" << token_to_str(vocab, t) << ")");

        if (t == end_tokens[0]) {
            exited_reasoning = true;
            break;
        }
        tokens_generated++;
    }

    CHECK(exited_reasoning);
    CHECK(tokens_generated >= config.reasoning_budget);
    // Ideally it should be exactly budget, or budget+1 depending on when check happens
    // Our logic checks: if count >= budget, next token is forced.
    // So if budget is 5:
    // 0: gen (count=1)
    // 1: gen (count=2)
    // 2: gen (count=3)
    // 3: gen (count=4)
    // 4: gen (count=5) -> Limit reached
    // 5: Force </think>
    // So tokens_generated should be 5.
    CHECK(tokens_generated == config.reasoning_budget);

    lfg_session_free(session);
    lfg_model_free(model);
}
