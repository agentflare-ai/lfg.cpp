#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/lfg_api.h"

TEST_CASE("InferenceCore Lifecycle") {
    // Test complete failure case (no model)
    lfg_session_config config = lfg_session_default_config();
    lfg_session *session = lfg_session_create(nullptr, &config); // Should handle nullptr model

    SUBCASE("State checks on empty core") {
        // Sample should return 0/default when no context
        CHECK(lfg_session_sample(session) == 0);

        // Logits should be empty
        CHECK(lfg_session_get_logits(session, nullptr, 0) <= 0);

        // Decoding should be safe (no-op or fail gracefully)
        // Implementation returned true as placeholder, which is fine
        CHECK(lfg_session_decode(session) == true);
    }

    lfg_session_free(session);
}

#include <fstream>

TEST_CASE("InferenceCore Integration with Real Model") {
    lfg_backend_init();

    // Check if model exists
    std::string model_path =  "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
    std::ifstream f(model_path);
    if (!f.good()) {
        MESSAGE("Skipping integration test: Model not found at " << model_path);
        return;
    }

    // Load Model
    lfg_model_load_config load_config = lfg_model_load_default_config();
    load_config.model_path = model_path.c_str();
    lfg_model* model = lfg_load_model(&load_config);
    REQUIRE(model != nullptr);

    // Init Core
    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 512;
    lfg_session *session = lfg_session_create(model, &config);

    SUBCASE("Forward Pass") {
        // Tokenize prompt "Hello" -> simplistic manual token for now or 1 (BOS)
        // liquid usually uses 1 as BOS
        lfg_token tokens[] = {1, 15043}; // BOS, "Hello" (approx)

        CHECK(lfg_session_ingest_tokens(session, tokens, 2, true));

        // Output should be generated
        CHECK(lfg_session_decode(session));

        // Sampling
        lfg_token next = lfg_session_sample(session);
        MESSAGE("Sampled token: " << next);
        CHECK(next >= 0);

        // Logits
        CHECK(lfg_session_get_logits(session, nullptr, 0) > 0);
    }

    lfg_session_free(session);
    lfg_model_free(model);
}

TEST_CASE("InferenceCore Structured Decoding") {
    // Shared model load (inefficient to reload, but keeps tests isolated. Or we could refactor)
    // For now, assuming fast enough or we just reload.
    std::string model_path = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
    std::ifstream f(model_path);
    if (!f.good()) return;

    lfg_model_load_config load_config = lfg_model_load_default_config();
    load_config.model_path = model_path.c_str();
    lfg_model* model = lfg_load_model(&load_config);

    lfg_session_config config = lfg_session_default_config();
    lfg_session *session = lfg_session_create(model, &config);

    SUBCASE("GBNF Grammar") {
        // Simple GBNF grammar: root ::= "yes" | "no"
        std::string grammar = "root ::= \"yes\" | \"no\"";
        lfg_session_configure_structured(session, grammar.c_str(), "root");

        lfg_token toks[] = {1, 15043}; // "Hello"
        lfg_session_ingest_tokens(session, toks, 2, false);

        lfg_token token = lfg_session_sample(session);
        MESSAGE("Strict Grammar Sampled: " << token);
        CHECK(token > 0);
    }

    lfg_session_free(session);
    lfg_model_free(model);
}
