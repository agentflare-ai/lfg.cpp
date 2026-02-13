#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfg_api.h"

#include <cstring>
#include <string>
#include <vector>
#include <cstdlib>

// Test model path from environment
static const char * get_model_path() {
    const char *p = std::getenv("LFG_MODEL_PATH");
    return p ? p : "models/lfm-2.5-1.2b-thinking.gguf";
}

TEST_CASE("Grammar forces EOS after valid JSON") {
    lfg_model_load_config lcfg = lfg_model_load_default_config();
    lcfg.model_path = get_model_path();
    lfg_model *model = lfg_load_model(&lcfg);
    if (!model) {
        MESSAGE("Skipping: model not available at ", lcfg.model_path);
        return;
    }

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = 512;
    cfg.sampling.temp = 0.0f; // greedy — deterministic
    lfg_session *session = lfg_session_create(model, &cfg);
    REQUIRE(session != nullptr);

    // Simple JSON schema: {"result": "<string>"}
    const char *schema = R"({"type":"object","properties":{"result":{"type":"string"}},"required":["result"],"additionalProperties":false})";
    bool ok = lfg_session_configure_structured(session, schema, "root");
    REQUIRE(ok);

    // Tokenize a simple prompt
    const char *prompt = "What is 2+2? Answer in JSON.";
    const lfg_vocab *vocab = lfg_model_get_vocab(model);
    std::vector<lfg_token> tokens(256);
    int n = lfg_tokenize(vocab, prompt, (int)strlen(prompt), tokens.data(), (int)tokens.size(), true, true);
    REQUIRE(n > 0);
    tokens.resize(n);

    ok = lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), false);
    REQUIRE(ok);

    // Generate up to 200 tokens — should stop well before that
    std::string output;
    int gen_count = 0;

    for (int i = 0; i < 200; ++i) {
        lfg_token tok = lfg_session_sample(session);
        if (lfg_vocab_is_eog(vocab, tok)) break;
        gen_count++;

        char piece_buf[64];
        int piece_len = lfg_token_to_piece(vocab, tok, piece_buf, sizeof(piece_buf), 0, true);
        if (piece_len > 0) {
            output.append(piece_buf, piece_len);
        }

        ok = lfg_session_ingest_tokens(session, &tok, 1, true);
        REQUIRE(ok);
    }

    MESSAGE("Generated (", gen_count, " tokens): ", output);

    // Should have hit EOS, not just exhausted 200 tokens
    CHECK(gen_count < 100);

    // Output should be valid JSON (starts with { and ends with })
    // Trim any whitespace
    size_t start = output.find('{');
    size_t end = output.rfind('}');
    CHECK(start != std::string::npos);
    CHECK(end != std::string::npos);

    // Should NOT have trailing whitespace after the closing brace
    if (end != std::string::npos && end + 1 < output.size()) {
        std::string trailing = output.substr(end + 1);
        // At most 1-2 chars of trailing content (from the token that includes })
        CHECK(trailing.size() <= 2);
    }

    lfg_session_free(session);
    lfg_model_free(model);
}

TEST_CASE("Max tokens enforcement") {
    lfg_model_load_config lcfg = lfg_model_load_default_config();
    lcfg.model_path = get_model_path();
    lfg_model *model = lfg_load_model(&lcfg);
    if (!model) {
        MESSAGE("Skipping: model not available at ", lcfg.model_path);
        return;
    }

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = 256;
    cfg.max_tokens = 5;
    cfg.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &cfg);
    REQUIRE(session != nullptr);

    const char *prompt = "Count from one to one hundred:";
    const lfg_vocab *vocab = lfg_model_get_vocab(model);
    std::vector<lfg_token> tokens(128);
    int n = lfg_tokenize(vocab, prompt, (int)strlen(prompt), tokens.data(), (int)tokens.size(), true, true);
    REQUIRE(n > 0);
    tokens.resize(n);

    bool ok = lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true);
    REQUIRE(ok);

    int gen_count = 0;
    for (int i = 0; i < 50; ++i) {
        lfg_token tok = lfg_session_sample(session);
        if (lfg_vocab_is_eog(vocab, tok)) break;
        gen_count++;
        ok = lfg_session_ingest_tokens(session, &tok, 1, true);
        REQUIRE(ok);
    }

    // Should stop at exactly max_tokens
    CHECK(gen_count == 5);

    lfg_session_free(session);
    lfg_model_free(model);
}
