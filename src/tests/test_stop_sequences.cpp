#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfg_api.h"

#include <cstring>
#include <string>
#include <vector>
#include <cstdlib>

static const char * get_model_path() {
    const char *p = std::getenv("LFG_MODEL_PATH");
    return p ? p : "models/lfm-2.5-1.2b-thinking.gguf";
}

TEST_CASE("Stop sequence - single token colon") {
    lfg_model_load_config lcfg = lfg_model_load_default_config();
    lcfg.model_path = get_model_path();
    lfg_model *model = lfg_load_model(&lcfg);
    if (!model) {
        MESSAGE("Skipping: model not available at ", lcfg.model_path);
        return;
    }

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = 256;
    cfg.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &cfg);
    REQUIRE(session != nullptr);

    // Use colon ":" as stop token — appears early in most structured outputs
    const lfg_vocab *vocab = lfg_model_get_vocab(model);
    std::vector<lfg_token> stop_tokens(4);
    int n_stop = lfg_tokenize(vocab, ":", 1, stop_tokens.data(), (int)stop_tokens.size(), false, false);
    REQUIRE(n_stop > 0);
    stop_tokens.resize(n_stop);

    const lfg_token *seq_ptrs[] = { stop_tokens.data() };
    size_t seq_lens[] = { (size_t)n_stop };
    bool ok = lfg_session_configure_stop_sequences(session, seq_ptrs, seq_lens, 1);
    REQUIRE(ok);

    const char *prompt = "Write a very long story about dragons:";
    std::vector<lfg_token> tokens(128);
    int n = lfg_tokenize(vocab, prompt, (int)strlen(prompt), tokens.data(), (int)tokens.size(), true, true);
    REQUIRE(n > 0);
    tokens.resize(n);

    ok = lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true);
    REQUIRE(ok);

    // Generate — should stop when ":" appears
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

    MESSAGE("Stopped after ", gen_count, " tokens: ", output.substr(0, 200));
    // Should stop quickly — ":" appears in most generations within first 50 tokens
    CHECK(gen_count < 100);

    lfg_session_free(session);
    lfg_model_free(model);
}

TEST_CASE("Stop sequence - max_tokens combined with stop") {
    // Test that max_tokens and stop sequences work together
    lfg_model_load_config lcfg = lfg_model_load_default_config();
    lcfg.model_path = get_model_path();
    lfg_model *model = lfg_load_model(&lcfg);
    if (!model) {
        MESSAGE("Skipping: model not available at ", lcfg.model_path);
        return;
    }

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = 256;
    cfg.max_tokens = 3;
    cfg.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &cfg);
    REQUIRE(session != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    // Set a stop sequence that won't match — max_tokens should trigger first
    lfg_token stop_tok = 999999; // unlikely token
    const lfg_token *seq_ptrs[] = { &stop_tok };
    size_t seq_lens[] = { 1 };
    bool ok = lfg_session_configure_stop_sequences(session, seq_ptrs, seq_lens, 1);
    REQUIRE(ok);

    const char *prompt = "Hello";
    std::vector<lfg_token> tokens(64);
    int n = lfg_tokenize(vocab, prompt, (int)strlen(prompt), tokens.data(), (int)tokens.size(), true, true);
    REQUIRE(n > 0);
    tokens.resize(n);

    ok = lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true);
    REQUIRE(ok);

    int gen_count = 0;
    for (int i = 0; i < 50; ++i) {
        lfg_token tok = lfg_session_sample(session);
        if (lfg_vocab_is_eog(vocab, tok)) break;
        gen_count++;
        ok = lfg_session_ingest_tokens(session, &tok, 1, true);
        REQUIRE(ok);
    }

    CHECK(gen_count == 3);

    lfg_session_free(session);
    lfg_model_free(model);
}

TEST_CASE("Stop sequence - clear sequences") {
    lfg_model_load_config lcfg = lfg_model_load_default_config();
    lcfg.model_path = get_model_path();
    lfg_model *model = lfg_load_model(&lcfg);
    if (!model) {
        MESSAGE("Skipping: model not available at ", lcfg.model_path);
        return;
    }

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = 128;
    lfg_session *session = lfg_session_create(model, &cfg);
    REQUIRE(session != nullptr);

    // Configure and then clear
    const lfg_vocab *vocab = lfg_model_get_vocab(model);
    lfg_token stop_tok = lfg_vocab_eos(vocab);
    const lfg_token *seq_ptrs[] = { &stop_tok };
    size_t seq_lens[] = { 1 };
    bool ok = lfg_session_configure_stop_sequences(session, seq_ptrs, seq_lens, 1);
    CHECK(ok);

    // Clear by passing 0 sequences
    ok = lfg_session_configure_stop_sequences(session, nullptr, nullptr, 0);
    CHECK(ok);

    lfg_session_free(session);
    lfg_model_free(model);
}

TEST_CASE("Stop sequence - null session returns false") {
    bool ok = lfg_session_configure_stop_sequences(nullptr, nullptr, nullptr, 0);
    CHECK(!ok);
}
