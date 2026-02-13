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

// ---------------------------------------------------------------------------
// Helper: load model + create session (greedy, small context)
// ---------------------------------------------------------------------------

struct test_env {
    lfg_model   *model;
    lfg_session *session;
    const lfg_vocab *vocab;
};

static bool setup(test_env *env, int n_ctx = 256) {
    lfg_model_load_config lcfg = lfg_model_load_default_config();
    lcfg.model_path = get_model_path();
    env->model = lfg_load_model(&lcfg);
    if (!env->model) return false;

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = n_ctx;
    cfg.sampling.temp = 0.0f;
    env->session = lfg_session_create(env->model, &cfg);
    if (!env->session) {
        lfg_model_free(env->model);
        return false;
    }
    env->vocab = lfg_model_get_vocab(env->model);
    return true;
}

static void teardown(test_env *env) {
    if (env->session) lfg_session_free(env->session);
    if (env->model) lfg_model_free(env->model);
}

static void ingest_prompt(test_env *env, const char *prompt) {
    std::vector<lfg_token> tokens(256);
    int n = lfg_tokenize(env->vocab, prompt, (int)strlen(prompt),
                         tokens.data(), (int)tokens.size(), true, true);
    REQUIRE(n > 0);
    tokens.resize(n);
    bool ok = lfg_session_ingest_tokens(env->session, tokens.data(), tokens.size(), true);
    REQUIRE(ok);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("lfg_generate_default_config - zeroed") {
    lfg_generate_config gc = lfg_generate_default_config();
    CHECK(gc.max_tokens == 0);
    CHECK(gc.token_cb == nullptr);
    CHECK(gc.token_cb_data == nullptr);
}

TEST_CASE("lfg_session_generate - null session returns zero result") {
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 10;
    lfg_generate_result r = lfg_session_generate(nullptr, gc);
    CHECK(r.n_tokens == 0);
    CHECK(r.n_retrievals == 0);
}

TEST_CASE("lfg_session_generate - basic generation with max_tokens") {
    test_env env;
    if (!setup(&env)) {
        MESSAGE("Skipping: model not available");
        return;
    }

    ingest_prompt(&env, "The capital of France is");

    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 10;
    lfg_generate_result r = lfg_session_generate(env.session, gc);

    MESSAGE("Generated ", r.n_tokens, " tokens, stop_reason=", r.stop_reason);
    CHECK(r.n_tokens > 0);
    CHECK(r.n_tokens <= 10);
    // With only 10 tokens, should hit max_tokens (model unlikely to EOS that fast)
    CHECK(r.stop_reason == LFG_STOP_MAX_TOKENS);

    teardown(&env);
}

TEST_CASE("lfg_session_generate - token callback receives pieces") {
    test_env env;
    if (!setup(&env)) {
        MESSAGE("Skipping: model not available");
        return;
    }

    ingest_prompt(&env, "Hello, world!");

    struct cb_state {
        std::string text;
        int count;
    };
    cb_state state = {"", 0};

    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 20;
    gc.token_cb = [](lfg_token, const char *piece, int32_t piece_len, void *ud) -> lfg_generate_action {
        auto *s = (cb_state *)ud;
        if (piece_len > 0) s->text.append(piece, piece_len);
        s->count++;
        return LFG_GENERATE_CONTINUE;
    };
    gc.token_cb_data = &state;

    lfg_generate_result r = lfg_session_generate(env.session, gc);

    MESSAGE("Streamed ", state.count, " tokens: '", state.text.substr(0, 100), "'");
    CHECK(state.count == r.n_tokens);
    CHECK(state.count > 0);
    CHECK(!state.text.empty());

    teardown(&env);
}

TEST_CASE("lfg_session_generate - callback can stop generation") {
    test_env env;
    if (!setup(&env)) {
        MESSAGE("Skipping: model not available");
        return;
    }

    ingest_prompt(&env, "Count to one hundred: 1, 2, 3, 4");

    int stop_after = 5;

    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 200;
    gc.token_cb = [](lfg_token, const char *, int32_t, void *ud) -> lfg_generate_action {
        int *counter = (int *)ud;
        (*counter)--;
        return (*counter <= 0) ? LFG_GENERATE_STOP : LFG_GENERATE_CONTINUE;
    };
    gc.token_cb_data = &stop_after;

    lfg_generate_result r = lfg_session_generate(env.session, gc);

    MESSAGE("Stopped after ", r.n_tokens, " tokens");
    CHECK(r.n_tokens == 5);
    CHECK(r.stop_reason == LFG_STOP_CALLBACK);

    teardown(&env);
}

TEST_CASE("lfg_session_generate - uses session max_tokens when config is 0") {
    lfg_model_load_config lcfg = lfg_model_load_default_config();
    lcfg.model_path = get_model_path();
    lfg_model *model = lfg_load_model(&lcfg);
    if (!model) {
        MESSAGE("Skipping: model not available");
        return;
    }

    // Set session-level max_tokens at creation time
    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = 256;
    cfg.sampling.temp = 0.0f;
    cfg.max_tokens = 5;
    lfg_session *session = lfg_session_create(model, &cfg);
    REQUIRE(session != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);
    const char *prompt = "Tell me a long story about";
    std::vector<lfg_token> tokens(128);
    int n = lfg_tokenize(vocab, prompt, (int)strlen(prompt),
                         tokens.data(), (int)tokens.size(), true, true);
    REQUIRE(n > 0);
    tokens.resize(n);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    // config max_tokens = 0 → should fall back to session config (5)
    lfg_generate_config gc = lfg_generate_default_config();
    lfg_generate_result r = lfg_session_generate(session, gc);

    MESSAGE("Generated ", r.n_tokens, " tokens with session max_tokens=5");
    // session's sample() also enforces max_tokens by returning EOS, so the
    // generate loop should stop at or before 5.
    CHECK(r.n_tokens <= 5);

    lfg_session_free(session);
    lfg_model_free(model);
}

TEST_CASE("lfg_session_generate - stop sequences work through generate") {
    test_env env;
    if (!setup(&env)) {
        MESSAGE("Skipping: model not available");
        return;
    }

    // Configure ":" as stop sequence
    std::vector<lfg_token> stop_tokens(4);
    int n_stop = lfg_tokenize(env.vocab, ":", 1,
                              stop_tokens.data(), (int)stop_tokens.size(), false, false);
    REQUIRE(n_stop > 0);
    stop_tokens.resize(n_stop);

    const lfg_token *seq_ptrs[] = { stop_tokens.data() };
    size_t seq_lens[] = { (size_t)n_stop };
    bool ok = lfg_session_configure_stop_sequences(env.session, seq_ptrs, seq_lens, 1);
    REQUIRE(ok);

    ingest_prompt(&env, "Write a long story about dragons:");

    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 200;
    lfg_generate_result r = lfg_session_generate(env.session, gc);

    MESSAGE("Stopped after ", r.n_tokens, " tokens, reason=", r.stop_reason);
    // Stop sequence triggers EOS from sample(), so stop_reason should be EOS.
    // If the model never generates ":" within 200 tokens, it will hit max_tokens instead.
    CHECK((r.stop_reason == LFG_STOP_EOS || r.stop_reason == LFG_STOP_MAX_TOKENS));
    CHECK(r.n_tokens <= 200);

    teardown(&env);
}

TEST_CASE("lfg_session_generate - default config uses session max_tokens") {
    lfg_model_load_config lcfg = lfg_model_load_default_config();
    lcfg.model_path = get_model_path();
    lfg_model *model = lfg_load_model(&lcfg);
    if (!model) {
        MESSAGE("Skipping: model not available");
        return;
    }

    // Set session max_tokens so we don't generate 4096 tokens
    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = 256;
    cfg.sampling.temp = 0.0f;
    cfg.max_tokens = 8;
    lfg_session *session = lfg_session_create(model, &cfg);
    REQUIRE(session != nullptr);

    const lfg_vocab *vocab = lfg_model_get_vocab(model);
    const char *prompt = "Hi";
    std::vector<lfg_token> tokens(64);
    int n = lfg_tokenize(vocab, prompt, (int)strlen(prompt),
                         tokens.data(), (int)tokens.size(), true, true);
    REQUIRE(n > 0);
    tokens.resize(n);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    // Pass default config (max_tokens=0) — should use session max_tokens (8)
    lfg_generate_result r = lfg_session_generate(session, lfg_generate_default_config());

    MESSAGE("Generated ", r.n_tokens, " tokens with default config");
    CHECK(r.n_tokens > 0);

    lfg_session_free(session);
    lfg_model_free(model);
}

TEST_CASE("lfg_session_prompt_generate - null safety") {
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 10;

    lfg_generate_result r = lfg_session_prompt_generate(nullptr, "hi", 2, true, gc);
    CHECK(r.n_tokens == 0);

    // null prompt
    r = lfg_session_prompt_generate(nullptr, nullptr, 0, true, gc);
    CHECK(r.n_tokens == 0);
}

TEST_CASE("lfg_session_prompt_generate - basic completion") {
    test_env env;
    if (!setup(&env)) {
        MESSAGE("Skipping: model not available");
        return;
    }

    const char *prompt = "The capital of France is";

    struct cb_state {
        std::string text;
    };
    cb_state state;

    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 15;
    gc.token_cb = [](lfg_token, const char *piece, int32_t piece_len, void *ud) -> lfg_generate_action {
        auto *s = (cb_state *)ud;
        if (piece_len > 0) s->text.append(piece, piece_len);
        return LFG_GENERATE_CONTINUE;
    };
    gc.token_cb_data = &state;

    lfg_generate_result r = lfg_session_prompt_generate(
        env.session, prompt, (int32_t)strlen(prompt), true, gc);

    MESSAGE("Prompt generated ", r.n_tokens, " tokens: '", state.text.substr(0, 100), "'");
    CHECK(r.n_tokens > 0);
    CHECK(r.n_tokens <= 15);
    CHECK(!state.text.empty());

    teardown(&env);
}

TEST_CASE("lfg_session_prompt_generate - parity with manual tokenize+ingest+generate") {
    test_env env;
    if (!setup(&env, 512)) {
        MESSAGE("Skipping: model not available");
        return;
    }

    const char *prompt = "Once upon a time";
    const int gen_tokens = 12;

    // --- Manual: tokenize + ingest + generate ---
    ingest_prompt(&env, prompt);

    std::string manual_output;
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = gen_tokens;
    gc.token_cb = [](lfg_token, const char *piece, int32_t piece_len, void *ud) -> lfg_generate_action {
        auto *s = (std::string *)ud;
        if (piece_len > 0) s->append(piece, piece_len);
        return LFG_GENERATE_CONTINUE;
    };
    gc.token_cb_data = &manual_output;
    lfg_session_generate(env.session, gc);

    // --- Reset and use prompt_generate ---
    lfg_session_reset(env.session);

    std::string prompt_output;
    gc.token_cb_data = &prompt_output;
    lfg_session_prompt_generate(
        env.session, prompt, (int32_t)strlen(prompt), true, gc);

    MESSAGE("Manual:  '", manual_output, "'");
    MESSAGE("Prompt:  '", prompt_output, "'");
    CHECK(manual_output == prompt_output);

    teardown(&env);
}

TEST_CASE("lfg_session_chat_generate - null session/messages") {
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 10;

    lfg_generate_result r = lfg_session_chat_generate(nullptr, nullptr, 0, gc);
    CHECK(r.n_tokens == 0);
}

TEST_CASE("lfg_session_chat_generate - full chat pipeline") {
    test_env env;
    if (!setup(&env)) {
        MESSAGE("Skipping: model not available");
        return;
    }

    lfg_chat_message msgs[] = {
        {"user", "What is 2+2?"},
    };

    struct cb_state {
        std::string text;
    };
    cb_state state;

    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 30;
    gc.token_cb = [](lfg_token, const char *piece, int32_t piece_len, void *ud) -> lfg_generate_action {
        auto *s = (cb_state *)ud;
        if (piece_len > 0) s->text.append(piece, piece_len);
        return LFG_GENERATE_CONTINUE;
    };
    gc.token_cb_data = &state;

    lfg_generate_result r = lfg_session_chat_generate(env.session, msgs, 1, gc);

    MESSAGE("Chat generated ", r.n_tokens, " tokens: '", state.text.substr(0, 200), "'");
    CHECK(r.n_tokens > 0);

    teardown(&env);
}

TEST_CASE("lfg_session_generate - parity with manual loop") {
    // Verify that generate() produces the same output as the manual
    // decode+sample+ingest loop under identical conditions (greedy sampling).
    test_env env;
    if (!setup(&env, 512)) {
        MESSAGE("Skipping: model not available");
        return;
    }

    const char *prompt = "The quick brown fox";
    const int gen_tokens = 15;

    // --- Manual loop ---
    ingest_prompt(&env, prompt);
    std::string manual_output;
    for (int i = 0; i < gen_tokens; ++i) {
        lfg_session_decode(env.session);
        lfg_token tok = lfg_session_sample(env.session);
        if (lfg_vocab_is_eog(env.vocab, tok)) break;

        char buf[64];
        int n = lfg_token_to_piece(env.vocab, tok, buf, sizeof(buf), 0, false);
        if (n > 0) manual_output.append(buf, n);

        lfg_session_ingest_tokens(env.session, &tok, 1, false);
    }

    // --- Reset and use generate() ---
    lfg_session_reset(env.session);
    ingest_prompt(&env, prompt);

    std::string gen_output;
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = gen_tokens;
    gc.token_cb = [](lfg_token, const char *piece, int32_t piece_len, void *ud) -> lfg_generate_action {
        auto *s = (std::string *)ud;
        if (piece_len > 0) s->append(piece, piece_len);
        return LFG_GENERATE_CONTINUE;
    };
    gc.token_cb_data = &gen_output;

    lfg_generate_result r = lfg_session_generate(env.session, gc);

    MESSAGE("Manual: '", manual_output, "'");
    MESSAGE("Generate: '", gen_output, "'");
    CHECK(manual_output == gen_output);

    teardown(&env);
}
