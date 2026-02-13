#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfg_api.h"

#include <fstream>
#include <vector>

static const char *MODEL_PATH = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";

static lfg_model *load_model() {
    std::ifstream f(MODEL_PATH);
    if (!f.good()) return nullptr;

    lfg_model_load_config cfg = lfg_model_load_default_config();
    cfg.model_path = MODEL_PATH;
    cfg.n_gpu_layers = 0;
    return lfg_load_model(&cfg);
}

TEST_CASE("max_tokens does not interrupt forced reasoning end sequence") {
    lfg_backend_init();
    lfg_model *model = load_model();
    if (!model) { MESSAGE("Skipping: model not found at ", MODEL_PATH); return; }

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = 1024;
    cfg.sampling.temp = 0.0f;
    cfg.reasoning_budget = 3;  // Very short — forces end quickly
    cfg.max_tokens = 4;        // Deliberately low so it would conflict with forced end
    lfg_session *session = lfg_session_create(model, &cfg);
    REQUIRE(session != nullptr);

    // Configure reasoning tokens (<think>=32001, </think>=32002)
    lfg_token start_tok = 32001;
    lfg_token end_tok = 32002;
    lfg_session_configure_reasoning(session, &start_tok, 1, &end_tok, 1);

    // Ingest BOS + <think>
    lfg_token bos = 1;
    lfg_session_ingest_tokens(session, &bos, 1, true);
    lfg_session_ingest_tokens(session, &start_tok, 1, true);

    const auto *vocab = lfg_model_get_vocab(model);
    lfg_token eos = lfg_vocab_eos(vocab);

    bool saw_end_token = false;
    std::vector<lfg_token> tokens;

    for (int i = 0; i < 20; ++i) {
        lfg_session_decode(session);
        lfg_token t = lfg_session_sample(session);
        tokens.push_back(t);

        if (t == end_tok) {
            saw_end_token = true;
            break;
        }
        if (t == eos) break;
        lfg_session_ingest_tokens(session, &t, 1, true);
    }

    // The forced </think> end token must appear — max_tokens should not have
    // cut it short mid-sequence.
    CHECK(saw_end_token);
}

TEST_CASE("After forced reasoning end, max_tokens takes effect") {
    lfg_backend_init();
    lfg_model *model = load_model();
    if (!model) { MESSAGE("Skipping: model not found at ", MODEL_PATH); return; }

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = 1024;
    cfg.sampling.temp = 0.0f;
    cfg.reasoning_budget = 2;
    cfg.max_tokens = 5; // budget(2) + end_token(1) + at most 2 more after reasoning
    lfg_session *session = lfg_session_create(model, &cfg);
    REQUIRE(session != nullptr);

    lfg_token start_tok = 32001;
    lfg_token end_tok = 32002;
    lfg_session_configure_reasoning(session, &start_tok, 1, &end_tok, 1);

    lfg_token bos = 1;
    lfg_session_ingest_tokens(session, &bos, 1, true);
    lfg_session_ingest_tokens(session, &start_tok, 1, true);

    const auto *vocab = lfg_model_get_vocab(model);
    lfg_token eos = lfg_vocab_eos(vocab);

    int total_sampled = 0;
    for (int i = 0; i < 30; ++i) {
        lfg_session_decode(session);
        lfg_token t = lfg_session_sample(session);
        total_sampled++;
        if (t == eos) break;
        lfg_session_ingest_tokens(session, &t, 1, true);
    }

    // max_tokens should eventually stop generation
    // Total should be bounded by max_tokens + forced end tokens
    CHECK(total_sampled <= cfg.max_tokens + 2); // +2 for forced end sequence slack
}

TEST_CASE("max_tokens=0 (unlimited) with reasoning budget still works") {
    lfg_backend_init();
    lfg_model *model = load_model();
    if (!model) { MESSAGE("Skipping: model not found at ", MODEL_PATH); return; }

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = 1024;
    cfg.sampling.temp = 0.0f;
    cfg.reasoning_budget = 3;
    cfg.max_tokens = 0; // Unlimited
    lfg_session *session = lfg_session_create(model, &cfg);
    REQUIRE(session != nullptr);

    lfg_token start_tok = 32001;
    lfg_token end_tok = 32002;
    lfg_session_configure_reasoning(session, &start_tok, 1, &end_tok, 1);

    lfg_token bos = 1;
    lfg_session_ingest_tokens(session, &bos, 1, true);
    lfg_session_ingest_tokens(session, &start_tok, 1, true);

    bool saw_end_token = false;
    for (int i = 0; i < 20; ++i) {
        lfg_session_decode(session);
        lfg_token t = lfg_session_sample(session);

        if (t == end_tok) {
            saw_end_token = true;
            break;
        }
        lfg_session_ingest_tokens(session, &t, 1, true);
    }

    // Reasoning budget should still force the end token
    CHECK(saw_end_token);

    lfg_session_free(session);
    lfg_model_free(model);
}
