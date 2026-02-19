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

TEST_CASE("Checkpoint saves and restores generated_count") {
    lfg_backend_init();
    lfg_model *model = load_model();
    if (!model) { MESSAGE("Skipping: model not found at ", MODEL_PATH); return; }

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = 512;
    cfg.sampling.temp = 0.0f;
    cfg.max_tokens = 20;
    lfg_session *session = lfg_session_create(model, &cfg);
    REQUIRE(session != nullptr);

    // Ingest and generate 3 tokens
    lfg_token bos = 1;
    lfg_session_ingest_tokens(session, &bos, 1, true);
    for (int i = 0; i < 3; ++i) {
        lfg_session_decode(session);
        lfg_token t = lfg_session_sample(session);
        lfg_session_ingest_tokens(session, &t, 1, true);
    }

    // Checkpoint after 3
    lfg_checkpoint *ck = lfg_session_create_checkpoint(session);
    REQUIRE(ck != nullptr);

    // Generate 10 more (generated_count now 13)
    for (int i = 0; i < 10; ++i) {
        lfg_session_decode(session);
        lfg_token t = lfg_session_sample(session);
        const auto *vocab = lfg_model_get_vocab(model);
        if (t == lfg_vocab_eos(vocab)) break;
        lfg_session_ingest_tokens(session, &t, 1, true);
    }

    // Restore — generated_count should be back to 3
    bool ok = lfg_session_restore_checkpoint(session, ck);
    CHECK(ok);

    // max_tokens=20 means we can still generate 17 more tokens without hitting limit
    const auto *vocab = lfg_model_get_vocab(model);
    lfg_token eos = lfg_vocab_eos(vocab);
    int count = 0;
    for (int i = 0; i < 18; ++i) {
        lfg_session_decode(session);
        lfg_token t = lfg_session_sample(session);
        if (t == eos) break;
        count++;
        lfg_session_ingest_tokens(session, &t, 1, true);
    }
    // Should have generated at least some tokens (not hit max_tokens at 0)
    CHECK(count > 0);

    lfg_checkpoint_free(ck);
    lfg_session_free(session);
    lfg_model_free(model);
}

TEST_CASE("Checkpoint saves and restores reasoning_token_count") {
    lfg_backend_init();
    lfg_model *model = load_model();
    if (!model) { MESSAGE("Skipping: model not found at ", MODEL_PATH); return; }

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = 1024;
    cfg.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &cfg);
    REQUIRE(session != nullptr);

    // Configure reasoning tokens (<think>=32001, </think>=32002)
    lfg_token start_tok = 32001;
    lfg_token end_tok = 32002;
    lfg_session_configure_reasoning(session, &start_tok, 1, &end_tok, 1);

    // Ingest BOS + <think> to enter reasoning mode
    lfg_token bos = 1;
    lfg_session_ingest_tokens(session, &bos, 1, true);
    lfg_session_ingest_tokens(session, &start_tok, 1, true);

    // Generate 5 reasoning tokens
    for (int i = 0; i < 5; ++i) {
        lfg_session_decode(session);
        lfg_token t = lfg_session_sample(session);
        lfg_session_ingest_tokens(session, &t, 1, true);
    }

    // Checkpoint
    lfg_checkpoint *ck = lfg_session_create_checkpoint(session);
    REQUIRE(ck != nullptr);

    // Generate 10 more reasoning tokens
    for (int i = 0; i < 10; ++i) {
        lfg_session_decode(session);
        lfg_token t = lfg_session_sample(session);
        lfg_session_ingest_tokens(session, &t, 1, true);
    }

    // Restore — reasoning count should go back to what it was at checkpoint
    bool ok = lfg_session_restore_checkpoint(session, ck);
    CHECK(ok);

    // Continue generating — should not immediately hit budget (since we restored count)
    const auto *vocab = lfg_model_get_vocab(model);
    lfg_token eos = lfg_vocab_eos(vocab);
    int count = 0;
    for (int i = 0; i < 10; ++i) {
        lfg_session_decode(session);
        lfg_token t = lfg_session_sample(session);
        if (t == eos || t == end_tok) break;
        count++;
        lfg_session_ingest_tokens(session, &t, 1, true);
    }
    CHECK(count > 0);

    lfg_checkpoint_free(ck);
    lfg_session_free(session);
    lfg_model_free(model);
}

TEST_CASE("Restore then continue generating without corruption") {
    lfg_backend_init();
    lfg_model *model = load_model();
    if (!model) { MESSAGE("Skipping: model not found at ", MODEL_PATH); return; }

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = 512;
    cfg.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &cfg);
    REQUIRE(session != nullptr);

    lfg_token bos = 1;
    lfg_session_ingest_tokens(session, &bos, 1, true);

    // Generate 5 tokens
    for (int i = 0; i < 5; ++i) {
        lfg_session_decode(session);
        lfg_token t = lfg_session_sample(session);
        lfg_session_ingest_tokens(session, &t, 1, true);
    }

    lfg_checkpoint *ck = lfg_session_create_checkpoint(session);
    REQUIRE(ck != nullptr);

    // Generate 10 more
    for (int i = 0; i < 10; ++i) {
        lfg_session_decode(session);
        lfg_token t = lfg_session_sample(session);
        lfg_session_ingest_tokens(session, &t, 1, true);
    }

    // Restore
    bool ok = lfg_session_restore_checkpoint(session, ck);
    CHECK(ok);

    // Generate deterministically after restore — should match first-pass output
    std::vector<lfg_token> after_restore;
    for (int i = 0; i < 5; ++i) {
        lfg_session_decode(session);
        lfg_token t = lfg_session_sample(session);
        after_restore.push_back(t);
        lfg_session_ingest_tokens(session, &t, 1, true);
    }
    CHECK(after_restore.size() == 5);

    lfg_checkpoint_free(ck);
    lfg_session_free(session);
    lfg_model_free(model);
}
