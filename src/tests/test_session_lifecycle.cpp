#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfg_api.h"

#include <fstream>
#include <string>
#include <vector>

static const char *MODEL_PATH = "models/lfm2-350M.gguf";

static lfg_model *load_model() {
    std::ifstream f(MODEL_PATH);
    if (!f.good()) return nullptr;

    lfg_model_load_config cfg = lfg_model_load_default_config();
    cfg.model_path = MODEL_PATH;
    cfg.n_gpu_layers = 0;
    return lfg_load_model(&cfg);
}

TEST_CASE("Reset and re-use with different grammars") {
    lfg_backend_init();
    lfg_model *model = load_model();
    if (!model) { MESSAGE("Skipping: model not found at ", MODEL_PATH); return; }

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = 256;
    cfg.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &cfg);
    REQUIRE(session != nullptr);

    // Grammar 1: simple JSON object
    const char *grammar1 = R"(root ::= "{" ws "\"value\"" ws ":" ws [0-9]+ ws "}"
ws ::= [ \t\n]*)";
    bool ok = lfg_session_configure_structured(session, grammar1, "root");
    CHECK(ok);

    // Ingest BOS and generate a few tokens
    lfg_token bos = 1;
    lfg_session_ingest_tokens(session, &bos, 1, true);
    std::vector<lfg_token> gen1;
    for (int i = 0; i < 10; ++i) {
        lfg_session_decode(session);
        lfg_token t = lfg_session_sample(session);
        const auto *vocab = lfg_model_get_vocab(model);
        if (t == lfg_vocab_eos(vocab)) break;
        gen1.push_back(t);
        lfg_session_ingest_tokens(session, &t, 1, true);
    }
    CHECK(gen1.size() > 0);

    // Reset
    lfg_session_reset(session);

    // Grammar 2: simple string
    const char *grammar2 = R"(root ::= "\"hello\"")";
    ok = lfg_session_configure_structured(session, grammar2, "root");
    CHECK(ok);

    lfg_session_ingest_tokens(session, &bos, 1, true);
    std::vector<lfg_token> gen2;
    for (int i = 0; i < 10; ++i) {
        lfg_session_decode(session);
        lfg_token t = lfg_session_sample(session);
        const auto *vocab = lfg_model_get_vocab(model);
        if (t == lfg_vocab_eos(vocab)) break;
        gen2.push_back(t);
        lfg_session_ingest_tokens(session, &t, 1, true);
    }
    CHECK(gen2.size() > 0);

    lfg_session_free(session);
    lfg_model_free(model);
}

TEST_CASE("generated_count resets on lfg_session_reset") {
    lfg_backend_init();
    lfg_model *model = load_model();
    if (!model) { MESSAGE("Skipping: model not found at ", MODEL_PATH); return; }

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = 256;
    cfg.sampling.temp = 0.0f;
    cfg.max_tokens = 20; // Set a limit
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

    // Reset
    lfg_session_reset(session);

    // After reset, max_tokens should not be exhausted — generate 5 more
    lfg_session_ingest_tokens(session, &bos, 1, true);
    const auto *vocab = lfg_model_get_vocab(model);
    lfg_token eos = lfg_vocab_eos(vocab);

    int count = 0;
    for (int i = 0; i < 10; ++i) {
        lfg_session_decode(session);
        lfg_token t = lfg_session_sample(session);
        if (t == eos) break;
        count++;
        lfg_session_ingest_tokens(session, &t, 1, true);
    }
    // Should have generated more than 0 (i.e., max_tokens didn't fire prematurely)
    CHECK(count > 0);

    lfg_session_free(session);
    lfg_model_free(model);
}

TEST_CASE("Double reset does not crash") {
    lfg_backend_init();
    lfg_model *model = load_model();
    if (!model) { MESSAGE("Skipping: model not found at ", MODEL_PATH); return; }

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = 128;
    lfg_session *session = lfg_session_create(model, &cfg);
    REQUIRE(session != nullptr);

    lfg_session_reset(session);
    lfg_session_reset(session);

    // Should still be usable
    lfg_token bos = 1;
    bool ok = lfg_session_ingest_tokens(session, &bos, 1, true);
    CHECK(ok);

    lfg_session_free(session);
    lfg_model_free(model);
}
