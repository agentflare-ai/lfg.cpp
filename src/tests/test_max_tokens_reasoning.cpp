#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfg_api.h"

#include <fstream>
#include <vector>
#include <cstring>

static const char *MODEL_PATH = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";

static bool tokenize_literal(const lfg_vocab *vocab, const char *text, std::vector<lfg_token> &out) {
    if (!vocab || !text) return false;
    int32_t cap = (int32_t)std::strlen(text) + 8;
    out.resize(cap);
    int32_t n = lfg_tokenize(vocab, text, (int32_t)std::strlen(text), out.data(), cap, false, true);
    if (n < 0) {
        cap = -n;
        out.resize(cap);
        n = lfg_tokenize(vocab, text, (int32_t)std::strlen(text), out.data(), cap, false, true);
    }
    if (n <= 0) {
        out.clear();
        return false;
    }
    out.resize(n);
    return true;
}

static lfg_model *load_model() {
    std::ifstream f(MODEL_PATH);
    if (!f.good()) return nullptr;

    lfg_model_load_config cfg = lfg_model_load_default_config();
    cfg.model_path = MODEL_PATH;
    cfg.n_gpu_layers = 0;
    return lfg_load_model(&cfg);
}

TEST_CASE("max_tokens stops even during reasoning") {
    lfg_backend_init();
    lfg_model *model = load_model();
    if (!model) { MESSAGE("Skipping: model not found at ", MODEL_PATH); return; }

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = 1024;
    cfg.sampling.temp = 0.0f;
    cfg.max_tokens = 4;
    lfg_session *session = lfg_session_create(model, &cfg);
    REQUIRE(session != nullptr);

    std::vector<lfg_token> start_tokens;
    std::vector<lfg_token> end_tokens;
    if (!tokenize_literal(lfg_model_get_vocab(model), "<think>", start_tokens) ||
        !tokenize_literal(lfg_model_get_vocab(model), "</think>", end_tokens)) {
        MESSAGE("Skipping: reasoning markers not found in vocab");
        lfg_session_free(session);
        lfg_model_free(model);
        return;
    }
    lfg_session_configure_reasoning(session, start_tokens.data(), start_tokens.size(),
                                    end_tokens.data(), end_tokens.size());

    lfg_token bos = 1;
    lfg_session_ingest_tokens(session, &bos, 1, true);
    lfg_session_ingest_tokens(session, start_tokens.data(), start_tokens.size(), true);

    const auto *vocab = lfg_model_get_vocab(model);
    lfg_token eos = lfg_vocab_eos(vocab);

    bool saw_eos = false;
    for (int i = 0; i < 10; ++i) {
        lfg_session_decode(session);
        lfg_token t = lfg_session_sample(session);
        if (t == eos) {
            saw_eos = true;
            break;
        }
        lfg_session_ingest_tokens(session, &t, 1, true);
    }

    CHECK(saw_eos);

    lfg_session_free(session);
    lfg_model_free(model);
}
