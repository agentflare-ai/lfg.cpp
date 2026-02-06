#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/lfg_api.h"
#include <fstream>

TEST_CASE("Structured decoding does not double-accept sampled tokens") {
    lfg_backend_init();

    const std::string model_path = "models/lfm2-350M.gguf";
    std::ifstream f(model_path);
    if (!f.good()) {
        MESSAGE("Model not found at " << model_path << ". Skipping test.");
        return;
    }

    lfg_model_load_config load_config = lfg_model_load_default_config();
    load_config.model_path = model_path.c_str();
    load_config.n_gpu_layers = 0;

    lfg_model * model = lfg_load_model(&load_config);
    REQUIRE(model != nullptr);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 512;
    config.sampling.temp = 0.0f;

    lfg_session * session = lfg_session_create(model, &config);

    const std::string grammar = R"GBNF(
root ::= "ab"
)GBNF";

    lfg_session_configure_structured(session, grammar.c_str(), "root");

    const auto * vocab = lfg_model_get_vocab(model);
    const lfg_token bos = lfg_vocab_bos(vocab);

    CHECK(lfg_session_ingest_tokens(session, &bos, 1, false));

    lfg_session_decode(session);
    lfg_token token = lfg_session_sample(session);

    bool ingest_ok = false;
    CHECK_NOTHROW(ingest_ok = lfg_session_ingest_tokens(session, &token, 1, true));
    CHECK(ingest_ok);

    lfg_session_free(session);
    lfg_model_free(model);
}
