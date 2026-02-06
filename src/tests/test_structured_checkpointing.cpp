#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfg_api.h"

#include <fstream>
#include <string>

static std::string token_piece(lfg_model * model, lfg_token token) {
    const auto * vocab = lfg_model_get_vocab(model);
    char buf[256];
    const int n = lfg_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
    if (n <= 0) {
        return std::string();
    }
    return std::string(buf, n);
}

TEST_CASE("Structured checkpoint defaults (C API)") {
    lfg_session_config cfg = lfg_session_default_config();
    CHECK(cfg.structured_checkpointing == true);

    lfg_checkpoint_restore_options opts = lfg_checkpoint_restore_default_options();
    CHECK(opts.restore_sampler_state == true);
    CHECK(opts.restore_grammar == true);
}

TEST_CASE("Structured checkpoint restore options") {
    lfg_backend_init();

    const std::string model_path = "models/lfm2-350M.gguf";
    std::ifstream f(model_path);
    if (!f.good()) {
        MESSAGE("Skipping structured checkpoint test: Model not found at " << model_path);
        return;
    }

    lfg_model_load_config load_config = lfg_model_load_default_config();
    load_config.model_path = model_path.c_str();
    load_config.n_gpu_layers = 0;

    lfg_model * model = lfg_load_model(&load_config);
    REQUIRE(model != nullptr);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 512;
    config.sampling.seed = 42;
    config.sampling.temp = 0.0f;
    config.structured_checkpointing = true;

    lfg_session * session = lfg_session_create(model, &config);

    // Ingest BOS so we have logits to sample from
    lfg_token bos = 1;
    lfg_session_ingest_tokens(session, &bos, 1, false);

    const std::string grammar_yes = "root ::= \"yes\"";
    const std::string grammar_no = "root ::= \"no\"";

    lfg_session_configure_structured(session, grammar_yes.c_str(), "root");
    lfg_checkpoint * cp = lfg_session_create_checkpoint(session);

    lfg_session_configure_structured(session, grammar_no.c_str(), "root");

    lfg_checkpoint_restore_options keep_grammar = lfg_checkpoint_restore_default_options();
    keep_grammar.restore_grammar = false;
    keep_grammar.restore_sampler_state = true;
    CHECK(lfg_session_restore_checkpoint_ex(session, cp, &keep_grammar));

    lfg_token token_no = lfg_session_sample(session);
    std::string piece_no = token_piece(model, token_no);
    CHECK(!piece_no.empty());
    CHECK(piece_no[0] == 'n');
    lfg_session_ingest_tokens(session, &token_no, 1, false);

    lfg_checkpoint_restore_options restore_grammar = lfg_checkpoint_restore_default_options();
    restore_grammar.restore_grammar = true;
    restore_grammar.restore_sampler_state = true;
    CHECK(lfg_session_restore_checkpoint_ex(session, cp, &restore_grammar));

    lfg_token token_yes = lfg_session_sample(session);
    std::string piece_yes = token_piece(model, token_yes);
    CHECK(!piece_yes.empty());
    CHECK(piece_yes[0] == 'y');

    lfg_checkpoint_free(cp);
    lfg_session_free(session);
    lfg_model_free(model);
}
