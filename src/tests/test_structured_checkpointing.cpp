#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/inference_core.h"
#include "../inference/lfm_api.h"
#include "../loader/model_loader.h"

#include <fstream>
#include <string>

using namespace liquid;

static std::string token_piece(lfm_model * model, lfm_token token) {
    const auto * vocab = lfm_model_get_vocab(model);
    char buf[256];
    const int n = lfm_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
    if (n <= 0) {
        return std::string();
    }
    return std::string(buf, n);
}

TEST_CASE("Structured checkpoint defaults (C API)") {
    lfm_session_config cfg = lfm_session_default_config();
    CHECK(cfg.structured_checkpointing == true);

    lfm_checkpoint_restore_options opts = lfm_checkpoint_restore_default_options();
    CHECK(opts.restore_sampler_state == true);
    CHECK(opts.restore_grammar == true);
}

TEST_CASE("Structured checkpoint restore options") {
    lfm_backend_init();

    const std::string model_path = "models/lfm2-350M.gguf";
    std::ifstream f(model_path);
    if (!f.good()) {
        MESSAGE("Skipping structured checkpoint test: Model not found at " << model_path);
        return;
    }

    ModelLoader::ModelConfig load_config;
    load_config.model_path = model_path;
    load_config.n_gpu_layers = 0;

    lfm_model * model = ModelLoader::LoadModel(load_config);
    REQUIRE(model != nullptr);

    InferenceCore::Config config;
    config.n_ctx = 512;
    config.sampling.seed = 42;
    config.sampling.temp = 0.0f;
    config.structured_checkpointing = true;

    InferenceCore core(model, config);

    const std::string grammar_yes = "root ::= \"yes\"";
    const std::string grammar_no = "root ::= \"no\"";

    core.ConfigureStructuredDecoding(grammar_yes);
    auto cp = core.CreateCheckpoint();

    core.ConfigureStructuredDecoding(grammar_no);

    InferenceCore::RestoreOptions keep_grammar;
    keep_grammar.restore_grammar = false;
    keep_grammar.restore_sampler_state = true;
    CHECK(core.RestoreCheckpoint(cp, keep_grammar));

    core.Decode();
    lfm_token token_no = core.Sample();
    std::string piece_no = token_piece(model, token_no);
    CHECK(!piece_no.empty());
    CHECK(piece_no[0] == 'n');
    core.IngestTokens({token_no}, false);

    InferenceCore::RestoreOptions restore_grammar;
    restore_grammar.restore_grammar = true;
    restore_grammar.restore_sampler_state = true;
    CHECK(core.RestoreCheckpoint(cp, restore_grammar));

    core.Decode();
    lfm_token token_yes = core.Sample();
    std::string piece_yes = token_piece(model, token_yes);
    CHECK(!piece_yes.empty());
    CHECK(piece_yes[0] == 'y');

    lfm_model_free(model);
}
