#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/inference_core.h"
#include "../loader/model_loader.h"
#include <fstream>

using namespace liquid;

TEST_CASE("Structured decoding does not double-accept sampled tokens") {
    lfm_backend_init();

    const std::string model_path = "models/lfm2-350M.gguf";
    std::ifstream f(model_path);
    if (!f.good()) {
        MESSAGE("Model not found at " << model_path << ". Skipping test.");
        return;
    }

    ModelLoader::ModelConfig load_config;
    load_config.model_path = model_path;
    load_config.n_gpu_layers = 0;

    lfm_model * model = ModelLoader::LoadModel(load_config);
    REQUIRE(model != nullptr);

    InferenceCore::Config config;
    config.n_ctx = 512;
    config.sampling.temp = 0.0f;

    InferenceCore core(model, config);

    const std::string grammar = R"GBNF(
root ::= "ab"
)GBNF";

    core.ConfigureStructuredDecoding(grammar);

    const auto * vocab = lfm_model_get_vocab(model);
    const lfm_token bos = lfm_vocab_bos(vocab);

    CHECK(core.IngestTokens({bos}, false));

    core.Decode();
    lfm_token token = core.Sample();

    bool ingest_ok = false;
    CHECK_NOTHROW(ingest_ok = core.IngestTokens({token}));
    CHECK(ingest_ok);

    lfm_model_free(model);
}
