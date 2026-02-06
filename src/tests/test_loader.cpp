#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "model_loader.h"
#include <spdlog/spdlog.h>
#include <string>

using namespace liquid;

TEST_CASE("ModelLoader Verification") {
    SUBCASE("Load non-existent model") {
        ModelLoader::ModelConfig config;
        config.model_path = "non_existent.gguf";
        config.n_gpu_layers = 0;
        
        lfm_model* model = ModelLoader::LoadModel(config);
        CHECK(model == nullptr);
        
        // Ensure manual cleanup isn't needed for nullptr, but safe to call
        ModelLoader::FreeModel(model);
    }

    SUBCASE("GetStats on null model") {
        ModelLoader::ModelStats stats = ModelLoader::GetModelStats(nullptr);
        CHECK(stats.n_params == 0);
        CHECK(stats.size_bytes == 0);
        CHECK(stats.n_vocab == 0);
    }

    SUBCASE("GetMetadata on null model") {
        std::string unknown = ModelLoader::GetMetadata(nullptr, "general.name");
        CHECK(unknown == "");
    }
}
