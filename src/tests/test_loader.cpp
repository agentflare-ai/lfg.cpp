#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/lfg_api.h"
#include <spdlog/spdlog.h>
#include <string>

TEST_CASE("ModelLoader Verification") {
    SUBCASE("Load non-existent model") {
        lfg_model_load_config config = lfg_model_load_default_config();
        config.model_path = "non_existent.gguf";
        config.n_gpu_layers = 0;

        lfg_model* model = lfg_load_model(&config);
        CHECK(model == nullptr);

        // Ensure manual cleanup isn't needed for nullptr, but safe to call
        lfg_model_free(model);
    }

    SUBCASE("GetStats on null model") {
        lfg_model_stats stats = lfg_model_get_stats(nullptr);
        CHECK(stats.n_params == 0);
        CHECK(stats.size_bytes == 0);
        CHECK(stats.n_vocab == 0);
    }

    SUBCASE("GetMetadata on null model") {
        char buf[256] = {};
        int32_t n = lfg_model_get_metadata_str(nullptr, "general.name", buf, sizeof(buf));
        CHECK(n <= 0);
    }
}
