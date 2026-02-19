#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/lfg_api.h"
#include <cstring>

static lfg_model * g_350m = nullptr;

static lfg_model * get_350m() {
    if (!g_350m) {
        lfg_backend_init();
        lfg_model_load_config cfg = lfg_model_load_default_config();
        cfg.model_path = "models/lfm2-350M.gguf";
        cfg.n_gpu_layers = 0;
        g_350m = lfg_load_model(&cfg);
    }
    return g_350m;
}

TEST_CASE("Guardrails track throughput and trigger level") {
    lfg_model *model = get_350m();
    if (!model) { MESSAGE("Skipping: 350M model not found"); return; }

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 512;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_guardrail_config gcfg = lfg_guardrail_default_config();
    gcfg.enabled = true;
    gcfg.window_size = 8;
    gcfg.p50_tps_min = 1e9f;  // force a guardrail level
    gcfg.p95_latency_ms_max = 0.0f;
    REQUIRE(lfg_session_configure_guardrails(session, &gcfg));

    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 8;
    const char *prompt = "Hello.\n";
    lfg_generate_result r = lfg_session_prompt_generate(session, prompt, (int32_t)std::strlen(prompt), true, gc);
    (void)r;

    lfg_guardrail_stats stats{};
    REQUIRE(lfg_session_get_guardrail_stats(session, &stats));
    CHECK(stats.sample_count > 0);
    CHECK(stats.window_size == gcfg.window_size);
    CHECK(stats.p50_tps > 0.0f);
    CHECK(stats.p95_latency_ms > 0.0f);
    CHECK(stats.level != LFG_GUARDRAIL_LEVEL_NONE);

    lfg_session_free(session);
}

TEST_CASE("Guardrails disable cleanly") {
    lfg_model *model = get_350m();
    if (!model) { MESSAGE("Skipping: 350M model not found"); return; }

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 256;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_guardrail_config gcfg = lfg_guardrail_default_config();
    gcfg.enabled = true;
    gcfg.window_size = 4;
    gcfg.p50_tps_min = 0.0f;
    gcfg.p95_latency_ms_max = 0.0f;
    REQUIRE(lfg_session_configure_guardrails(session, &gcfg));

    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 4;
    const char *prompt = "Hi.\n";
    lfg_session_prompt_generate(session, prompt, (int32_t)std::strlen(prompt), true, gc);

    lfg_guardrail_stats stats{};
    REQUIRE(lfg_session_get_guardrail_stats(session, &stats));
    CHECK(stats.level == LFG_GUARDRAIL_LEVEL_NONE);

    REQUIRE(lfg_session_configure_guardrails(session, nullptr));
    CHECK_FALSE(lfg_session_get_guardrail_stats(session, &stats));

    lfg_session_free(session);
}
