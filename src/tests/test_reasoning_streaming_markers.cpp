#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfg_api.h"

#include <fstream>
#include <string>

static const char *MODEL_PATH = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";

struct StreamCapture {
    std::string text;
};

static lfg_generate_action capture_cb(
    lfg_token token, const char *piece, int32_t piece_len, void *data) {
    (void)token;
    auto *cap = static_cast<StreamCapture *>(data);
    if (cap && piece && piece_len > 0) {
        cap->text.append(piece, (size_t)piece_len);
    }
    return LFG_GENERATE_CONTINUE;
}

TEST_CASE("Streaming includes <think> markers for thinking models") {
    lfg_backend_init();

    std::ifstream f(MODEL_PATH);
    if (!f.good()) {
        MESSAGE("Skipping: model not found at ", MODEL_PATH);
        return;
    }

    lfg_model_load_config cfg = lfg_model_load_default_config();
    cfg.model_path = MODEL_PATH;
    cfg.n_gpu_layers = 0;
    lfg_model *model = lfg_load_model(&cfg);
    REQUIRE(model != nullptr);

    lfg_session_config scfg = lfg_session_default_config();
    scfg.n_ctx = 2048;
    scfg.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &scfg);
    REQUIRE(session != nullptr);

    const lfg_chat_message msgs[] = {
        {"user", "Say hello in one sentence."},
    };

    StreamCapture cap;
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 128;
    gc.token_cb = capture_cb;
    gc.token_cb_data = &cap;

    (void)lfg_session_chat_generate(session, msgs, 1, gc);

    int32_t raw_len = 0;
    const char *raw_ptr = lfg_session_get_last_output(session, &raw_len);
    std::string raw_output;
    if (raw_ptr && raw_len > 0) {
        raw_output.assign(raw_ptr, (size_t)raw_len);
    }

    if (raw_output.find("<think>") == std::string::npos ||
        raw_output.find("</think>") == std::string::npos) {
        MESSAGE("Skipping: model output lacks <think> markers");
        lfg_session_free(session);
        lfg_model_free(model);
        return;
    }

    CHECK(cap.text.find("<think>") != std::string::npos);
    CHECK(cap.text.find("</think>") != std::string::npos);

    lfg_session_free(session);
    lfg_model_free(model);
}
