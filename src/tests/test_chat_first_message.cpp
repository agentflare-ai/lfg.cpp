#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfg_api.h"
#include "../inference/lfg_inference.h"

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Regression test: first chat message must produce clean output.
//
// Root cause was parse_special=false in lfg_session_chat_generate's tokenizer
// call, which split <|im_start|> into subwords instead of recognizing it as a
// single special token.  The model never saw proper chat delimiters.
//
// Reproduces the demo's exact flow:
//   1. Load model, create session
//   2. Reset session
//   3. Build message array: [system, user]
//   4. Call lfg_session_chat_generate
//   5. Verify output is non-empty and contains no stop-string fragments
// ---------------------------------------------------------------------------

static const char *MODEL_350M = "models/lfm2-350M.gguf";
static const char *MODEL_THINKING =
    "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";

struct collect_state {
    std::string text;
    int call_count = 0;
};

static lfg_generate_action collect_cb(
    lfg_token, const char *piece, int32_t len, void *ud)
{
    auto *st = static_cast<collect_state *>(ud);
    st->call_count++;
    if (piece && len > 0) {
        st->text.append(piece, len);
    }
    return LFG_GENERATE_CONTINUE;
}

struct test_env {
    lfg_model   *model   = nullptr;
    lfg_session *session  = nullptr;
};

static bool setup(test_env *env, const char *model_path, int n_ctx = 2048) {
    lfg_model_load_config lcfg = lfg_model_load_default_config();
    lcfg.model_path = model_path;
    env->model = lfg_load_model(&lcfg);
    if (!env->model) return false;

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = n_ctx;
    cfg.sampling.temp = 0.0f;
    cfg.sampling.seed = 42;
    env->session = lfg_session_create(env->model, &cfg);
    if (!env->session) {
        lfg_model_free(env->model);
        return false;
    }
    return true;
}

static void teardown(test_env *env) {
    if (env->session) lfg_session_free(env->session);
    if (env->model) lfg_model_free(env->model);
}

// Check that a string doesn't contain partial stop tokens
static bool has_stop_fragment(const std::string &s) {
    const char *fragments[] = {
        "<|",
        "|>",
        "<|im_end",
        "<|im_start",
        "<|endoftext",
        "</|",
    };
    for (auto *f : fragments) {
        if (s.find(f) != std::string::npos) return true;
    }
    return false;
}

// Strip <think>...</think> from output (thinking model wraps responses)
static std::string strip_thinking(const std::string &s) {
    std::string cleaned = s;
    auto think_end = cleaned.find("</think>");
    if (think_end != std::string::npos) {
        cleaned = cleaned.substr(think_end + 8);
    }
    auto start = cleaned.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    auto end = cleaned.find_last_not_of(" \t\n\r");
    return cleaned.substr(start, end - start + 1);
}

// ---------------------------------------------------------------------------
// 350M model tests
// ---------------------------------------------------------------------------

TEST_CASE("350M: first chat message produces clean output") {
    test_env env;
    if (!setup(&env, MODEL_350M)) {
        MESSAGE("Skipping: model not available at ", MODEL_350M);
        return;
    }

    lfg_session_reset(env.session);

    lfg_chat_message msgs[] = {
        {"system", "You are a helpful assistant."},
        {"user",   "hey"},
    };

    collect_state st;
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 64;
    gc.token_cb = collect_cb;
    gc.token_cb_data = &st;

    lfg_generate_result r = lfg_session_chat_generate(
        env.session, msgs, 2, gc);

    MESSAGE("Raw output: [", st.text, "]");
    MESSAGE("Tokens: ", r.n_tokens, " stop: ", (int)r.stop_reason);

    CHECK(r.n_tokens > 0);
    CHECK(!st.text.empty());
    CHECK_MESSAGE(!has_stop_fragment(st.text),
        "Output contains stop fragment: [", st.text, "]");

    teardown(&env);
}

// ---------------------------------------------------------------------------
// 1.2B Thinking model tests
// ---------------------------------------------------------------------------

TEST_CASE("Thinking: first chat message produces clean output") {
    test_env env;
    if (!setup(&env, MODEL_THINKING, 4096)) {
        MESSAGE("Skipping: thinking model not available at ", MODEL_THINKING);
        return;
    }

    lfg_session_reset(env.session);

    lfg_chat_message msgs[] = {
        {"system", "You are a helpful assistant."},
        {"user",   "hey"},
    };

    collect_state st;
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 128;
    gc.token_cb = collect_cb;
    gc.token_cb_data = &st;

    lfg_generate_result r = lfg_session_chat_generate(
        env.session, msgs, 2, gc);

    std::string visible = strip_thinking(st.text);
    MESSAGE("Raw output: [", st.text, "]");
    MESSAGE("Visible: [", visible, "]");
    MESSAGE("Tokens: ", r.n_tokens, " stop: ", (int)r.stop_reason);

    CHECK(r.n_tokens > 0);
    CHECK(!st.text.empty());
    CHECK_MESSAGE(!has_stop_fragment(st.text),
        "Output contains stop fragment: [", st.text, "]");
    CHECK_MESSAGE(!has_stop_fragment(visible),
        "Visible output contains stop fragment: [", visible, "]");

    teardown(&env);
}

TEST_CASE("Thinking: multi-turn chat no stop fragments") {
    test_env env;
    if (!setup(&env, MODEL_THINKING, 4096)) {
        MESSAGE("Skipping: thinking model not available at ", MODEL_THINKING);
        return;
    }

    const char *user_msgs[] = {
        "hey",
        "what is 2+2?",
        "thanks",
    };
    int n_turns = 3;

    std::vector<lfg_chat_message> history;
    std::vector<std::string> stored;

    history.push_back({"system", "You are a helpful assistant."});

    for (int i = 0; i < n_turns; i++) {
        lfg_session_reset(env.session);

        history.push_back({"user", user_msgs[i]});

        collect_state st;
        lfg_generate_config gc = lfg_generate_default_config();
        gc.max_tokens = 128;
        gc.token_cb = collect_cb;
        gc.token_cb_data = &st;

        lfg_generate_result r = lfg_session_chat_generate(
            env.session, history.data(), history.size(), gc);

        std::string visible = strip_thinking(st.text);
        MESSAGE("Turn ", i + 1, " raw: [", st.text, "]");
        MESSAGE("Turn ", i + 1, " visible: [", visible, "]");
        MESSAGE("Tokens: ", r.n_tokens, " stop: ", (int)r.stop_reason);

        CHECK(r.n_tokens > 0);
        CHECK(!st.text.empty());
        CHECK_MESSAGE(!has_stop_fragment(st.text),
            "Turn ", i + 1, ": stop fragment in raw [", st.text, "]");
        CHECK_MESSAGE(!has_stop_fragment(visible),
            "Turn ", i + 1, ": stop fragment in visible [", visible, "]");

        stored.push_back(st.text);
        history.push_back({"assistant", stored.back().c_str()});
    }

    teardown(&env);
}
