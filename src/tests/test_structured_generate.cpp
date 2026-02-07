#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfg_api.h"

#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Reproduce: ConfigureStructured + PromptGenerate/ChatGenerate crashes because
// prompt tokens are fed through the grammar via lfg_sampler_accept.
// The grammar (JSON schema -> GBNF) rejects prompt tokens (e.g. <, chat
// template markers), causing "Unexpected empty grammar stack after accepting
// piece".  Grammar should only constrain generated tokens, not ingested prompt.
// ---------------------------------------------------------------------------

static const char *MODEL_PATH = "models/lfm2-350M.gguf";

struct test_env {
    lfg_model   *model   = nullptr;
    lfg_session *session = nullptr;
    const lfg_vocab *vocab = nullptr;
};

static bool setup(test_env *env, int n_ctx = 512) {
    lfg_model_load_config lcfg = lfg_model_load_default_config();
    lcfg.model_path = MODEL_PATH;
    env->model = lfg_load_model(&lcfg);
    if (!env->model) return false;

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = n_ctx;
    cfg.sampling.temp = 0.0f;
    env->session = lfg_session_create(env->model, &cfg);
    if (!env->session) {
        lfg_model_free(env->model);
        return false;
    }
    env->vocab = lfg_model_get_vocab(env->model);
    return true;
}

static void teardown(test_env *env) {
    if (env->session) lfg_session_free(env->session);
    if (env->model) lfg_model_free(env->model);
}

// Collect generated text via token callback
struct collect_state {
    std::string text;
};

static lfg_generate_action collect_cb(lfg_token, const char *piece, int32_t len, void *ud) {
    auto *st = static_cast<collect_state *>(ud);
    if (piece && len > 0) st->text.append(piece, len);
    return LFG_GENERATE_CONTINUE;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static const char *SCHEMA_SIMPLE = R"({
    "type": "object",
    "properties": {
        "name": { "type": "string" },
        "age":  { "type": "integer" }
    },
    "required": ["name", "age"]
})";

TEST_CASE("prompt_generate + structured decoding does not crash on prompt tokens") {
    test_env env;
    REQUIRE(setup(&env));

    bool ok = lfg_session_configure_structured(env.session, SCHEMA_SIMPLE, nullptr);
    REQUIRE(ok);

    collect_state st;
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 64;
    gc.token_cb = collect_cb;
    gc.token_cb_data = &st;

    // Before fix: this crashes with "Unexpected empty grammar stack after
    // accepting piece" because prompt tokens go through the grammar sampler.
    lfg_generate_result r = lfg_session_prompt_generate(
        env.session, "Generate a person:", 18, true, gc);

    CHECK(r.n_tokens > 0);
    // Output should start with '{' since we constrained to JSON object
    REQUIRE(!st.text.empty());
    CHECK(st.text[0] == '{');

    teardown(&env);
}

TEST_CASE("chat_generate + structured decoding does not crash on chat template tokens") {
    test_env env;
    REQUIRE(setup(&env));

    const char *schema = R"({
        "type": "object",
        "properties": {
            "answer": { "type": "string" }
        },
        "required": ["answer"]
    })";
    bool ok = lfg_session_configure_structured(env.session, schema, nullptr);
    REQUIRE(ok);

    lfg_chat_message msgs[1];
    msgs[0].role = "user";
    msgs[0].content = "What is the capital of France?";

    collect_state st;
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 64;
    gc.token_cb = collect_cb;
    gc.token_cb_data = &st;

    // Before fix: chat template tokens (e.g. <|im_start|>) crash the grammar.
    lfg_generate_result r = lfg_session_chat_generate(
        env.session, msgs, 1, gc);

    CHECK(r.n_tokens > 0);
    REQUIRE(!st.text.empty());
    CHECK(st.text[0] == '{');

    teardown(&env);
}

TEST_CASE("prompt_generate + structured still constrains generated output") {
    test_env env;
    REQUIRE(setup(&env));

    // Constrain to a simple yes/no grammar
    bool ok = lfg_session_configure_structured(
        env.session, R"(root ::= "yes" | "no")", "root");
    REQUIRE(ok);

    collect_state st;
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 8;
    gc.token_cb = collect_cb;
    gc.token_cb_data = &st;

    lfg_generate_result r = lfg_session_prompt_generate(
        env.session, "Answer yes or no:", 17, true, gc);

    CHECK(r.n_tokens > 0);
    // Output must be "yes" or "no" — grammar must still constrain generation
    REQUIRE(!st.text.empty());
    bool valid = (st.text.find("yes") == 0 || st.text.find("no") == 0);
    CHECK(valid);

    teardown(&env);
}

TEST_CASE("structured + reset + prompt_generate works across multiple rounds") {
    test_env env;
    REQUIRE(setup(&env));

    bool ok = lfg_session_configure_structured(env.session, SCHEMA_SIMPLE, nullptr);
    REQUIRE(ok);

    for (int round = 0; round < 3; round++) {
        lfg_session_reset(env.session);

        // Re-configure after reset (grammar state is cleared)
        ok = lfg_session_configure_structured(env.session, SCHEMA_SIMPLE, nullptr);
        REQUIRE(ok);

        collect_state st;
        lfg_generate_config gc = lfg_generate_default_config();
        gc.max_tokens = 64;
        gc.token_cb = collect_cb;
        gc.token_cb_data = &st;

        lfg_generate_result r = lfg_session_prompt_generate(
            env.session, "Generate a person:", 18, true, gc);

        CHECK(r.n_tokens > 0);
        REQUIRE(!st.text.empty());
        CHECK(st.text[0] == '{');
    }

    teardown(&env);
}
