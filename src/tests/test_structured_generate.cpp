#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfg_api.h"

#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Integration tests for ConfigureStructured + PromptGenerate/ChatGenerate.
//
// These verify the fix for a bug where grammar sampler received prompt tokens
// (e.g. chat template markers like <|im_start|>), causing:
//   "Unexpected empty grammar stack after accepting piece"
//
// Tests run against the 350M model (fast) and the 1.2B thinking model
// (full chat template integration with multi-turn, system messages, and
// JSON schema output validation).
// ---------------------------------------------------------------------------

static const char *MODEL_350M    = "models/lfm2-350M.gguf";
static const char *MODEL_THINKING =
    "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";

struct test_env {
    lfg_model   *model   = nullptr;
    lfg_session *session = nullptr;
    const lfg_vocab *vocab = nullptr;
};

static bool setup(test_env *env, const char *model_path, int n_ctx = 512) {
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

// Helper: generate config with collector
static lfg_generate_config make_gc(collect_state *st, int max_tokens) {
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = max_tokens;
    gc.token_cb = collect_cb;
    gc.token_cb_data = st;
    return gc;
}

// Helper: check string starts with a given prefix
static bool starts_with(const std::string &s, const char *prefix) {
    return s.rfind(prefix, 0) == 0;
}

// Helper: check string contains substring
static bool contains(const std::string &s, const char *sub) {
    return s.find(sub) != std::string::npos;
}

// ===========================================================================
// 350M model — basic regression tests (fast)
// ===========================================================================

static const char *SCHEMA_PERSON = R"({
    "type": "object",
    "properties": {
        "name": { "type": "string" },
        "age":  { "type": "integer" }
    },
    "required": ["name", "age"]
})";

TEST_CASE("350M: prompt_generate + JSON schema does not crash") {
    test_env env;
    REQUIRE(setup(&env, MODEL_350M));

    REQUIRE(lfg_session_configure_structured(env.session, SCHEMA_PERSON, nullptr));

    collect_state st;
    lfg_generate_config gc = make_gc(&st, 64);

    lfg_generate_result r = lfg_session_prompt_generate(
        env.session, "Generate a person:", 18, true, gc);

    MESSAGE("Output: ", st.text);
    CHECK(r.n_tokens > 0);
    REQUIRE(!st.text.empty());
    CHECK(st.text[0] == '{');

    teardown(&env);
}

TEST_CASE("350M: chat_generate + JSON schema does not crash") {
    test_env env;
    REQUIRE(setup(&env, MODEL_350M));

    REQUIRE(lfg_session_configure_structured(env.session, SCHEMA_PERSON, nullptr));

    lfg_chat_message msgs[] = {{"user", "Generate a person with name and age."}};

    collect_state st;
    lfg_generate_config gc = make_gc(&st, 64);

    lfg_generate_result r = lfg_session_chat_generate(env.session, msgs, 1, gc);

    MESSAGE("Output: ", st.text);
    CHECK(r.n_tokens > 0);
    REQUIRE(!st.text.empty());
    CHECK(st.text[0] == '{');

    teardown(&env);
}

TEST_CASE("350M: GBNF grammar still constrains output through prompt_generate") {
    test_env env;
    REQUIRE(setup(&env, MODEL_350M));

    REQUIRE(lfg_session_configure_structured(
        env.session, R"(root ::= "yes" | "no")", "root"));

    collect_state st;
    lfg_generate_config gc = make_gc(&st, 8);

    lfg_session_prompt_generate(env.session, "Answer yes or no:", 17, true, gc);

    MESSAGE("Output: ", st.text);
    REQUIRE(!st.text.empty());
    CHECK((starts_with(st.text, "yes") || starts_with(st.text, "no")));

    teardown(&env);
}

TEST_CASE("350M: structured + reset + prompt_generate across multiple rounds") {
    test_env env;
    REQUIRE(setup(&env, MODEL_350M));

    for (int round = 0; round < 3; round++) {
        lfg_session_reset(env.session);
        REQUIRE(lfg_session_configure_structured(env.session, SCHEMA_PERSON, nullptr));

        collect_state st;
        lfg_generate_config gc = make_gc(&st, 64);

        lfg_generate_result r = lfg_session_prompt_generate(
            env.session, "Generate a person:", 18, true, gc);

        MESSAGE("Round ", round, " output: ", st.text);
        CHECK(r.n_tokens > 0);
        REQUIRE(!st.text.empty());
        CHECK(st.text[0] == '{');
    }

    teardown(&env);
}

// ===========================================================================
// 1.2B Thinking model — real chat template integration tests
// ===========================================================================

static const char *SCHEMA_CAPITAL = R"({
    "type": "object",
    "properties": {
        "country": { "type": "string" },
        "capital": { "type": "string" }
    },
    "required": ["country", "capital"]
})";

static const char *SCHEMA_MATH = R"({
    "type": "object",
    "properties": {
        "expression": { "type": "string" },
        "result":     { "type": "integer" }
    },
    "required": ["expression", "result"]
})";

static const char *SCHEMA_TOOL_CALL = R"({
    "type": "object",
    "properties": {
        "function": { "type": "string" },
        "parameters": {
            "type": "object",
            "properties": {
                "location": { "type": "string" },
                "unit":     { "enum": ["celsius", "fahrenheit"] }
            },
            "required": ["location", "unit"]
        }
    },
    "required": ["function", "parameters"]
})";

static const char *SCHEMA_SENTIMENT = R"({
    "type": "object",
    "properties": {
        "text":      { "type": "string" },
        "sentiment": { "enum": ["positive", "negative", "neutral"] },
        "score":     { "type": "number" }
    },
    "required": ["text", "sentiment", "score"]
})";

static const char *SCHEMA_LIST = R"({
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": { "type": "string" }
        }
    },
    "required": ["items"]
})";

TEST_CASE("Thinking: chat_generate + JSON schema — single user message") {
    test_env env;
    if (!setup(&env, MODEL_THINKING, 2048)) {
        MESSAGE("Skipping: thinking model not available");
        return;
    }

    REQUIRE(lfg_session_configure_structured(env.session, SCHEMA_CAPITAL, nullptr));

    lfg_chat_message msgs[] = {
        {"user", "What is the capital of France? Return JSON with country and capital fields."},
    };

    collect_state st;
    lfg_generate_config gc = make_gc(&st, 128);

    lfg_generate_result r = lfg_session_chat_generate(env.session, msgs, 1, gc);

    MESSAGE("Generated ", r.n_tokens, " tokens");
    MESSAGE("Output: ", st.text);

    CHECK(r.n_tokens > 0);
    REQUIRE(!st.text.empty());
    CHECK(st.text[0] == '{');
    CHECK(contains(st.text, "\"country\""));
    CHECK(contains(st.text, "\"capital\""));

    teardown(&env);
}

TEST_CASE("Thinking: chat_generate + JSON schema — system + user message") {
    test_env env;
    if (!setup(&env, MODEL_THINKING, 2048)) {
        MESSAGE("Skipping: thinking model not available");
        return;
    }

    REQUIRE(lfg_session_configure_structured(env.session, SCHEMA_MATH, nullptr));

    lfg_chat_message msgs[] = {
        {"system", "You are a math assistant. Always return JSON."},
        {"user",   "What is 17 * 3? Return JSON with expression and result fields."},
    };

    collect_state st;
    lfg_generate_config gc = make_gc(&st, 128);

    lfg_generate_result r = lfg_session_chat_generate(env.session, msgs, 2, gc);

    MESSAGE("Generated ", r.n_tokens, " tokens");
    MESSAGE("Output: ", st.text);

    CHECK(r.n_tokens > 0);
    REQUIRE(!st.text.empty());
    CHECK(st.text[0] == '{');
    CHECK(contains(st.text, "\"expression\""));
    CHECK(contains(st.text, "\"result\""));

    teardown(&env);
}

TEST_CASE("Thinking: chat_generate + JSON schema — multi-turn conversation") {
    test_env env;
    if (!setup(&env, MODEL_THINKING, 2048)) {
        MESSAGE("Skipping: thinking model not available");
        return;
    }

    REQUIRE(lfg_session_configure_structured(env.session, SCHEMA_TOOL_CALL, nullptr));

    lfg_chat_message msgs[] = {
        {"system",    "You are an assistant that returns tool calls as JSON."},
        {"user",      "I need the weather."},
        {"assistant", "Sure, which city?"},
        {"user",      "Get the weather in Tokyo in celsius."},
    };

    collect_state st;
    lfg_generate_config gc = make_gc(&st, 128);

    lfg_generate_result r = lfg_session_chat_generate(env.session, msgs, 4, gc);

    MESSAGE("Generated ", r.n_tokens, " tokens");
    MESSAGE("Output: ", st.text);

    CHECK(r.n_tokens > 0);
    REQUIRE(!st.text.empty());
    CHECK(st.text[0] == '{');
    CHECK(contains(st.text, "\"function\""));
    CHECK(contains(st.text, "\"parameters\""));
    CHECK(contains(st.text, "\"location\""));
    CHECK(contains(st.text, "\"unit\""));

    teardown(&env);
}

TEST_CASE("Thinking: chat_generate + enum constraint — sentiment classification") {
    test_env env;
    if (!setup(&env, MODEL_THINKING, 2048)) {
        MESSAGE("Skipping: thinking model not available");
        return;
    }

    REQUIRE(lfg_session_configure_structured(env.session, SCHEMA_SENTIMENT, nullptr));

    lfg_chat_message msgs[] = {
        {"system", "You are a sentiment analysis engine. Return JSON."},
        {"user",   "Classify the sentiment of: 'I love this product, it is amazing!' Return text, sentiment, and score."},
    };

    collect_state st;
    lfg_generate_config gc = make_gc(&st, 128);

    lfg_generate_result r = lfg_session_chat_generate(env.session, msgs, 2, gc);

    MESSAGE("Generated ", r.n_tokens, " tokens");
    MESSAGE("Output: ", st.text);

    CHECK(r.n_tokens > 0);
    REQUIRE(!st.text.empty());
    CHECK(st.text[0] == '{');
    CHECK(contains(st.text, "\"sentiment\""));
    // The enum constraint must produce one of: positive, negative, neutral
    bool has_valid_sentiment =
        contains(st.text, "\"positive\"") ||
        contains(st.text, "\"negative\"") ||
        contains(st.text, "\"neutral\"");
    CHECK(has_valid_sentiment);

    teardown(&env);
}

TEST_CASE("Thinking: chat_generate + array schema — list generation") {
    test_env env;
    if (!setup(&env, MODEL_THINKING, 2048)) {
        MESSAGE("Skipping: thinking model not available");
        return;
    }

    REQUIRE(lfg_session_configure_structured(env.session, SCHEMA_LIST, nullptr));

    lfg_chat_message msgs[] = {
        {"user", "List 3 programming languages as JSON with an items array."},
    };

    collect_state st;
    lfg_generate_config gc = make_gc(&st, 128);

    lfg_generate_result r = lfg_session_chat_generate(env.session, msgs, 1, gc);

    MESSAGE("Generated ", r.n_tokens, " tokens");
    MESSAGE("Output: ", st.text);

    CHECK(r.n_tokens > 0);
    REQUIRE(!st.text.empty());
    CHECK(st.text[0] == '{');
    CHECK(contains(st.text, "\"items\""));
    CHECK(contains(st.text, "["));

    teardown(&env);
}

TEST_CASE("Thinking: prompt_generate + JSON schema — raw prompt with chat-like content") {
    test_env env;
    if (!setup(&env, MODEL_THINKING, 2048)) {
        MESSAGE("Skipping: thinking model not available");
        return;
    }

    REQUIRE(lfg_session_configure_structured(env.session, SCHEMA_PERSON, nullptr));

    const char *prompt = "Generate a JSON object describing a person with name and age:";

    collect_state st;
    lfg_generate_config gc = make_gc(&st, 128);

    lfg_generate_result r = lfg_session_prompt_generate(
        env.session, prompt, (int32_t)strlen(prompt), true, gc);

    MESSAGE("Generated ", r.n_tokens, " tokens");
    MESSAGE("Output: ", st.text);

    CHECK(r.n_tokens > 0);
    REQUIRE(!st.text.empty());
    CHECK(st.text[0] == '{');
    CHECK(contains(st.text, "\"name\""));
    CHECK(contains(st.text, "\"age\""));

    teardown(&env);
}

TEST_CASE("Thinking: chat_generate without structured — unconstrained generation") {
    test_env env;
    if (!setup(&env, MODEL_THINKING, 2048)) {
        MESSAGE("Skipping: thinking model not available");
        return;
    }

    // No structured decoding — just verify chat_generate works at all
    lfg_chat_message msgs[] = {
        {"user", "What is the capital of Japan? Answer in one word."},
    };

    collect_state st;
    lfg_generate_config gc = make_gc(&st, 64);

    lfg_generate_result r = lfg_session_chat_generate(env.session, msgs, 1, gc);

    MESSAGE("Generated ", r.n_tokens, " tokens");
    MESSAGE("Output: ", st.text);

    CHECK(r.n_tokens > 0);
    REQUIRE(!st.text.empty());

    teardown(&env);
}

TEST_CASE("Thinking: chat_generate + structured — reset between calls") {
    test_env env;
    if (!setup(&env, MODEL_THINKING, 2048)) {
        MESSAGE("Skipping: thinking model not available");
        return;
    }

    for (int round = 0; round < 2; round++) {
        lfg_session_reset(env.session);
        REQUIRE(lfg_session_configure_structured(env.session, SCHEMA_CAPITAL, nullptr));

        const char *countries[] = {"France", "Germany"};
        char prompt_buf[128];
        snprintf(prompt_buf, sizeof(prompt_buf),
            "What is the capital of %s? Return JSON.", countries[round]);

        lfg_chat_message msgs[] = {
            {"user", prompt_buf},
        };

        collect_state st;
        lfg_generate_config gc = make_gc(&st, 128);

        lfg_generate_result r = lfg_session_chat_generate(env.session, msgs, 1, gc);

        MESSAGE("Round ", round, " (", countries[round], "): ", st.text);

        CHECK(r.n_tokens > 0);
        REQUIRE(!st.text.empty());
        CHECK(st.text[0] == '{');
        CHECK(contains(st.text, "\"country\""));
        CHECK(contains(st.text, "\"capital\""));
    }

    teardown(&env);
}
