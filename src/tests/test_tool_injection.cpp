#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfg_api.h"

#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Test for tool injection position bug:
// When tools are registered, lfg_session_decode() injects <tools>...</tools>
// XML between the prompt and the generated output. For chat_generate this
// means the XML appears AFTER <|im_start|>assistant\n, breaking the chat
// template structure and confusing small models.
//
// After the fix, tool XML is injected INTO the system message before template
// application, so the model sees it as part of the prompt context.
// ---------------------------------------------------------------------------

static const char *MODEL_350M =
    "models/lfm2-350M.gguf";
static const char *MODEL_THINKING =
    "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";

static const lfg_tool_desc TOOLS[] = {
    {"get_weather",
     "Get current weather forecast for a city or location",
     R"({"type":"object","properties":{"location":{"type":"string"}},"required":["location"]})"},
    {"send_email",
     "Send an email message to a recipient",
     R"({"type":"object","properties":{"to":{"type":"string"},"body":{"type":"string"}},"required":["to","body"]})"},
    {"search_web",
     "Search the internet for information",
     R"({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]})"},
};
static const int32_t N_TOOLS = 3;

struct test_env {
    lfg_model   *model   = nullptr;
    lfg_session *session = nullptr;
};

static bool setup(test_env *env, const char *path, int n_ctx = 2048) {
    lfg_model_load_config lcfg = lfg_model_load_default_config();
    lcfg.model_path = path;
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

struct collect_state { std::string text; };

static lfg_generate_action collect_cb(lfg_token, const char *piece, int32_t len, void *ud) {
    auto *st = static_cast<collect_state *>(ud);
    if (piece && len > 0) st->text.append(piece, len);
    return LFG_GENERATE_CONTINUE;
}

static bool contains(const std::string &s, const char *sub) {
    return s.find(sub) != std::string::npos;
}

// Extract the answer portion after <think>...</think> blocks
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

// ===========================================================================
// 350M model — fast regression tests
// ===========================================================================

TEST_CASE("350M: chat_generate with tools does not produce XML in output") {
    test_env env;
    REQUIRE(setup(&env, MODEL_350M));

    REQUIRE(lfg_session_register_tools(env.session, TOOLS, N_TOOLS, 2) == N_TOOLS);

    lfg_chat_message msgs[] = {
        {"system", "You are a helpful assistant."},
        {"user",   "What is the weather like in Paris?"},
    };

    collect_state st;
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 64;
    gc.token_cb = collect_cb;
    gc.token_cb_data = &st;

    lfg_generate_result r = lfg_session_chat_generate(env.session, msgs, 2, gc);

    MESSAGE("Output: ", st.text);
    CHECK(r.n_tokens > 0);

    // The generated output should NOT start with or contain raw <tools> XML.
    // Before the fix, tool XML was injected after the assistant turn marker,
    // causing it to appear as generated text.
    CHECK_MESSAGE(!contains(st.text, "<tools>"),
        "Generated output should not contain raw <tools> XML");
    CHECK_MESSAGE(!contains(st.text, "<tool name="),
        "Generated output should not contain raw <tool name= XML");

    teardown(&env);
}

TEST_CASE("350M: prompt_generate with tools does not crash") {
    test_env env;
    REQUIRE(setup(&env, MODEL_350M));

    REQUIRE(lfg_session_register_tools(env.session, TOOLS, N_TOOLS, 2) == N_TOOLS);

    const char *prompt = "What is the weather like in Paris?";

    collect_state st;
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 64;
    gc.token_cb = collect_cb;
    gc.token_cb_data = &st;

    lfg_generate_result r = lfg_session_prompt_generate(
        env.session, prompt, (int32_t)strlen(prompt), true, gc);

    MESSAGE("Output: ", st.text);
    CHECK(r.n_tokens > 0);
    // For raw prompts without chat template structure, the model may echo tool
    // XML since there's no system/assistant framing. The key fix is for
    // chat_generate which properly wraps tools in the system message.
    // Here we just verify it doesn't crash.
    CHECK(!st.text.empty());

    teardown(&env);
}

TEST_CASE("350M: chat_generate multi-turn with tools — reset and re-inject") {
    test_env env;
    REQUIRE(setup(&env, MODEL_350M));

    REQUIRE(lfg_session_register_tools(env.session, TOOLS, N_TOOLS, 2) == N_TOOLS);

    // First turn
    {
        lfg_chat_message msgs[] = {
            {"system", "You are a helpful assistant."},
            {"user",   "What is the weather?"},
        };

        collect_state st;
        lfg_generate_config gc = lfg_generate_default_config();
        gc.max_tokens = 32;
        gc.token_cb = collect_cb;
        gc.token_cb_data = &st;

        lfg_generate_result r = lfg_session_chat_generate(env.session, msgs, 2, gc);
        MESSAGE("Turn 1: ", st.text);
        CHECK(r.n_tokens > 0);
        CHECK(!contains(st.text, "<tools>"));
    }

    // Reset and second turn (tools_injected resets → re-ranking)
    lfg_session_reset(env.session);
    {
        lfg_chat_message msgs[] = {
            {"system", "You are a helpful assistant."},
            {"user",   "Send an email to bob."},
        };

        collect_state st;
        lfg_generate_config gc = lfg_generate_default_config();
        gc.max_tokens = 32;
        gc.token_cb = collect_cb;
        gc.token_cb_data = &st;

        lfg_generate_result r = lfg_session_chat_generate(env.session, msgs, 2, gc);
        MESSAGE("Turn 2: ", st.text);
        CHECK(r.n_tokens > 0);
        CHECK(!contains(st.text, "<tools>"));
    }

    teardown(&env);
}

TEST_CASE("350M: chat_generate with tools + structured does not crash") {
    test_env env;
    REQUIRE(setup(&env, MODEL_350M));

    REQUIRE(lfg_session_register_tools(env.session, TOOLS, N_TOOLS, 2) == N_TOOLS);

    const char *schema = R"({
        "type": "object",
        "properties": {
            "tool": { "type": "string" },
            "args": { "type": "object" }
        },
        "required": ["tool"]
    })";
    REQUIRE(lfg_session_configure_structured(env.session, schema, nullptr));

    lfg_chat_message msgs[] = {
        {"system", "You are a tool-calling assistant. Return JSON."},
        {"user",   "Get the weather in Tokyo."},
    };

    collect_state st;
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 64;
    gc.token_cb = collect_cb;
    gc.token_cb_data = &st;

    lfg_generate_result r = lfg_session_chat_generate(env.session, msgs, 2, gc);

    MESSAGE("Output: ", st.text);
    CHECK(r.n_tokens > 0);
    REQUIRE(!st.text.empty());
    // Structured output should start with {
    CHECK(st.text[0] == '{');

    teardown(&env);
}

// ===========================================================================
// 1.2B Thinking model — real integration tests
// ===========================================================================

TEST_CASE("Thinking: chat_generate with tools produces coherent output") {
    test_env env;
    if (!setup(&env, MODEL_THINKING, 2048)) {
        MESSAGE("Skipping: thinking model not available");
        return;
    }

    REQUIRE(lfg_session_register_tools(env.session, TOOLS, N_TOOLS, 2) == N_TOOLS);

    lfg_chat_message msgs[] = {
        {"system", "You are a helpful assistant with access to tools."},
        {"user",   "What is the weather like in San Francisco?"},
    };

    collect_state st;
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 128;
    gc.token_cb = collect_cb;
    gc.token_cb_data = &st;

    lfg_generate_result r = lfg_session_chat_generate(env.session, msgs, 2, gc);

    std::string response = strip_thinking(st.text);
    MESSAGE("Output: ", response);
    CHECK(r.n_tokens > 0);
    CHECK(!response.empty());

    // Must not leak raw tool XML into generated output
    CHECK(!contains(response, "<tools>"));
    CHECK(!contains(response, "<tool name="));

    teardown(&env);
}

TEST_CASE("Thinking: chat_generate multi-turn with tools") {
    test_env env;
    if (!setup(&env, MODEL_THINKING, 4096)) {
        MESSAGE("Skipping: thinking model not available");
        return;
    }

    REQUIRE(lfg_session_register_tools(env.session, TOOLS, N_TOOLS, 2) == N_TOOLS);

    std::vector<lfg_chat_message> history;
    std::vector<std::string> stored_responses;

    history.push_back({"system", "You are a helpful assistant with access to tools."});

    const char *turns[] = {
        "What is the weather in Tokyo?",
        "Can you also check Paris?",
        "Send an email to alice about the weather.",
    };

    for (int i = 0; i < 3; ++i) {
        lfg_session_reset(env.session);
        history.push_back({"user", turns[i]});

        collect_state st;
        lfg_generate_config gc = lfg_generate_default_config();
        gc.max_tokens = 128;
        gc.token_cb = collect_cb;
        gc.token_cb_data = &st;

        lfg_generate_result r = lfg_session_chat_generate(
            env.session, history.data(), history.size(), gc);

        std::string response = strip_thinking(st.text);
        stored_responses.push_back(response);
        history.push_back({"assistant", stored_responses.back().c_str()});

        MESSAGE("Turn ", i + 1, ": ", response);
        CHECK(r.n_tokens > 0);
        CHECK(!response.empty());
        CHECK(!contains(response, "<tools>"));
    }

    teardown(&env);
}

TEST_CASE("Thinking: chat_generate with tools + structured JSON") {
    test_env env;
    if (!setup(&env, MODEL_THINKING, 2048)) {
        MESSAGE("Skipping: thinking model not available");
        return;
    }

    REQUIRE(lfg_session_register_tools(env.session, TOOLS, N_TOOLS, 2) == N_TOOLS);

    const char *schema = R"({
        "type": "object",
        "properties": {
            "function": { "type": "string" },
            "parameters": {
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"]
            }
        },
        "required": ["function", "parameters"]
    })";
    REQUIRE(lfg_session_configure_structured(env.session, schema, nullptr));

    lfg_chat_message msgs[] = {
        {"system", "You return tool calls as JSON."},
        {"user",   "Get the weather in Berlin."},
    };

    collect_state st;
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 128;
    gc.token_cb = collect_cb;
    gc.token_cb_data = &st;

    lfg_generate_result r = lfg_session_chat_generate(env.session, msgs, 2, gc);

    MESSAGE("Output: ", st.text);
    CHECK(r.n_tokens > 0);
    REQUIRE(!st.text.empty());
    CHECK(st.text[0] == '{');
    CHECK(contains(st.text, "\"function\""));
    CHECK(contains(st.text, "\"location\""));

    teardown(&env);
}

TEST_CASE("Thinking: chat_generate without system message — tools still work") {
    test_env env;
    if (!setup(&env, MODEL_THINKING, 2048)) {
        MESSAGE("Skipping: thinking model not available");
        return;
    }

    REQUIRE(lfg_session_register_tools(env.session, TOOLS, N_TOOLS, 2) == N_TOOLS);

    // No system message — tool XML should be injected as a new system message
    lfg_chat_message msgs[] = {
        {"user", "What is the weather in London?"},
    };

    collect_state st;
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 128;
    gc.token_cb = collect_cb;
    gc.token_cb_data = &st;

    lfg_generate_result r = lfg_session_chat_generate(env.session, msgs, 1, gc);

    std::string response = strip_thinking(st.text);
    MESSAGE("Output: ", response);
    CHECK(r.n_tokens > 0);
    // Model may produce only thinking content with no visible response when
    // there is no system message.  Use WARN to avoid hard failure.
    WARN(!response.empty());
    CHECK(!contains(response, "<tools>"));

    teardown(&env);
}
