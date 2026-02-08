#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../inference/lfg_api.h"

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Integration tests: real multi-turn conversations where the model has tools
// available and uses them when appropriate.  Tests verify:
//   1. The model produces valid JSON tool calls via structured output
//   2. The model picks the semantically correct tool for each query
//   3. After receiving a tool result, the model produces a coherent response
//   4. Normal (non-tool) conversation turns remain coherent
//   5. Multi-turn context is maintained across tool-call / result / response
// ---------------------------------------------------------------------------

static const char *MODEL_THINKING =
    "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";

static const lfg_tool_desc TOOLS[] = {
    {"get_weather",
     "Get current weather forecast for a city or location. Returns temperature, conditions, and humidity.",
     R"({"type":"object","properties":{"location":{"type":"string","description":"City name"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]})"},
    {"send_email",
     "Send an email message to a recipient with subject and body.",
     R"({"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","subject","body"]})"},
    {"search_web",
     "Search the internet for information, articles, and current events.",
     R"({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]})"},
};
static const int32_t N_TOOLS = 3;

// Tool-call JSON schema: forces output like {"tool":"...","arguments":{...}}
static const char *TOOL_CALL_SCHEMA = R"({
    "type": "object",
    "properties": {
        "tool": { "type": "string" },
        "arguments": { "type": "object" }
    },
    "required": ["tool", "arguments"]
})";

struct test_env {
    lfg_model   *model   = nullptr;
    lfg_session *session = nullptr;
};

static bool setup(test_env *env, int n_ctx = 4096) {
    lfg_model_load_config lcfg = lfg_model_load_default_config();
    lcfg.model_path = MODEL_THINKING;
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

// Generate one turn: reset, optionally configure structured, chat_generate.
static std::string run_turn(test_env *env,
                            const std::vector<lfg_chat_message> &msgs,
                            bool structured, int max_tokens = 256) {
    lfg_session_reset(env->session);

    if (structured) {
        REQUIRE(lfg_session_configure_structured(env->session, TOOL_CALL_SCHEMA, nullptr));
    } else {
        // Clear structured constraint — empty grammar string removes it
        lfg_session_configure_structured(env->session, "", nullptr);
    }

    collect_state st;
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = max_tokens;
    gc.token_cb = collect_cb;
    gc.token_cb_data = &st;

    lfg_session_chat_generate(env->session, msgs.data(), msgs.size(), gc);

    return structured ? st.text : strip_thinking(st.text);
}

// ===========================================================================

TEST_CASE("Tool chat: weather lookup with tool call and natural follow-up") {
    test_env env;
    if (!setup(&env)) {
        MESSAGE("Skipping: thinking model not available");
        return;
    }

    REQUIRE(lfg_session_register_tools(env.session, TOOLS, N_TOOLS, 3) == N_TOOLS);

    std::vector<lfg_chat_message> history;
    std::vector<std::string> stored;  // keep strings alive

    history.push_back({"system",
        "You are a helpful assistant with access to tools. "
        "When you need information you don't have, call a tool by returning JSON "
        "with \"tool\" and \"arguments\" fields. When answering normally, respond "
        "in plain text."});

    // --- Turn 1: user asks about weather → expect tool call ---
    history.push_back({"user", "What is the weather like in Tokyo right now?"});

    std::string tool_call = run_turn(&env, history, /*structured=*/true, 128);
    MESSAGE("Turn 1 (tool call): ", tool_call);

    CHECK(!tool_call.empty());
    CHECK(tool_call[0] == '{');
    CHECK(contains(tool_call, "\"tool\""));
    CHECK(contains(tool_call, "\"arguments\""));
    // Model should pick get_weather for a weather question
    bool picked_weather = contains(tool_call, "get_weather") ||
                          contains(tool_call, "weather");
    CHECK_MESSAGE(picked_weather,
        "Expected get_weather tool for weather query, got: " << tool_call);

    stored.push_back(tool_call);
    history.push_back({"assistant", stored.back().c_str()});

    // --- Simulate tool result ---
    history.push_back({"user",
        "[Tool result from get_weather]: "
        "{\"location\":\"Tokyo\",\"temperature\":\"22°C\","
        "\"condition\":\"partly cloudy\",\"humidity\":\"65%\"}"});

    // --- Turn 2: model should summarize the weather naturally ---
    std::string weather_resp = run_turn(&env, history, /*structured=*/false, 256);
    MESSAGE("Turn 2 (weather response): ", weather_resp);

    CHECK(!weather_resp.empty());
    // Should mention the actual weather data
    bool mentions_data = contains(weather_resp, "22") ||
                         contains(weather_resp, "Tokyo") ||
                         contains(weather_resp, "cloud") ||
                         contains(weather_resp, "temperature") ||
                         contains(weather_resp, "weather");
    CHECK_MESSAGE(mentions_data,
        "Expected weather details in response, got: " << weather_resp);

    stored.push_back(weather_resp);
    history.push_back({"assistant", stored.back().c_str()});

    // --- Turn 3: normal question, no tool needed ---
    history.push_back({"user", "What is the capital of France?"});

    std::string normal_resp = run_turn(&env, history, /*structured=*/false, 128);
    MESSAGE("Turn 3 (normal): ", normal_resp);

    CHECK(!normal_resp.empty());
    CHECK_MESSAGE(contains(normal_resp, "Paris"),
        "Expected 'Paris' in answer about French capital, got: " << normal_resp);

    teardown(&env);
}

TEST_CASE("Tool chat: multi-tool conversation — weather, search, email") {
    test_env env;
    if (!setup(&env)) {
        MESSAGE("Skipping: thinking model not available");
        return;
    }

    REQUIRE(lfg_session_register_tools(env.session, TOOLS, N_TOOLS, 3) == N_TOOLS);

    std::vector<lfg_chat_message> history;
    std::vector<std::string> stored;

    history.push_back({"system",
        "You are an assistant with tools. When you need to use a tool, "
        "return a JSON object with \"tool\" (the tool name) and \"arguments\". "
        "Available tools: get_weather, search_web, send_email."});

    // --- Turn 1: weather tool call ---
    history.push_back({"user", "Check the weather in Paris."});
    std::string tc1 = run_turn(&env, history, true, 128);
    MESSAGE("Turn 1 (tool call): ", tc1);

    CHECK(tc1[0] == '{');
    CHECK(contains(tc1, "\"tool\""));
    bool has_weather_tool = contains(tc1, "get_weather") || contains(tc1, "weather");
    CHECK_MESSAGE(has_weather_tool, "Expected weather tool, got: " << tc1);

    stored.push_back(tc1);
    history.push_back({"assistant", stored.back().c_str()});

    // Simulate weather result
    history.push_back({"user",
        "[Tool result]: {\"temperature\":\"18°C\",\"condition\":\"rainy\",\"humidity\":\"80%\"}"});

    // --- Turn 2: model discusses weather (free-form) ---
    std::string resp1 = run_turn(&env, history, false, 256);
    MESSAGE("Turn 2 (weather discussion): ", resp1);
    CHECK(!resp1.empty());

    stored.push_back(resp1);
    history.push_back({"assistant", stored.back().c_str()});

    // --- Turn 3: search tool call ---
    history.push_back({"user", "Search for the best restaurants in Paris."});
    std::string tc2 = run_turn(&env, history, true, 128);
    MESSAGE("Turn 3 (tool call): ", tc2);

    CHECK(tc2[0] == '{');
    CHECK(contains(tc2, "\"tool\""));
    bool has_search_tool = contains(tc2, "search") || contains(tc2, "web");
    CHECK_MESSAGE(has_search_tool, "Expected search tool, got: " << tc2);

    stored.push_back(tc2);
    history.push_back({"assistant", stored.back().c_str()});

    // Simulate search result
    history.push_back({"user",
        "[Tool result]: {\"results\":[\"Le Comptoir du Panthéon - French bistro\","
        "\"L'Ambroisie - Michelin 3-star\",\"Chez Janou - classic French\"]}"});

    // --- Turn 4: model discusses restaurants ---
    std::string resp2 = run_turn(&env, history, false, 256);
    MESSAGE("Turn 4 (restaurant discussion): ", resp2);
    CHECK(!resp2.empty());

    stored.push_back(resp2);
    history.push_back({"assistant", stored.back().c_str()});

    // --- Turn 5: email tool call ---
    history.push_back({"user",
        "Send an email to alice@example.com with a summary of the weather "
        "and restaurant recommendations."});
    std::string tc3 = run_turn(&env, history, true, 256);
    MESSAGE("Turn 5 (tool call): ", tc3);

    CHECK(tc3[0] == '{');
    CHECK(contains(tc3, "\"tool\""));
    bool has_email_tool = contains(tc3, "send_email") || contains(tc3, "email");
    CHECK_MESSAGE(has_email_tool, "Expected email tool, got: " << tc3);
    // The email arguments should reference alice
    bool has_alice = contains(tc3, "alice");
    CHECK_MESSAGE(has_alice, "Expected 'alice' in email arguments, got: " << tc3);

    stored.push_back(tc3);
    history.push_back({"assistant", stored.back().c_str()});

    // Simulate email result
    history.push_back({"user", "[Tool result]: {\"status\":\"sent\",\"message_id\":\"msg-12345\"}"});

    // --- Turn 6: model confirms email sent ---
    std::string resp3 = run_turn(&env, history, false, 128);
    MESSAGE("Turn 6 (confirmation): ", resp3);
    CHECK(!resp3.empty());

    teardown(&env);
}

TEST_CASE("Tool chat: model selects correct tool for different queries") {
    test_env env;
    if (!setup(&env)) {
        MESSAGE("Skipping: thinking model not available");
        return;
    }

    REQUIRE(lfg_session_register_tools(env.session, TOOLS, N_TOOLS, 3) == N_TOOLS);

    struct tool_query {
        const char *user_msg;
        const char *expected_tool;  // substring to find in the tool call
    };

    tool_query queries[] = {
        {"What is the weather forecast in Berlin?", "weather"},
        {"Search the web for recent AI breakthroughs.", "search"},
        {"Send an email to bob@example.com about the meeting.", "email"},
        {"What is the temperature in New York today?", "weather"},
        {"Look up information about climate change.", "search"},
    };

    int correct = 0;
    int n = sizeof(queries) / sizeof(queries[0]);

    for (int i = 0; i < n; ++i) {
        std::vector<lfg_chat_message> msgs = {
            {"system",
             "You are a tool-calling assistant. Return a JSON object with "
             "\"tool\" and \"arguments\" to call the appropriate tool."},
            {"user", queries[i].user_msg},
        };

        std::string tc = run_turn(&env, msgs, true, 128);
        MESSAGE("Query ", i + 1, ": '", queries[i].user_msg, "'");
        MESSAGE("  Tool call: ", tc);

        CHECK(!tc.empty());
        CHECK(tc[0] == '{');

        bool matched = contains(tc, queries[i].expected_tool);
        if (matched) correct++;
        CHECK_MESSAGE(matched,
            "Query " << i + 1 << ": expected '" << queries[i].expected_tool
                     << "' in tool call, got: " << tc);
    }

    MESSAGE("=== Correct tool selections: ", correct, "/", n, " ===");
    CHECK(correct >= 3);  // Allow some flexibility for a 1.2B model

    teardown(&env);
}

TEST_CASE("Tool chat: 8-turn conversation with interleaved tool use and chat") {
    test_env env;
    if (!setup(&env)) {
        MESSAGE("Skipping: thinking model not available");
        return;
    }

    REQUIRE(lfg_session_register_tools(env.session, TOOLS, N_TOOLS, 3) == N_TOOLS);

    std::vector<lfg_chat_message> history;
    std::vector<std::string> stored;

    history.push_back({"system",
        "You are a travel planning assistant with access to tools. "
        "Use tools when you need real data. Answer normally for general questions."});

    int tool_calls_made = 0;
    int coherent_responses = 0;
    int total_turns = 0;

    // Helper lambda to run a tool-call turn
    auto do_tool_call = [&](const char *user_msg, const char *expect_tool,
                            const char *tool_result) {
        history.push_back({"user", user_msg});
        std::string tc = run_turn(&env, history, true, 128);
        total_turns++;

        MESSAGE("Turn ", total_turns, " (tool call): ", tc);
        CHECK(!tc.empty());
        CHECK(tc[0] == '{');

        if (contains(tc, expect_tool)) tool_calls_made++;
        if (!tc.empty()) coherent_responses++;

        stored.push_back(tc);
        history.push_back({"assistant", stored.back().c_str()});

        // Add tool result
        history.push_back({"user", tool_result});

        // Model response after tool result
        std::string resp = run_turn(&env, history, false, 256);
        total_turns++;

        MESSAGE("Turn ", total_turns, " (response): ", resp);
        if (!resp.empty()) coherent_responses++;

        // Output must be clean — no leaked stop/turn tokens
        CHECK_MESSAGE(!contains(resp, "<|im_end|>"),
            "Turn " << total_turns << " leaked <|im_end|>: " << resp);
        CHECK_MESSAGE(!contains(resp, "</|im_end|>"),
            "Turn " << total_turns << " leaked </|im_end|>: " << resp);
        CHECK_MESSAGE(!contains(resp, "<|im_start|>"),
            "Turn " << total_turns << " leaked <|im_start|>: " << resp);

        stored.push_back(resp);
        history.push_back({"assistant", stored.back().c_str()});
    };

    // Helper lambda to run a normal chat turn
    auto do_chat = [&](const char *user_msg, const char *expect_substr) {
        history.push_back({"user", user_msg});
        std::string resp = run_turn(&env, history, false, 256);
        total_turns++;

        MESSAGE("Turn ", total_turns, " (chat): ", resp);
        if (!resp.empty()) coherent_responses++;

        // Output must be clean — no leaked stop/turn tokens
        CHECK_MESSAGE(!contains(resp, "<|im_end|>"),
            "Turn " << total_turns << " leaked <|im_end|>: " << resp);
        CHECK_MESSAGE(!contains(resp, "</|im_end|>"),
            "Turn " << total_turns << " leaked </|im_end|>: " << resp);
        CHECK_MESSAGE(!contains(resp, "<|im_start|>"),
            "Turn " << total_turns << " leaked <|im_start|>: " << resp);

        if (expect_substr) {
            CHECK_MESSAGE(contains(resp, expect_substr),
                "Expected '" << expect_substr << "' in response: " << resp);
        }

        stored.push_back(resp);
        history.push_back({"assistant", stored.back().c_str()});
    };

    // --- Turn 1-2: weather tool call + response ---
    do_tool_call(
        "I'm planning a trip to Rome. What's the weather like there?",
        "weather",
        "[Tool result]: {\"temperature\":\"25°C\",\"condition\":\"sunny\","
        "\"humidity\":\"45%\",\"forecast\":\"Clear skies for the next 3 days\"}");

    // --- Turn 3: normal chat (no tool needed) ---
    do_chat("What are some must-see attractions in Rome?", nullptr);

    // --- Turn 4-5: search tool call + response ---
    do_tool_call(
        "Search for the best hotels near the Colosseum.",
        "search",
        "[Tool result]: {\"results\":[\"Hotel Palazzo Manfredi - 5 star, 50m from Colosseum\","
        "\"Hotel Capo d'Africa - 4 star, 200m\","
        "\"Residence Monti - boutique, 300m\"]}");

    // --- Turn 6: normal chat ---
    do_chat("Which hotel do you recommend and why?", nullptr);

    // --- Turn 7-8: email tool call + response ---
    do_tool_call(
        "Send an email to travel@example.com with my Rome itinerary including "
        "the weather, hotel, and attractions.",
        "email",
        "[Tool result]: {\"status\":\"sent\",\"to\":\"travel@example.com\"}");

    MESSAGE("=== Summary ===");
    MESSAGE("Total turns: ", total_turns);
    MESSAGE("Correct tool calls: ", tool_calls_made, "/3");
    MESSAGE("Coherent responses: ", coherent_responses, "/", total_turns);

    CHECK(total_turns == 8);
    CHECK(tool_calls_made >= 2);  // At least 2 of 3 tools picked correctly
    CHECK(coherent_responses >= 6);  // Most responses should be non-empty

    teardown(&env);
}

TEST_CASE("Tool chat: structured tool call contains valid arguments") {
    test_env env;
    if (!setup(&env)) {
        MESSAGE("Skipping: thinking model not available");
        return;
    }

    REQUIRE(lfg_session_register_tools(env.session, TOOLS, N_TOOLS, 3) == N_TOOLS);

    // Use a more specific schema to verify argument structure
    const char *weather_call_schema = R"({
        "type": "object",
        "properties": {
            "tool": { "type": "string" },
            "arguments": {
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"]
            }
        },
        "required": ["tool", "arguments"]
    })";

    std::vector<lfg_chat_message> msgs = {
        {"system",
         "You are a tool-calling assistant. To check weather, call get_weather "
         "with a location argument."},
        {"user", "What's the weather in San Francisco?"},
    };

    lfg_session_reset(env.session);
    REQUIRE(lfg_session_configure_structured(env.session, weather_call_schema, nullptr));

    collect_state st;
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 128;
    gc.token_cb = collect_cb;
    gc.token_cb_data = &st;

    lfg_session_chat_generate(env.session, msgs.data(), msgs.size(), gc);

    MESSAGE("Tool call: ", st.text);

    CHECK(!st.text.empty());
    CHECK(st.text[0] == '{');
    CHECK(contains(st.text, "\"tool\""));
    CHECK(contains(st.text, "\"arguments\""));
    CHECK(contains(st.text, "\"location\""));
    // The location should reference San Francisco
    bool has_sf = contains(st.text, "San Francisco") ||
                  contains(st.text, "san_francisco") ||
                  contains(st.text, "San_Francisco") ||
                  contains(st.text, "san francisco");
    CHECK_MESSAGE(has_sf, "Expected San Francisco in location argument: " << st.text);

    teardown(&env);
}
