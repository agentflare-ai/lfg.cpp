#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/lfg_api.h"
#include <string>
#include <vector>
#include <fstream>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string token_to_string(const lfg_vocab *vocab, lfg_token token) {
    char buf[256];
    int32_t n = lfg_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
    if (n < 0) return "";
    return std::string(buf, n);
}

static std::vector<lfg_token> tokenize(const lfg_vocab *vocab, const std::string &text, bool add_special) {
    std::vector<lfg_token> tokens(text.size() + 16);
    int32_t n = lfg_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), add_special, false);
    if (n < 0) {
        tokens.resize(-n);
        n = lfg_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), add_special, false);
    }
    tokens.resize(n);
    return tokens;
}

// ---------------------------------------------------------------------------
// 350M model — unit / mechanical tests (fast, always available)
// ---------------------------------------------------------------------------

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

TEST_CASE("Tool Registration") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    SUBCASE("Register tools returns correct count") {
        lfg_tool_desc tools[] = {
            {"get_weather", "Get current weather for a location", "{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}"},
            {"calculator", "Perform arithmetic operations", "{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"number\"},\"y\":{\"type\":\"number\"},\"op\":{\"type\":\"string\"}}}"},
            {"search_web", "Search the web for information", "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}},\"required\":[\"query\"]}"},
        };
        int32_t n = lfg_session_register_tools(session, tools, 3, 3);
        CHECK(n == 3);
    }

    SUBCASE("Register with null session returns -1") {
        lfg_tool_desc tools[] = { {"test", "test tool", nullptr} };
        CHECK(lfg_session_register_tools(nullptr, tools, 1, 1) == -1);
    }

    SUBCASE("Register with zero tools returns -1") {
        lfg_tool_desc tools[] = { {"test", "test tool", nullptr} };
        CHECK(lfg_session_register_tools(session, tools, 0, 1) == -1);
    }

    SUBCASE("Clear tools is idempotent") {
        lfg_tool_desc tools[] = { {"get_weather", "Get weather", nullptr} };
        CHECK(lfg_session_register_tools(session, tools, 1, 1) == 1);
        lfg_session_clear_tools(session);
        lfg_session_clear_tools(session); // safe to call twice
    }

    SUBCASE("Re-registration reuses embedding cache") {
        lfg_tool_desc tools[] = {
            {"get_weather", "Get weather for a location", "{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}}}"},
            {"calculator", "Do math", nullptr},
        };
        CHECK(lfg_session_register_tools(session, tools, 2, 2) == 2);
        CHECK(lfg_session_register_tools(session, tools, 2, 2) == 2); // cache hit
    }

    lfg_session_free(session);
}

TEST_CASE("Top-k constraint") {
    lfg_model *model = get_350m();
    REQUIRE(model != nullptr);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 512;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    lfg_tool_desc tools[] = {
        {"a", "A tool with a long description that takes many tokens", nullptr},
        {"b", "Another tool with a long description to exceed budget", nullptr},
    };
    // top_k=1: only the highest-ranked tool appears in output
    REQUIRE(lfg_session_register_tools(session, tools, 2, 1) == 2);

    // rank_tools should work without crashing
    const char *query = "Hello";
    int32_t needed = lfg_session_rank_tools(session, query, 5, nullptr, 0);
    CHECK(needed > 0);

    lfg_session_free(session);
}

// ---------------------------------------------------------------------------
// 1.2B Thinking model — real integration tests
// ---------------------------------------------------------------------------

static const char * MODEL_1_2B_PATH = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";

static lfg_model * g_1_2b = nullptr;

static lfg_model * get_1_2b() {
    if (!g_1_2b) {
        lfg_backend_init();
        std::ifstream f(MODEL_1_2B_PATH);
        if (!f.good()) return nullptr;
        lfg_model_load_config cfg = lfg_model_load_default_config();
        cfg.model_path = MODEL_1_2B_PATH;
        cfg.n_gpu_layers = 0;
        g_1_2b = lfg_load_model(&cfg);
    }
    return g_1_2b;
}

// The 5 tools used across integration tests.
static const lfg_tool_desc TOOLS[] = {
    {"get_weather",
     "Get current weather forecast for a city or location. Returns temperature, conditions, and humidity.",
     R"({"type":"object","properties":{"location":{"type":"string","description":"City name, e.g. San Francisco"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]})"},
    {"calculator",
     "Perform arithmetic calculations. Supports add, subtract, multiply, divide.",
     R"({"type":"object","properties":{"operation":{"type":"string","enum":["add","subtract","multiply","divide"]},"x":{"type":"number"},"y":{"type":"number"}},"required":["operation","x","y"]})"},
    {"search_web",
     "Search the internet for information, articles, and current events.",
     R"({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]})"},
    {"send_email",
     "Send an email message to a recipient with subject and body.",
     R"({"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","subject","body"]})"},
    {"set_reminder",
     "Set a timed reminder or alarm that fires after a delay.",
     R"({"type":"object","properties":{"message":{"type":"string"},"delay_seconds":{"type":"number"}},"required":["message","delay_seconds"]})"},
};
static const int32_t N_TOOLS = 5;

TEST_CASE("Integration: Weather query ranks weather tool highest") {
    lfg_model *model = get_1_2b();
    if (!model) { MESSAGE("Skipping: 1.2B model not found"); return; }

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    REQUIRE(lfg_session_register_tools(session, TOOLS, N_TOOLS, 5) == N_TOOLS);

    // Rank tools directly
    const char *query = "What is the weather like in San Francisco today?";
    int32_t needed = lfg_session_rank_tools(session, query, (int32_t)strlen(query), nullptr, 0);
    REQUIRE(needed > 0);

    std::vector<char> buf(needed + 1);
    int32_t written = lfg_session_rank_tools(session, query, (int32_t)strlen(query), buf.data(), (int32_t)buf.size());
    REQUIRE(written > 0);

    std::string result(buf.data(), written);
    MESSAGE("Weather ranking output: " << result);

    // get_weather should appear first in the formatted tool list
    auto weather_pos = result.find("get_weather");
    auto calc_pos = result.find("calculator");
    auto search_pos = result.find("search_web");
    CHECK_MESSAGE(weather_pos != std::string::npos, "Expected get_weather in output");
    if (weather_pos != std::string::npos && calc_pos != std::string::npos) {
        CHECK_MESSAGE(weather_pos < calc_pos, "get_weather should rank before calculator");
    }
    if (weather_pos != std::string::npos && search_pos != std::string::npos) {
        CHECK_MESSAGE(weather_pos < search_pos, "get_weather should rank before search_web");
    }

    lfg_session_free(session);
}

TEST_CASE("Integration: Math query ranks calculator tool highest") {
    lfg_model *model = get_1_2b();
    if (!model) { MESSAGE("Skipping: 1.2B model not found"); return; }

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    REQUIRE(lfg_session_register_tools(session, TOOLS, N_TOOLS, 5) == N_TOOLS);

    const char *query = "What is 1234 multiplied by 5678?";
    int32_t needed = lfg_session_rank_tools(session, query, (int32_t)strlen(query), nullptr, 0);
    REQUIRE(needed > 0);

    std::vector<char> buf(needed + 1);
    int32_t written = lfg_session_rank_tools(session, query, (int32_t)strlen(query), buf.data(), (int32_t)buf.size());
    REQUIRE(written > 0);

    std::string result(buf.data(), written);
    MESSAGE("Math ranking output: " << result);

    // calculator should appear first
    auto calc_pos = result.find("calculator");
    auto weather_pos = result.find("get_weather");
    CHECK_MESSAGE(calc_pos != std::string::npos, "Expected calculator in output");
    if (calc_pos != std::string::npos && weather_pos != std::string::npos) {
        CHECK_MESSAGE(calc_pos < weather_pos, "calculator should rank before get_weather");
    }

    lfg_session_free(session);
}

TEST_CASE("Integration: Session reset allows re-ranking with different query") {
    lfg_model *model = get_1_2b();
    if (!model) { MESSAGE("Skipping: 1.2B model not found"); return; }

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    REQUIRE(lfg_session_register_tools(session, TOOLS, N_TOOLS, 3) == N_TOOLS);

    // First ranking: weather query
    std::string result1;
    {
        const char *query = "What is the weather in Tokyo?";
        int32_t needed = lfg_session_rank_tools(session, query, (int32_t)strlen(query), nullptr, 0);
        REQUIRE(needed > 0);
        std::vector<char> buf(needed + 1);
        lfg_session_rank_tools(session, query, (int32_t)strlen(query), buf.data(), (int32_t)buf.size());
        result1 = std::string(buf.data());
        MESSAGE("Cycle 1 (weather): " << result1);
    }

    // Reset and re-query with email topic
    lfg_session_reset(session);
    std::string result2;
    {
        const char *query = "Send an email to alice@example.com about the meeting tomorrow.";
        int32_t needed = lfg_session_rank_tools(session, query, (int32_t)strlen(query), nullptr, 0);
        REQUIRE(needed > 0);
        std::vector<char> buf(needed + 1);
        lfg_session_rank_tools(session, query, (int32_t)strlen(query), buf.data(), (int32_t)buf.size());
        result2 = std::string(buf.data());
        MESSAGE("Cycle 2 (email): " << result2);
    }

    // After reset, different queries should produce different tool orderings
    // Weather query should have get_weather first, email query should have send_email first
    auto w1_pos = result1.find("get_weather");
    auto e1_pos = result1.find("send_email");
    auto w2_pos = result2.find("get_weather");
    auto e2_pos = result2.find("send_email");

    if (w1_pos != std::string::npos && e1_pos != std::string::npos) {
        CHECK_MESSAGE(w1_pos < e1_pos, "Weather query should rank get_weather before send_email");
    }
    if (e2_pos != std::string::npos && w2_pos != std::string::npos) {
        CHECK_MESSAGE(e2_pos < w2_pos, "Email query should rank send_email before get_weather");
    }

    lfg_session_free(session);
}

TEST_CASE("Integration: Different top_k values produce different output lengths") {
    lfg_model *model = get_1_2b();
    if (!model) { MESSAGE("Skipping: 1.2B model not found"); return; }

    // top_k=1 vs top_k=5: fewer tools means shorter formatted output
    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;

    const char *query = "Search the web for recent news about AI.";
    int32_t query_len = (int32_t)strlen(query);

    int32_t len_k1 = 0;
    {
        lfg_session *session = lfg_session_create(model, &config);
        REQUIRE(session != nullptr);
        REQUIRE(lfg_session_register_tools(session, TOOLS, N_TOOLS, 1) == N_TOOLS);
        len_k1 = lfg_session_rank_tools(session, query, query_len, nullptr, 0);
        MESSAGE("top_k=1 length: " << len_k1);
        CHECK(len_k1 > 0);
        lfg_session_free(session);
    }

    int32_t len_k5 = 0;
    {
        lfg_session *session = lfg_session_create(model, &config);
        REQUIRE(session != nullptr);
        REQUIRE(lfg_session_register_tools(session, TOOLS, N_TOOLS, 5) == N_TOOLS);
        len_k5 = lfg_session_rank_tools(session, query, query_len, nullptr, 0);
        MESSAGE("top_k=5 length: " << len_k5);
        CHECK(len_k5 > 0);
        lfg_session_free(session);
    }

    CHECK_MESSAGE(len_k1 < len_k5, "top_k=1 should produce shorter output than top_k=5");
}

TEST_CASE("Integration: No tools vs with tools via prompt_generate") {
    lfg_model *model = get_1_2b();
    if (!model) { MESSAGE("Skipping: 1.2B model not found"); return; }

    const char *prompt = "I need to check the weather in Paris and then send an email about it.";
    int32_t prompt_len = (int32_t)strlen(prompt);

    // Collect output via token callback
    struct cb_data { std::string output; };

    auto token_cb = [](lfg_token, const char *piece, int32_t piece_len, void *ud) -> lfg_generate_action {
        auto *d = (cb_data *)ud;
        d->output.append(piece, piece_len);
        return LFG_GENERATE_CONTINUE;
    };

    // Run without tools
    std::string output_no_tools;
    {
        lfg_session_config config = lfg_session_default_config();
        config.n_ctx = 2048;
        config.sampling.temp = 0.0f;
        lfg_session *session = lfg_session_create(model, &config);

        cb_data data;
        lfg_generate_config gcfg = lfg_generate_default_config();
        gcfg.max_tokens = 80;
        gcfg.token_cb = token_cb;
        gcfg.token_cb_data = &data;

        lfg_session_prompt_generate(session, prompt, prompt_len, true, gcfg);
        output_no_tools = data.output;
        MESSAGE("Without tools: " << output_no_tools);

        lfg_session_free(session);
    }

    // Run with tools
    std::string output_with_tools;
    {
        lfg_session_config config = lfg_session_default_config();
        config.n_ctx = 2048;
        config.sampling.temp = 0.0f;
        lfg_session *session = lfg_session_create(model, &config);

        lfg_session_register_tools(session, TOOLS, N_TOOLS, 3);

        cb_data data;
        lfg_generate_config gcfg = lfg_generate_default_config();
        gcfg.max_tokens = 80;
        gcfg.token_cb = token_cb;
        gcfg.token_cb_data = &data;

        lfg_session_prompt_generate(session, prompt, prompt_len, true, gcfg);
        output_with_tools = data.output;
        MESSAGE("With tools: " << output_with_tools);

        lfg_session_free(session);
    }

    // Tool injection changes the context, so outputs should differ
    CHECK_MESSAGE(output_no_tools != output_with_tools,
                  "Expected different outputs with/without tools");
}

TEST_CASE("Integration: top_k=2 vs top_k=5 via prompt_generate") {
    lfg_model *model = get_1_2b();
    if (!model) { MESSAGE("Skipping: 1.2B model not found"); return; }

    const char *prompt = "What is the weather like in San Francisco today?";
    int32_t prompt_len = (int32_t)strlen(prompt);

    struct cb_data { std::string output; };

    auto token_cb = [](lfg_token, const char *piece, int32_t piece_len, void *ud) -> lfg_generate_action {
        auto *d = (cb_data *)ud;
        d->output.append(piece, piece_len);
        return LFG_GENERATE_CONTINUE;
    };

    std::string output_top2;
    {
        lfg_session_config config = lfg_session_default_config();
        config.n_ctx = 2048;
        config.sampling.temp = 0.0f;
        lfg_session *session = lfg_session_create(model, &config);
        REQUIRE(session != nullptr);

        REQUIRE(lfg_session_register_tools(session, TOOLS, N_TOOLS, 2) == N_TOOLS);

        cb_data data;
        lfg_generate_config gcfg = lfg_generate_default_config();
        gcfg.max_tokens = 60;
        gcfg.token_cb = token_cb;
        gcfg.token_cb_data = &data;

        lfg_session_prompt_generate(session, prompt, prompt_len, true, gcfg);
        output_top2 = data.output;
        MESSAGE("top_k=2 output: " << output_top2);

        CHECK(!output_top2.empty());
        lfg_session_free(session);
    }

    std::string output_all;
    {
        lfg_session_config config = lfg_session_default_config();
        config.n_ctx = 2048;
        config.sampling.temp = 0.0f;
        lfg_session *session = lfg_session_create(model, &config);
        REQUIRE(session != nullptr);

        REQUIRE(lfg_session_register_tools(session, TOOLS, N_TOOLS, 5) == N_TOOLS);

        cb_data data;
        lfg_generate_config gcfg = lfg_generate_default_config();
        gcfg.max_tokens = 60;
        gcfg.token_cb = token_cb;
        gcfg.token_cb_data = &data;

        lfg_session_prompt_generate(session, prompt, prompt_len, true, gcfg);
        output_all = data.output;
        MESSAGE("top_k=5 output: " << output_all);

        lfg_session_free(session);
    }

    CHECK_MESSAGE(output_top2 != output_all,
                  "Expected different output between top-2 and all-5 tools");
}
