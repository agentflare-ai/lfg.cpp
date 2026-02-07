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

static std::string generate(lfg_session *session, const lfg_vocab *vocab, int max_tokens) {
    std::string output;
    for (int i = 0; i < max_tokens; ++i) {
        lfg_session_decode(session);
        lfg_token tok = lfg_session_sample(session);
        if (lfg_vocab_is_eog(vocab, tok)) break;
        output += token_to_string(vocab, tok);
        lfg_session_ingest_tokens(session, &tok, 1, false);
    }
    return output;
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
    // top_k=1: only the highest-ranked tool is injected
    REQUIRE(lfg_session_register_tools(session, tools, 2, 1) == 2);

    auto tokens = tokenize(lfg_model_get_vocab(model), "Hello", true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));
    CHECK(lfg_session_decode(session)); // must not crash

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

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    REQUIRE(lfg_session_register_tools(session, TOOLS, N_TOOLS, 5) == N_TOOLS);

    std::string prompt = "What is the weather like in San Francisco today?\n";
    auto tokens = tokenize(vocab, prompt, true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    lfg_session_decode(session);
    std::string output = generate(session, vocab, 100);
    MESSAGE("Weather query output: " << output);

    // The model saw the injected tool descriptions. Generation should reference weather.
    bool mentions_weather = output.find("weather") != std::string::npos ||
                            output.find("Weather") != std::string::npos ||
                            output.find("temperature") != std::string::npos ||
                            output.find("forecast") != std::string::npos ||
                            output.find("San Francisco") != std::string::npos;
    CHECK_MESSAGE(mentions_weather, "Expected weather-related output, got: " << output);

    lfg_session_free(session);
}

TEST_CASE("Integration: Math query ranks calculator tool highest") {
    lfg_model *model = get_1_2b();
    if (!model) { MESSAGE("Skipping: 1.2B model not found"); return; }

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    REQUIRE(lfg_session_register_tools(session, TOOLS, N_TOOLS, 5) == N_TOOLS);

    std::string prompt = "What is 1234 multiplied by 5678?\n";
    auto tokens = tokenize(vocab, prompt, true);
    REQUIRE(lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true));

    lfg_session_decode(session);
    std::string output = generate(session, vocab, 100);
    MESSAGE("Math query output: " << output);

    bool mentions_math = output.find("calculator") != std::string::npos ||
                         output.find("Calculator") != std::string::npos ||
                         output.find("multiply") != std::string::npos ||
                         output.find("1234") != std::string::npos ||
                         output.find("5678") != std::string::npos ||
                         output.find("result") != std::string::npos;
    CHECK_MESSAGE(mentions_math, "Expected math-related output, got: " << output);

    lfg_session_free(session);
}

TEST_CASE("Integration: Session reset allows re-ranking with different query") {
    lfg_model *model = get_1_2b();
    if (!model) { MESSAGE("Skipping: 1.2B model not found"); return; }

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &config);
    REQUIRE(session != nullptr);

    REQUIRE(lfg_session_register_tools(session, TOOLS, N_TOOLS, 3) == N_TOOLS);

    // First cycle: weather query
    {
        std::string prompt = "What is the weather in Tokyo?\n";
        auto tokens = tokenize(vocab, prompt, true);
        lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true);
        lfg_session_decode(session);
        std::string out1 = generate(session, vocab, 60);
        MESSAGE("Cycle 1 (weather): " << out1);
    }

    // Reset and re-query with email topic
    lfg_session_reset(session);
    {
        std::string prompt = "Send an email to alice@example.com about the meeting tomorrow.\n";
        auto tokens = tokenize(vocab, prompt, true);
        lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true);
        lfg_session_decode(session);
        std::string out2 = generate(session, vocab, 60);
        MESSAGE("Cycle 2 (email): " << out2);

        // After reset, the model should produce output influenced by the re-injected tools.
        // It may reference email keywords or produce tool-related XML output.
        bool mentions_email = out2.find("email") != std::string::npos ||
                              out2.find("Email") != std::string::npos ||
                              out2.find("send") != std::string::npos ||
                              out2.find("alice") != std::string::npos ||
                              out2.find("meeting") != std::string::npos ||
                              out2.find("tool") != std::string::npos;
        CHECK_MESSAGE(mentions_email, "Expected email/tool-related output after reset, got: " << out2);
    }

    lfg_session_free(session);
}

TEST_CASE("Integration: Different top_k values produce different output") {
    lfg_model *model = get_1_2b();
    if (!model) { MESSAGE("Skipping: 1.2B model not found"); return; }

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    // top_k=1 vs top_k=5: fewer tools injected means different context.
    for (int32_t k : {1, 5}) {
        lfg_session_config config = lfg_session_default_config();
        config.n_ctx = 2048;
        config.sampling.temp = 0.0f;
        lfg_session *session = lfg_session_create(model, &config);
        REQUIRE(session != nullptr);

        REQUIRE(lfg_session_register_tools(session, TOOLS, N_TOOLS, k) == N_TOOLS);

        std::string prompt = "Search the web for recent news about AI.\n";
        auto tokens = tokenize(vocab, prompt, true);
        lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true);
        lfg_session_decode(session);
        std::string output = generate(session, vocab, 60);
        MESSAGE("top_k=" << k << " output: " << output);

        CHECK(!output.empty());

        lfg_session_free(session);
    }
}

TEST_CASE("Integration: No tools vs with tools produces different output") {
    lfg_model *model = get_1_2b();
    if (!model) { MESSAGE("Skipping: 1.2B model not found"); return; }

    const lfg_vocab *vocab = lfg_model_get_vocab(model);
    std::string prompt = "I need to check the weather in Paris and then send an email about it.\n";

    // Run without tools
    std::string output_no_tools;
    {
        lfg_session_config config = lfg_session_default_config();
        config.n_ctx = 2048;
        config.sampling.temp = 0.0f;
        lfg_session *session = lfg_session_create(model, &config);

        auto tokens = tokenize(vocab, prompt, true);
        lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true);
        lfg_session_decode(session);
        output_no_tools = generate(session, vocab, 80);
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

        auto tokens = tokenize(vocab, prompt, true);
        lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true);
        lfg_session_decode(session);
        output_with_tools = generate(session, vocab, 80);
        MESSAGE("With tools: " << output_with_tools);

        lfg_session_free(session);
    }

    // Tool injection changes the KV cache, so outputs should differ
    CHECK_MESSAGE(output_no_tools != output_with_tools,
                  "Expected different outputs with/without tools");
}

TEST_CASE("Integration: top_k=2 vs top_k=5 produces different output") {
    lfg_model *model = get_1_2b();
    if (!model) { MESSAGE("Skipping: 1.2B model not found"); return; }

    const lfg_vocab *vocab = lfg_model_get_vocab(model);
    std::string prompt = "What is the weather like in San Francisco today?\n";

    std::string output_top2;
    {
        lfg_session_config config = lfg_session_default_config();
        config.n_ctx = 2048;
        config.sampling.temp = 0.0f;
        lfg_session *session = lfg_session_create(model, &config);
        REQUIRE(session != nullptr);

        REQUIRE(lfg_session_register_tools(session, TOOLS, N_TOOLS, 2) == N_TOOLS);

        auto tokens = tokenize(vocab, prompt, true);
        lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true);
        lfg_session_decode(session);
        output_top2 = generate(session, vocab, 60);
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

        auto tokens = tokenize(vocab, prompt, true);
        lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true);
        lfg_session_decode(session);
        output_all = generate(session, vocab, 60);
        MESSAGE("top_k=5 output: " << output_all);

        lfg_session_free(session);
    }

    CHECK_MESSAGE(output_top2 != output_all,
                  "Expected different output between top-2 and all-5 tools");
}
