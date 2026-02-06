#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/lfg_api.h"
#include <spdlog/spdlog.h>
#include <vector>
#include <string>
#include <fstream>

// Helper to convert a token to a string piece using the C API
static std::string token_to_string(const lfg_vocab *vocab, lfg_token token) {
    char buf[256];
    int32_t n = lfg_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
    if (n < 0) return "";
    return std::string(buf, n);
}

// Helper to tokenize a string using the C API
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

// Define the tool call schema (JSON Schema) for the OUTPUT
const std::string TOOL_SCHEMA = R"({
    "type": "object",
    "properties": {
        "function": { "const": "get_weather" },
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    },
    "required": ["function", "parameters"]
})";
TEST_CASE("Tool Calling with LFM2-350M") {
    lfg_backend_init();

    // Specific model requested: LFM2-350M-GGUF Q4_K_M
    // We found 'lfm2-350M.gguf' in 'models/'. We assume this is the one.
    std::string model_path = "models/lfm2-350M.gguf";
    std::ifstream f(model_path);
    if (!f.good()) {
        MESSAGE("Model not found at " << model_path << ". Skipping test. Ensure LFM2-350M-GGUF is present in 'models/' directory.");
        return;
    }

    lfg_model_load_config load_config = lfg_model_load_default_config();
    load_config.model_path = model_path.c_str();
    load_config.n_gpu_layers = 0; // CPU for consistency

    lfg_model* model = lfg_load_model(&load_config);
    REQUIRE_MESSAGE(model != nullptr, "Failed to load model");

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f; // Deterministic output
    lfg_session *session = lfg_session_create(model, &config);

    SUBCASE("Weather Tool Call Generation") {
        // Configure the grammar for the tool
        lfg_session_configure_structured(session, TOOL_SCHEMA.c_str(), "root");

        // Simple prompt to trigger the tool
        // Note: Real tool use might require specific chat templates (e.g. <|user|> ... <|model|>)
        // We'll try a generic instruct format if we can guess it, or just raw text.
        // "Call the get_weather function for London in Celsius."
        // We'll append a newline to encourage generation.
        std::string prompt = "Call the get_weather function for London in Celsius.\n";

        auto tokens = tokenize(vocab, prompt, true);
        lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), false);

        // Decode loop
        std::string output_text;
        int max_tokens = 100;

        for (int i = 0; i < max_tokens; ++i) {
            lfg_session_decode(session);
            lfg_token token = lfg_session_sample(session);

            // Check for EOS
            if (token == lfg_vocab_eos(vocab)) {
                break;
            }

            // Decode token to string
            output_text += token_to_string(vocab, token);

            // Ingest simulated generation
            // Note: Sample() already calls lfg_sampler_accept, so we must NOT update sampler here
            lfg_session_ingest_tokens(session, &token, 1, false);
        }

        MESSAGE("Generated Output: " << output_text);

        // Validation
        // We expect a JSON object calling the function
        CHECK(output_text.find("get_weather") != std::string::npos);
        CHECK(output_text.find("location") != std::string::npos);
        CHECK(output_text.find("celsius") != std::string::npos);
    }

    SUBCASE("Calculator Tool Call Generation") {
        const std::string CALC_SCHEMA = R"({
            "type": "object",
            "properties": {
                "function": { "const": "calculator" },
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": { "enum": ["add", "subtract", "multiply", "divide"] },
                        "x": { "type": "number" },
                        "y": { "type": "number" }
                    },
                    "required": ["operation", "x", "y"]
                }
            },
            "required": ["function", "parameters"]
        })";

        lfg_session_configure_structured(session, CALC_SCHEMA.c_str(), "root");

        // Prompt
        std::string prompt = "Calculate 123 multiplied by 456.\n";
        auto tokens = tokenize(vocab, prompt, true);
        lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), false);

        std::string output_text;
        int max_tokens = 200;

        for (int i = 0; i < max_tokens; ++i) {
            lfg_session_decode(session);
            lfg_token token = lfg_session_sample(session);
            if (token == lfg_vocab_eos(vocab)) break;
            output_text += token_to_string(vocab, token);
            lfg_session_ingest_tokens(session, &token, 1, false);
        }

        MESSAGE("Generated Calculator Output: " << output_text);

        CHECK(output_text.find("calculator") != std::string::npos);
        CHECK(output_text.find("multiply") != std::string::npos);
        // Verify x and y fields exist in the output (exact values depend on model quality)
        CHECK(output_text.find("\"x\"") != std::string::npos);
        CHECK(output_text.find("\"y\"") != std::string::npos);
    }

    lfg_session_free(session);
    lfg_model_free(model);
}
