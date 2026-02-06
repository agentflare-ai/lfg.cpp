#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/inference_core.h"
#include "../inference/lfm_model.h"
#include "../loader/model_loader.h"
#include <spdlog/spdlog.h>
#include <vector>
#include <string>
#include <fstream>

using namespace liquid;

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
    lfm_backend_init();

    // Specific model requested: LFM2-350M-GGUF Q4_K_M
    // We found 'lfm2-350M.gguf' in 'models/'. We assume this is the one.
    std::string model_path = "models/lfm2-350M.gguf";
    std::ifstream f(model_path);
    if (!f.good()) {
        MESSAGE("Model not found at " << model_path << ". Skipping test. Ensure LFM2-350M-GGUF is present in 'models/' directory.");
        return;
    }

    ModelLoader::ModelConfig load_config;
    load_config.model_path = model_path;
    load_config.n_gpu_layers = 0; // CPU for consistency

    lfm_model* model = ModelLoader::LoadModel(load_config);
    REQUIRE_MESSAGE(model != nullptr, "Failed to load model");

    InferenceCore::Config config;
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f; // Deterministic output
    InferenceCore core(model, config);

    SUBCASE("Weather Tool Call Generation") {
        // Configure the grammar for the tool
        core.ConfigureStructuredDecoding(TOOL_SCHEMA);

        // Simple prompt to trigger the tool
        // Note: Real tool use might require specific chat templates (e.g. <|user|> ... <|model|>)
        // We'll try a generic instruct format if we can guess it, or just raw text.
        // "Call the get_weather function for London in Celsius."
        // We'll append a newline to encourage generation.
        std::string prompt = "Call the get_weather function for London in Celsius.\n";
        
        auto tokens = model->vocab.tokenize(prompt, true);
        core.IngestTokens(tokens, false);

        // Decode loop
        std::string output_text;
        int max_tokens = 100;
        
        for (int i = 0; i < max_tokens; ++i) {
            core.Decode();
            lfm_token token = core.Sample();
            
            // Check for EOS
            if (token == model->vocab.token_eos()) {
                break;
            }
            
            // Decode token to string (simplistic, might fail on partial multibyte)
            // Using vocab.token_to_piece
            output_text += model->vocab.token_to_piece(token);
            
            // Ingest simulated generation
            // Note: Sample() already calls lfm_sampler_accept, so we must NOT update sampler here
            core.IngestTokens({token}, false);
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

        core.ConfigureStructuredDecoding(CALC_SCHEMA);

        // Prompt
        std::string prompt = "Calculate 123 multiplied by 456.\n";
        auto tokens = model->vocab.tokenize(prompt, true);
        core.IngestTokens(tokens, false);

        std::string output_text;
        int max_tokens = 100;

        for (int i = 0; i < max_tokens; ++i) {
            core.Decode();
            lfm_token token = core.Sample();
            if (token == model->vocab.token_eos()) break;
            output_text += model->vocab.token_to_piece(token);
            core.IngestTokens({token}, false);
        }

        MESSAGE("Generated Calculator Output: " << output_text);

        CHECK(output_text.find("calculator") != std::string::npos);
        CHECK(output_text.find("multiply") != std::string::npos);
        // We verify x and y roughly. The model might handle numbers as strings or raw numbers depending on its training,
        // but the schema enforces "number" type.
        CHECK(output_text.find("123") != std::string::npos);
        CHECK(output_text.find("456") != std::string::npos);
    }
    
    lfm_model_free(model);
}
