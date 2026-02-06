#include <spdlog/spdlog.h>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include "model_loader.h"
#include "inference_core.h"
#include "lfm_inference.h" // for lfm_tokenize
#include "json_schema_to_grammar.h"
#include <nlohmann/json.hpp>

class Timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start_;
public:
    Timer() : start_(clock::now()) {}
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - start_).count();
    }
};

// Weather Tool Schema
const std::string WEATHER_TOOL_SCHEMA = R"({
  "type": "object",
  "properties": {
    "tool_name": { "const": "get_weather" },
    "parameters": {
      "type": "object",
      "properties": {
        "location": { "type": "string" },
        "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
      },
      "required": ["location", "unit"]
    }
  },
  "required": ["tool_name", "parameters"]
})";

void run_benchmark(const std::string& model_path, bool enable_healing) {
    spdlog::info("==================================================");
    spdlog::info("Benchmark Config: Healing {}", (enable_healing ? "ENABLED" : "DISABLED"));

    liquid::ModelLoader::ModelConfig load_config;
    load_config.model_path = model_path;
    load_config.n_gpu_layers = 99;

    auto* model = liquid::ModelLoader::LoadModel(load_config);
    if (!model) {
        spdlog::error("Failed to load model.");
        return;
    }

    liquid::InferenceCore::Config config;
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f; // Deterministic
    config.enable_healing = enable_healing;
    
    liquid::InferenceCore core(model, config);
    
    // Convert Schema to GBNF (strict)
    auto json_schema = nlohmann::ordered_json::parse(WEATHER_TOOL_SCHEMA);
    std::string grammar = json_schema_to_grammar(json_schema, true);
    
    // spdlog::info("DEBUG: Grammar:\n{}", grammar);

    // The grammar has a "root" rule. We want to allow text before it.
    // Rename "root" to "json-root" (hyphen is valid)
    size_t root_pos = grammar.find("root ::= ");
    if (root_pos != std::string::npos) {
        grammar.replace(root_pos, 9, "json-root ::= ");
    }
    
    // Add new root rule that allows preamble
    // Regex includes: alphanumeric, dot, comma, colon, space, quote, underscore, apostrophe, newline, hyphen
    std::string preamble_grammar = 
        "root ::= preamble json-root\n"
        "preamble ::= [a-zA-Z0-9.,: \"_'\\n-]*\n" + grammar;
        
    // spdlog::info("DEBUG: Modified Grammar:\n{}", preamble_grammar);

    core.ConfigureStructuredDecoding(preamble_grammar, "root");

    // Prompt construction:
    // Preamble + Tool Call
    std::string prompt = "You are a helper. Call the get_weather tool for Paris in Celsius.\n"
                         "Tool Call: {\"tool_name\": \"get_weather\", \"parameters\": {\"location\": \"Paris\", \"unit\": \"cels";
    
    // Tokenize
    int n_tokens = lfm_tokenize(lfm_model_get_vocab(model), prompt.c_str(), prompt.length(), nullptr, 0, false, false);
    if (n_tokens < 0) n_tokens = -n_tokens;
    std::vector<lfm_token> tokens(n_tokens);
    lfm_tokenize(lfm_model_get_vocab(model), prompt.c_str(), prompt.length(), tokens.data(), n_tokens, false, false);

    spdlog::info("Prompt: {}", prompt);
    spdlog::info("Ingesting {} tokens...", tokens.size());

    // 1. Ingest
    Timer t_ingest;
    core.IngestTokens(tokens, true);
    spdlog::info("Ingest Time: {} ms", t_ingest.elapsed_ms());

    // 2. Heal (if enabled)
    bool healed = false;
    double heal_ms = 0.0;
    if (enable_healing) {
        Timer t_heal;
        healed = core.HealLastToken();
        heal_ms = t_heal.elapsed_ms();
        spdlog::info("HealLastToken: {} ms. Result: {}", heal_ms, (healed ? "YES" : "NO"));
    } else {
        spdlog::info("HealLastToken: SKIPPED");
    }

    // 3. Generate completion
    std::string generated_text;
    const auto* vocab = lfm_model_get_vocab(model);
    char buf[256];

    int n_gen = 10;
    Timer t_gen;
    for (int i = 0; i < n_gen; ++i) {
        core.Decode();
        auto token = core.Sample();
        
        int n = lfm_token_to_piece(vocab, token, buf, sizeof(buf), 0, true); // true = render specials?
        if (n > 0) generated_text += std::string(buf, n);
        
        core.IngestTokens({token}, false);
        
        // Stop on '}' closure of JSON
        if (generated_text.find("}}") != std::string::npos) break;
    }
    double gen_ms = t_gen.elapsed_ms();

    spdlog::info("Generated Text: [{}]", generated_text);
    spdlog::info("Generation Time: {} ms", gen_ms);
    
    liquid::ModelLoader::FreeModel(model);
}
int main(int argc, char** argv) {
    if (argc < 2) {
        spdlog::error("Usage: {} <model_path>", argv[0]);
        return 1;
    }
    std::string model_path = argv[1];

    // Case 1: Healing Enabled (Correct behavior)
    run_benchmark(model_path, true);

    // Case 2: Healing Disabled (Broken behavior)
    run_benchmark(model_path, false);

    return 0;
}
