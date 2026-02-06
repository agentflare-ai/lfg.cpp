#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/inference_core.h"
#include "../inference/lfm_model.h"
#include "../loader/model_loader.h"
#include <fstream>
#include <spdlog/spdlog.h>
#include <vector>
#include <string>
#include <chrono>
#include "model_loader.h"
#include "inference_core.h"
#include "lfm_inference.h"

// Test Case: Verify that healing correctly recovers from partial tokens
// when using a complex grammar.
void run_integration_test(const std::string& model_path) {
    spdlog::info("Running Healing Integration Test...");
}

using namespace liquid;

// GBNF grammar that accepts a prefix text before the JSON
// Note: We use \x7B for '{' to avoid potential escaping issues in C++ raw strings or GBNF parser
const std::string HEALING_GRAMMAR = R"(
root   ::= text "\x7B" space key ":" space value "}"
text   ::= [^\x7B]*
key    ::= "\"function\""
value  ::= "\"search_database\""
space  ::= [ \t\n]*
)";

const std::string BOOLEAN_GRAMMAR = R"(
root   ::= text "\x7B" space key ":" space value "}"
text   ::= [^\x7B]*
key    ::= "\"valid\""
value  ::= "true" | "false"
space  ::= [ \t\n]*
)";

const std::string STRING_VALUE_GRAMMAR = R"(
root   ::= text "\x7B" space key ":" space value "}"
text   ::= [^\x7B]*
key    ::= "\"query\""
value  ::= "\"quantum physics\""
space  ::= [ \t\n]*
)";

TEST_CASE("Token Healing Integration with Structured Decoding") {
    lfm_backend_init();

    // Use the same model as other tests
    std::string model_path = "models/lfm2-350M.gguf";
    std::ifstream f(model_path);
    if (!f.good()) {
        MESSAGE("Model not found at " << model_path << ". Skipping test.");
        return;
    }

    ModelLoader::ModelConfig load_config;
    load_config.model_path = model_path;
    load_config.n_gpu_layers = 0;

    lfm_model* model = ModelLoader::LoadModel(load_config);
    REQUIRE(model != nullptr);

    InferenceCore::Config config;
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f; 
    config.enable_healing = true; // IMPORTANT
    InferenceCore core(model, config);

    SUBCASE("Heal Partial Key") {
        core.ConfigureStructuredDecoding(HEALING_GRAMMAR);

        std::string text_part = "Call the search_database function. Output JSON:\n";
        std::string json_start = "{\"" ;
        std::string partial_token = "fun"; 
        
        // 1. Ingest text part (checked against 'text' rule in grammar)
        auto tokens_text = model->vocab.tokenize(text_part, false); // No BOS
        if (!tokens_text.empty() && tokens_text[0] == model->vocab.token_bos()) {
            tokens_text.erase(tokens_text.begin());
        }
        core.IngestTokens(tokens_text, true);

        auto tokens_json_start = model->vocab.tokenize(json_start, false); 
        if (!tokens_json_start.empty() && tokens_json_start[0] == model->vocab.token_bos()) {
            tokens_json_start.erase(tokens_json_start.begin());
        }
        core.IngestTokens(tokens_json_start, true);

        auto tokens_partial = model->vocab.tokenize(partial_token, false);
        core.IngestTokens(tokens_partial, true);

        bool healed = core.HealLastToken();
        MESSAGE("Healing triggered: " << (healed ? "Yes" : "No"));
        
        std::string output_text;
        int max_tokens = 1;
        for (int i = 0; i < max_tokens; ++i) {
            core.Decode();
            lfm_token token = core.Sample();
            if (token == model->vocab.token_eos()) break;
            
            std::string piece = model->vocab.token_to_piece(token);
            output_text += piece;
            
            core.IngestTokens({token}, false);
        }
        
        MESSAGE("Generated suffix: " << output_text);
        
        bool healed_successfully = (output_text.find("ction") == std::string::npos);
        CHECK(healed_successfully);
        CHECK((output_text.find(":") != std::string::npos || output_text.find("\"") != std::string::npos));
    }

    SUBCASE("Heal Partial Boolean") {
        core.Reset(); // Ensure clean state
        core.ConfigureStructuredDecoding(BOOLEAN_GRAMMAR);

        std::string text_part = "Is it valid? Output JSON:\n";
        std::string json_start = "{\"valid\": "; 
        std::string partial_token = "fals"; 
        
        auto tokens_text = model->vocab.tokenize(text_part, false); 
        if (!tokens_text.empty() && tokens_text[0] == model->vocab.token_bos()) {
            tokens_text.erase(tokens_text.begin());
        }
        core.IngestTokens(tokens_text, true);

        auto tokens_json_start = model->vocab.tokenize(json_start, false);
        if (!tokens_json_start.empty() && tokens_json_start[0] == model->vocab.token_bos()) {
            tokens_json_start.erase(tokens_json_start.begin());
        }
        core.IngestTokens(tokens_json_start, true);

        auto tokens_partial = model->vocab.tokenize(partial_token, false);
        core.IngestTokens(tokens_partial, true);

        bool healed = core.HealLastToken();
        MESSAGE("Healing triggered (Boolean): " << (healed ? "Yes" : "No"));

        std::string output_text;
        int max_tokens = 5; // Generate enough to see "e" or closing brace
        for (int i = 0; i < max_tokens; ++i) {
            core.Decode();
            lfm_token token = core.Sample();
            if (token == model->vocab.token_eos()) break;
            output_text += model->vocab.token_to_piece(token);
            core.IngestTokens({token}, false); 
        }
        MESSAGE("Generated suffix (Boolean): " << output_text);

        CHECK(output_text.find("}") != std::string::npos);
    }

    SUBCASE("Heal Partial String Value") {
        core.Reset();
        core.ConfigureStructuredDecoding(STRING_VALUE_GRAMMAR);

        std::string text_part = "Query: quantum physics. JSON:\n";
        std::string json_start = "{\"query\": \""; 
        std::string partial_token = "quan"; 
        
        auto tokens_text = model->vocab.tokenize(text_part, false); 
        if (!tokens_text.empty() && tokens_text[0] == model->vocab.token_bos()) {
            tokens_text.erase(tokens_text.begin());
        }
        core.IngestTokens(tokens_text, true);

        auto tokens_json_start = model->vocab.tokenize(json_start, false);
        if (!tokens_json_start.empty() && tokens_json_start[0] == model->vocab.token_bos()) {
            tokens_json_start.erase(tokens_json_start.begin());
        }
        core.IngestTokens(tokens_json_start, true);

        auto tokens_partial = model->vocab.tokenize(partial_token, false);
        core.IngestTokens(tokens_partial, true);

        bool healed = core.HealLastToken();
        MESSAGE("Healing triggered (String): " << (healed ? "Yes" : "No"));

        std::string output_text;
        int max_tokens = 5;
        for (int i = 0; i < max_tokens; ++i) {
            core.Decode();
            lfm_token token = core.Sample();
            if (token == model->vocab.token_eos()) break;
            output_text += model->vocab.token_to_piece(token);
            core.IngestTokens({token}, false);
        }
        MESSAGE("Generated suffix (String): " << output_text);

        // "quan" -> "quantum physics"
        // Expect "tum" or "physics" in output.
        bool found_tum = output_text.find("tum") != std::string::npos;
        bool found_phys = output_text.find("phys") != std::string::npos;
        CHECK((found_tum || found_phys));
    }
    
    lfm_model_free(model);
}
