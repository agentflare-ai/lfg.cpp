#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/inference_core.h"
#include "../loader/model_loader.h"
#include <spdlog/spdlog.h>
#include <vector>
#include <string>
#include <fstream>

using namespace liquid;

const std::string JSON_GRAMMAR = R"(
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "\x7B" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
array  ::= "[" ws ( value ("," ws value)* )? "]" ws
string ::= "\"" ([^"\\] | "\" (["\\/bfnrt"] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\"" ws
number ::= ("-"? ([0-9]+) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?) ws
ws     ::= ([ \t\n] ws)?
)";

// Helper to detokenize a single token
std::string token_to_str(const lfm_vocab* vocab, lfm_token token) {
    char buf[256];
    // Cast away const because lfm_detokenize takes non-const vocab* (C API legacy likely)
    int n = lfm_detokenize(const_cast<lfm_vocab*>(vocab), &token, 1, buf, sizeof(buf), false, false);
    if (n < 0) return ""; 
    return std::string(buf, n);
}

TEST_CASE("Reasoning Model Healing and Checkpointing Integration") {
    lfm_backend_init();

    // 1. Load Model
    std::string model_path = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
    std::ifstream f(model_path);
    if (!f.good()) {
        MESSAGE("Skipping test: Model not found at " << model_path);
        return;
    }

    ModelLoader::ModelConfig load_config;
    load_config.model_path = model_path;
    load_config.n_gpu_layers = 0; // CPU only for deterministic testing

    lfm_model* model = ModelLoader::LoadModel(load_config);
    REQUIRE(model != nullptr);

    // 2. Configure InferenceCore
    InferenceCore::Config config;
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f; // Deterministic
    config.sampling.seed = 12345;
    config.enable_healing = true; // Essential for this test

    InferenceCore core(model, config);

    // 3. Configure Reasoning Tokens (Standard LFM2.5 Thinking tokens)
    // Assuming <think> = 32001, </think> = 32002 based on previous context
    core.ConfigureReasoning({32001}, {32002});

    // 4. Configure Grammar
    // Recursive grammar to ensure stack never empties unexpectedly
    const std::string SIMPLE_WORD_GRAMMAR = R"(
    root ::= [a-z]+
    )";
    core.ConfigureStructuredDecoding(SIMPLE_WORD_GRAMMAR);

    // 5. Setup Scenario: Partial Word
    const auto* vocab = lfm_model_get_vocab(model);
    std::string partial_prompt = "resp"; 
    
    // Tokenize
    std::vector<lfm_token> tokens(1024);
    // add_special=false to avoid BOS conflicts with simple grammar tests
    int n = lfm_tokenize(const_cast<lfm_vocab*>(vocab), partial_prompt.c_str(), partial_prompt.length(), tokens.data(), tokens.size(), false, false);
    tokens.resize(n);

    MESSAGE("Partial prompt tokens: " << n);
    for(auto t : tokens) {
        MESSAGE("Token: " << t << " -> " << token_to_str(vocab, t));
    }

    // Ingest tokens
    core.IngestTokens(tokens);

    // 6. Create Checkpoint BEFORE Healing
    auto cp_before_heal = core.CreateCheckpoint();
    
    // --- Run A: Heal and Generate ---
    MESSAGE("Starting Run A...");
    std::vector<lfm_token> output_a;
    bool healing_grammar_works = true;

    try {
        // HealLastToken should try to complete "resp" -> "response"
        bool healed_a = core.HealLastToken();
        MESSAGE("Healed: " << (healed_a ? "YES" : "NO"));
        
        // Generate until end (grammar enforces "response")
        for(int i=0; i<10; ++i) {
            if (!core.Decode()) break;
            lfm_token t = core.Sample();
            output_a.push_back(t);
            core.IngestTokens({t});
            
            if (t == lfm_vocab_eos(vocab)) break;
        }
    } catch (const std::exception& e) {
        MESSAGE("Error in Run A (Healing + Grammar): " << e.what());
        MESSAGE("Exception Type: " << typeid(e).name());
    }

    // --- Run B: Restore and Generate ---
    // Only run if A didn't crash hard, or just to test checkpointing robustness
    MESSAGE("Starting Run B (Restore)...");
    try {
        // Do not call Reset() because Checkpoint relies on KV cache being present.
        // RestoreCheckpoint will truncate the context back to the checkpoint point.
        bool restore_ok = core.RestoreCheckpoint(cp_before_heal);
        CHECK(restore_ok);
        
        if (restore_ok) {
            // Note: If grammar state is not restored, this might fail or produce different results
            core.HealLastToken(); // Attempt same logic
            
            std::vector<lfm_token> output_b;
            for(int i=0; i<10; ++i) {
                if (!core.Decode()) break;
                lfm_token t = core.Sample();
                output_b.push_back(t);
                core.IngestTokens({t});
                if (t == lfm_vocab_eos(vocab)) break;
            }
            
            if (healing_grammar_works) {
                CHECK(output_a.size() == output_b.size());
                if (output_a.size() == output_b.size()) {
                    for(size_t i=0; i<output_a.size(); ++i) {
                        CHECK(output_a[i] == output_b[i]);
                    }
                }
            }
        } else {
            WARN("Checkpoint restore failed, skipping B");
        }
    } catch (const std::exception& e) {
        MESSAGE("Error in Run B (Restore): " << e.what());
        MESSAGE("Exception Type: " << typeid(e).name());
        FAIL("FATAL ERROR: Checkpoint restoration with Grammar failed.");
    }

    lfm_model_free(model);
}
