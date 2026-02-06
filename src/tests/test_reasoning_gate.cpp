#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/lfg_api.h"
#include <spdlog/spdlog.h>
#include <vector>
#include <string>
#include <fstream>

const std::string JSON_GRAMMAR = R"(
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
array  ::= "[" ws ( value ("," ws value)* )? "]" ws
string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt"] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\"" ws
number ::= ("-"? ([0-9]+) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?) ws
ws     ::= ([ \t\n] ws)?
)";

// Helper to detokenize a single token
std::string token_to_str(const lfg_vocab* vocab, lfg_token token) {
    char buf[256];
    int n = lfg_detokenize(const_cast<lfg_vocab*>(vocab), &token, 1, buf, sizeof(buf), false, false);
    if (n < 0) return "";
    return std::string(buf, n);
}

TEST_CASE("Reasoning Gate and Budget Integration") {
    lfg_backend_init();

    std::string model_path = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
    std::ifstream f(model_path);
    if (!f.good()) {
        MESSAGE("Skipping test: Model not found at " << model_path);
        return;
    }

    lfg_model_load_config load_config = lfg_model_load_default_config();
    load_config.model_path = model_path.c_str();
    load_config.n_gpu_layers = 0; 

    lfg_model* model = lfg_load_model(&load_config);
    REQUIRE(model != nullptr);

    const auto* vocab = lfg_model_get_vocab(model);

    // Tokens
    lfg_token start_tok = 32001; // <think>
    lfg_token end_tok = 32002;   // </think>

    SUBCASE("Reasoning Budget Enforcement") {
        lfg_session_config config = lfg_session_default_config();
        config.n_ctx = 2048;
        config.sampling.temp = 0.0f;
        config.reasoning_budget = 5; // Limit reasoning to 5 tokens
        config.enable_healing = false; 

        lfg_session *session = lfg_session_create(model, &config);
        lfg_session_configure_reasoning(session, &start_tok, 1, &end_tok, 1);

        // Start with <think>
        lfg_session_ingest_tokens(session, &start_tok, 1, true);

        int count = 0;
        bool forced_end = false;
        for(int i=0; i<20; ++i) {
            lfg_session_decode(session);
            lfg_token t = lfg_session_sample(session);
            lfg_session_ingest_tokens(session, &t, 1, true);
            
            MESSAGE("Token: " << t << " " << token_to_str(vocab, t));

            if (t == end_tok) {
                forced_end = true;
                // Budget 5 means: 0,1,2,3,4 generated. 5th generated token is forced to be </think> if not already.
                // Or rather, if count >= budget, force next.
                // If budget is 5.
                // Gen 1 (count=1).
                // ...
                // Gen 5 (count=5).
                // Next sample sees count=5 >= budget -> Force.
                // So 6th token is </think>.
                CHECK(count == 5); 
                break;
            }
            count++;
        }
        CHECK(forced_end);
        lfg_session_free(session);
    }

    SUBCASE("Grammar Deferral During Reasoning") {
        lfg_session_config config = lfg_session_default_config();
        config.n_ctx = 2048;
        config.sampling.temp = 0.0f; // Deterministic
        config.reasoning_budget = 0; // Unlimited
        
        lfg_session *session = lfg_session_create(model, &config);
        
        // Configure both reasoning and grammar
        lfg_session_configure_reasoning(session, &start_tok, 1, &end_tok, 1);
        lfg_session_configure_structured(session, JSON_GRAMMAR.c_str(), "root");

        // Manually ingest <think> to trigger reasoning state
        // If gate works, grammar is disabled.
        // If gate fails, grammar is active -> likely fail to generate "valid" reasoning text (which is just English usually)
        // or force weird JSON tokens.
        lfg_session_ingest_tokens(session, &start_tok, 1, true);

        // We expect the model to generate some text. 
        // LFM2.5 thinking model should output text after <think>.
        // This text is NOT JSON.
        
        bool non_json_generated = false;
        bool end_reasoning_found = false;

        for(int i=0; i<20; ++i) {
            lfg_session_decode(session);
            lfg_token t = lfg_session_sample(session);
            lfg_session_ingest_tokens(session, &t, 1, true);

            std::string s = token_to_str(vocab, t);
            MESSAGE("Reasoning Token: " << t << " " << s);

            if (t == end_tok) {
                end_reasoning_found = true;
                break;
            }

            // Simple check: if we see a token that is definitely not start of JSON (like "The", "I", etc)
            // The grammar forces root ::= object, so it MUST start with "{" (ignoring whitespace).
            // If we generate something that doesn't contain "{", it's likely reasoning text.
            if (s.find("{") == std::string::npos) {
                non_json_generated = true;
            }
        }

        // If we didn't crash and generated non-JSON, success!
        CHECK(non_json_generated);
        
        if (!end_reasoning_found) {
            // Force end token to test grammar re-activation
            lfg_session_ingest_tokens(session, &end_tok, 1, true);
        }

        // Now grammar should be active. Next token MUST be valid JSON start.
        lfg_session_decode(session);
        lfg_token json_t = lfg_session_sample(session);
        std::string json_s = token_to_str(vocab, json_t);
        MESSAGE("JSON Start Token: " << json_t << " " << json_s);
        
        // Allowed starts for our grammar: { [ " - digit t f n
        bool valid_json_start = (json_s.find_first_of("{[\"-0123456789tfn") != std::string::npos);
        CHECK(valid_json_start);

        lfg_session_free(session);
    }

    lfg_model_free(model);
}
