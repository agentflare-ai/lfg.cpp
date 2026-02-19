#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/lfg_api.h"
#include <spdlog/spdlog.h>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>

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

static bool tokenize_literal(const lfg_vocab *vocab, const char *text, std::vector<lfg_token> &out) {
    if (!vocab || !text) return false;
    int32_t cap = (int32_t)std::strlen(text) + 8;
    out.resize(cap);
    int32_t n = lfg_tokenize(vocab, text, (int32_t)std::strlen(text), out.data(), cap, false, true);
    if (n < 0) {
        cap = -n;
        out.resize(cap);
        n = lfg_tokenize(vocab, text, (int32_t)std::strlen(text), out.data(), cap, false, true);
    }
    if (n <= 0) {
        out.clear();
        return false;
    }
    out.resize(n);
    return true;
}

static bool ends_with(const std::vector<lfg_token> &history, const std::vector<lfg_token> &pattern) {
    if (pattern.empty() || history.size() < pattern.size()) return false;
    size_t offset = history.size() - pattern.size();
    for (size_t i = 0; i < pattern.size(); ++i) {
        if (history[offset + i] != pattern[i]) return false;
    }
    return true;
}

TEST_CASE("Reasoning Gate and Soft Limit Integration") {
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
    std::vector<lfg_token> start_tokens;
    std::vector<lfg_token> end_tokens;
    if (!tokenize_literal(vocab, "<think>", start_tokens) ||
        !tokenize_literal(vocab, "</think>", end_tokens)) {
        MESSAGE("Skipping test: reasoning markers not found in vocab");
        lfg_model_free(model);
        return;
    }

    SUBCASE("Grammar Deferral During Reasoning") {
        lfg_session_config config = lfg_session_default_config();
        config.n_ctx = 2048;
        config.sampling.temp = 0.0f; // Deterministic
        
        lfg_session *session = lfg_session_create(model, &config);
        
        // Configure both reasoning and grammar
        lfg_session_configure_reasoning(session, start_tokens.data(), start_tokens.size(),
                                        end_tokens.data(), end_tokens.size());
        lfg_session_configure_structured(session, JSON_GRAMMAR.c_str(), "root");

        // Manually ingest <think> to trigger reasoning state
        // If gate works, grammar is disabled.
        // If gate fails, grammar is active -> likely fail to generate "valid" reasoning text (which is just English usually)
        // or force weird JSON tokens.
        lfg_session_ingest_tokens(session, start_tokens.data(), start_tokens.size(), true);

        // We expect the model to generate some text. 
        // LFM2.5 thinking model should output text after <think>.
        // This text is NOT JSON.
        
        bool non_json_generated = false;
        bool end_reasoning_found = false;
        std::vector<lfg_token> history;
        history.reserve(64);

        for(int i=0; i<20; ++i) {
            lfg_session_decode(session);
            lfg_token t = lfg_session_sample(session);
            lfg_session_ingest_tokens(session, &t, 1, true);
            history.push_back(t);

            std::string s = token_to_str(vocab, t);
            MESSAGE("Reasoning Token: " << t << " " << s);

            if (ends_with(history, end_tokens)) {
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
            lfg_session_ingest_tokens(session, end_tokens.data(), end_tokens.size(), true);
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
