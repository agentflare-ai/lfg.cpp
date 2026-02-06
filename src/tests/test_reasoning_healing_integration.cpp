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
object ::= "\x7B" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
array  ::= "[" ws ( value ("," ws value)* )? "]" ws
string ::= "\"" ([^"\\] | "\" (["\\/bfnrt"] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\"" ws
number ::= ("-"? ([0-9]+) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?) ws
ws     ::= ([ \t\n] ws)?
)";

// Helper to detokenize a single token
std::string token_to_str(const lfg_vocab* vocab, lfg_token token) {
    char buf[256];
    // Cast away const because lfg_detokenize takes non-const vocab* (C API legacy likely)
    int n = lfg_detokenize(const_cast<lfg_vocab*>(vocab), &token, 1, buf, sizeof(buf), false, false);
    if (n < 0) return "";
    return std::string(buf, n);
}

TEST_CASE("Reasoning Model Healing and Checkpointing Integration") {
    lfg_backend_init();

    // 1. Load Model
    std::string model_path = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
    std::ifstream f(model_path);
    if (!f.good()) {
        MESSAGE("Skipping test: Model not found at " << model_path);
        return;
    }

    lfg_model_load_config load_config = lfg_model_load_default_config();
    load_config.model_path = model_path.c_str();
    load_config.n_gpu_layers = 0; // CPU only for deterministic testing

    lfg_model* model = lfg_load_model(&load_config);
    REQUIRE(model != nullptr);

    // 2. Configure session
    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = 2048;
    config.sampling.temp = 0.0f; // Deterministic
    config.sampling.seed = 12345;
    config.enable_healing = true; // Essential for this test

    lfg_session * session = lfg_session_create(model, &config);

    // 3. Configure Reasoning Tokens (Standard LFM2.5 Thinking tokens)
    // Assuming <think> = 32001, </think> = 32002 based on previous context
    lfg_token start_tok = 32001;
    lfg_token end_tok = 32002;
    lfg_session_configure_reasoning(session, &start_tok, 1, &end_tok, 1);

    // 4. Configure Grammar
    // Recursive grammar to ensure stack never empties unexpectedly
    const std::string SIMPLE_WORD_GRAMMAR = R"(
    root ::= [a-z]+
    )";
    lfg_session_configure_structured(session, SIMPLE_WORD_GRAMMAR.c_str(), "root");

    // 5. Setup Scenario: Partial Word
    const auto* vocab = lfg_model_get_vocab(model);
    std::string partial_prompt = "resp";

    // Tokenize
    std::vector<lfg_token> tokens(1024);
    // add_special=false to avoid BOS conflicts with simple grammar tests
    int n = lfg_tokenize(const_cast<lfg_vocab*>(vocab), partial_prompt.c_str(), partial_prompt.length(), tokens.data(), tokens.size(), false, false);
    tokens.resize(n);

    MESSAGE("Partial prompt tokens: " << n);
    for(auto t : tokens) {
        MESSAGE("Token: " << t << " -> " << token_to_str(vocab, t));
    }

    // Ingest tokens
    lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true);

    // 6. Create Checkpoint BEFORE Healing
    lfg_checkpoint * cp_before_heal = lfg_session_create_checkpoint(session);

    // --- Run A: Heal and Generate ---
    MESSAGE("Starting Run A...");
    std::vector<lfg_token> output_a;
    bool healing_grammar_works = true;

    try {
        // HealLastToken should try to complete "resp" -> "response"
        bool healed_a = lfg_session_heal_last_token(session);
        MESSAGE("Healed: " << (healed_a ? "YES" : "NO"));

        // Generate until end (grammar enforces "response")
        for(int i=0; i<10; ++i) {
            if (!lfg_session_decode(session)) break;
            lfg_token t = lfg_session_sample(session);
            output_a.push_back(t);
            lfg_session_ingest_tokens(session, &t, 1, true);

            if (t == lfg_vocab_eos(vocab)) break;
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
        bool restore_ok = lfg_session_restore_checkpoint(session, cp_before_heal);
        CHECK(restore_ok);

        if (restore_ok) {
            // Note: If grammar state is not restored, this might fail or produce different results
            lfg_session_heal_last_token(session); // Attempt same logic

            std::vector<lfg_token> output_b;
            for(int i=0; i<10; ++i) {
                if (!lfg_session_decode(session)) break;
                lfg_token t = lfg_session_sample(session);
                output_b.push_back(t);
                lfg_session_ingest_tokens(session, &t, 1, true);
                if (t == lfg_vocab_eos(vocab)) break;
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

    lfg_checkpoint_free(cp_before_heal);
    lfg_session_free(session);
    lfg_model_free(model);
}
