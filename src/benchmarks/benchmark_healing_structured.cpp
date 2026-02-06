#include "../inference/lfg_api.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <vector>
#include <numeric>

// Simple timer
class Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start_;
public:
    Timer() : start_(Clock::now()) {}
    double elapsed_ms() const {
        auto end = Clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};

const std::string JSON_GRAMMAR = R"(
root   ::= text object
text   ::= [^\x7B]*
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "\x7B" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
array  ::= "[" ws ( value ("," ws value)* )? "]" ws
string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\"" ws
number ::= ("-"? ([0-9]+) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?) ws
ws     ::= ([ \t\n] ws)?
)";

void run_benchmark(const std::string& model_path, int n_tokens_prompt, bool use_grammar, bool perform_healing) {
    lfg_model_load_config load_config = lfg_model_load_default_config();
    load_config.model_path = model_path.c_str();
    load_config.n_gpu_layers = 99; // Try to use GPU if available, or CPU

    lfg_model* model = lfg_load_model(&load_config);
    if (!model) {
        spdlog::error("Failed to load model: {}", model_path);
        return;
    }

    lfg_session_config config = lfg_session_default_config();
    config.n_ctx = n_tokens_prompt + 512;
    config.sampling.temp = 0.0f;
    config.enable_healing = true;

    lfg_session *session = lfg_session_create(model, &config);

    if (use_grammar) {
        lfg_session_configure_structured(session, JSON_GRAMMAR.c_str(), "root");
    }

    // Generate a prompt of n_tokens
    // Simple repetition of "The quick brown fox "
    std::string base_text = "The quick brown fox jumps over the lazy dog. ";
    std::string prompt;
    while (true) {
        auto current_logits = lfg_session_get_logits(session, nullptr, 0); // Dummy, just need vocab access? No, GetLogits uses ctx.
        // We can use model vocab directly
        // Just approximation.
        if (prompt.length() / 4 > (size_t)n_tokens_prompt) break;
        prompt += base_text;
    }

    // Ensure the prompt ends with a partial token that needs healing
    // "qu" -> "quick" or "quantum"
    prompt += "Output JSON: {\"key\": \"val"; // Ends with "val" -> expect "value"

    int n_tokens = lfg_tokenize(lfg_model_get_vocab(model), prompt.c_str(), prompt.length(), nullptr, 0, false, false);
    if (n_tokens < 0) n_tokens = -n_tokens;
    std::vector<lfg_token> token_vec(n_tokens);
    lfg_tokenize(lfg_model_get_vocab(model), prompt.c_str(), prompt.length(), token_vec.data(), n_tokens, false, false);

    // Ingest all but last few to simulate realistic state?
    // IngestTokens handles splitting if enable_healing is true.
    // It processes all-but-last, snapshots, then last.

    spdlog::info("Benchmarking [{}] [{}] Prompt: ~{} tokens.",
              (use_grammar ? "Grammar" : "No Grammar"),
              (perform_healing ? "Healing" : "No Healing"), n_tokens);

    Timer pp_timer;
    lfg_session_ingest_tokens(session, token_vec.data(), token_vec.size(), true);
    double pp_ms = pp_timer.elapsed_ms();

    // Print PP time for reference
    spdlog::info("  Prompt Processing: {} ms", pp_ms);

    Timer timer;
    bool healed = false;
    double latency = 0.0;

    if (perform_healing) {
        healed = lfg_session_heal_last_token(session);
        latency = timer.elapsed_ms();
    }

    // Measure decoding TPS
    int n_gen = 20;
    Timer tps_timer;
    for (int i = 0; i < n_gen; ++i) {
        lfg_session_decode(session);
        lfg_token t = lfg_session_sample(session);
        lfg_session_ingest_tokens(session, &t, 1, false);
    }
    double gen_ms = tps_timer.elapsed_ms();
    double tps = (n_gen / gen_ms) * 1000.0;

    if (perform_healing) {
        spdlog::info("  HealLastToken: {} ms. Healed: {}", latency, (healed ? "YES" : "NO"));
    } else {
        spdlog::info("  HealLastToken: N/A");
    }
    spdlog::info("  Decoding TPS: {} t/s", tps);

    lfg_session_free(session);
    lfg_model_free(model);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        spdlog::error("Usage: {} <model_path>", argv[0]);
        return 1;
    }
    std::string model_path = argv[1];

    // Short prompt
    run_benchmark(model_path, 100, false, true); // No Grammar, Healing
    run_benchmark(model_path, 100, true, true);  // Grammar, Healing
    run_benchmark(model_path, 100, true, false); // Grammar, No Healing

    // Medium prompt
    run_benchmark(model_path, 1000, false, true);
    run_benchmark(model_path, 1000, true, true);
    run_benchmark(model_path, 1000, true, false);

    // Long prompt (if possible/fast enough)
    // run_benchmark(model_path, 4000, false, true);
    // run_benchmark(model_path, 4000, true, true);

    return 0;
}
