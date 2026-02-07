#include "lfg_api.h"
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<lfg_token> tokenize(const lfg_vocab *vocab, const std::string &text, bool add_special) {
    std::vector<lfg_token> tokens(text.size() + 16);
    int32_t n = lfg_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), add_special, false);
    if (n < 0) { tokens.resize(-n); n = lfg_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), add_special, false); }
    tokens.resize(n);
    return tokens;
}

static std::string token_to_string(const lfg_vocab *vocab, lfg_token token) {
    char buf[256];
    int32_t n = lfg_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
    if (n < 0) return "";
    return std::string(buf, n);
}

using Clock = std::chrono::high_resolution_clock;

// The 5 tools used in all tests.
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

// Pre-formatted XML block with all 5 tools (what you'd inject without ranking).
static const char ALL_TOOLS_XML[] =
    "<tools>\n"
    "<tool name=\"get_weather\" description=\"Get current weather forecast for a city or location. Returns temperature, conditions, and humidity.\" schema='{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"City name, e.g. San Francisco\"},\"unit\":{\"type\":\"string\",\"enum\":[\"celsius\",\"fahrenheit\"]}},\"required\":[\"location\"]}'/>\n"
    "<tool name=\"calculator\" description=\"Perform arithmetic calculations. Supports add, subtract, multiply, divide.\" schema='{\"type\":\"object\",\"properties\":{\"operation\":{\"type\":\"string\",\"enum\":[\"add\",\"subtract\",\"multiply\",\"divide\"]},\"x\":{\"type\":\"number\"},\"y\":{\"type\":\"number\"}},\"required\":[\"operation\",\"x\",\"y\"]}'/>\n"
    "<tool name=\"search_web\" description=\"Search the internet for information, articles, and current events.\" schema='{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}},\"required\":[\"query\"]}'/>\n"
    "<tool name=\"send_email\" description=\"Send an email message to a recipient with subject and body.\" schema='{\"type\":\"object\",\"properties\":{\"to\":{\"type\":\"string\"},\"subject\":{\"type\":\"string\"},\"body\":{\"type\":\"string\"}},\"required\":[\"to\",\"subject\",\"body\"]}'/>\n"
    "<tool name=\"set_reminder\" description=\"Set a timed reminder or alarm that fires after a delay.\" schema='{\"type\":\"object\",\"properties\":{\"message\":{\"type\":\"string\"},\"delay_seconds\":{\"type\":\"number\"}},\"required\":[\"message\",\"delay_seconds\"]}'/>\n"
    "</tools>\n";

struct BenchResult {
    double register_ms;      // tool registration time
    double ttft_ms;          // time to first token (from after ingest to first sampled token)
    double gen_total_ms;     // total generation time for N tokens
    double gen_per_tok_ms;   // per-token generation speed
    int    n_generated;      // number of tokens generated
    int    prompt_tokens;    // prompt tokens ingested
    int    tool_tokens;      // tool tokens injected (0 for no-tools / manual)
};

static BenchResult run_no_tools(lfg_model *model, const lfg_vocab *vocab,
                                 const std::string &prompt, int gen_tokens) {
    BenchResult r = {};
    lfg_session_config sc = lfg_session_default_config();
    sc.n_ctx = 2048; sc.sampling.temp = 0.0f;
    auto *session = lfg_session_create(model, &sc);

    auto toks = tokenize(vocab, prompt, true);
    r.prompt_tokens = (int)toks.size();

    lfg_session_ingest_tokens(session, toks.data(), toks.size(), true);

    auto t0 = Clock::now();
    lfg_session_decode(session);
    lfg_token tok = lfg_session_sample(session);
    auto t1 = Clock::now();
    r.ttft_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Generate
    auto tg0 = Clock::now();
    for (int i = 0; i < gen_tokens; i++) {
        if (lfg_vocab_is_eog(vocab, tok)) break;
        lfg_session_ingest_tokens(session, &tok, 1, false);
        lfg_session_decode(session);
        tok = lfg_session_sample(session);
        r.n_generated++;
    }
    auto tg1 = Clock::now();
    r.gen_total_ms = std::chrono::duration<double, std::milli>(tg1 - tg0).count();
    r.gen_per_tok_ms = r.n_generated > 0 ? r.gen_total_ms / r.n_generated : 0;

    lfg_session_free(session);
    return r;
}

static BenchResult run_manual_inject(lfg_model *model, const lfg_vocab *vocab,
                                      const std::string &prompt, int gen_tokens) {
    BenchResult r = {};
    lfg_session_config sc = lfg_session_default_config();
    sc.n_ctx = 2048; sc.sampling.temp = 0.0f;
    auto *session = lfg_session_create(model, &sc);

    auto prompt_toks = tokenize(vocab, prompt, true);
    r.prompt_tokens = (int)prompt_toks.size();

    // Ingest prompt
    lfg_session_ingest_tokens(session, prompt_toks.data(), prompt_toks.size(), true);

    // Manually inject all tools (no ranking, no embedding computation)
    auto tool_toks = tokenize(vocab, ALL_TOOLS_XML, false);
    r.tool_tokens = (int)tool_toks.size();

    auto t0 = Clock::now();
    lfg_session_ingest_tokens(session, tool_toks.data(), tool_toks.size(), false);
    lfg_session_decode(session);
    lfg_token tok = lfg_session_sample(session);
    auto t1 = Clock::now();
    r.ttft_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Generate
    auto tg0 = Clock::now();
    for (int i = 0; i < gen_tokens; i++) {
        if (lfg_vocab_is_eog(vocab, tok)) break;
        lfg_session_ingest_tokens(session, &tok, 1, false);
        lfg_session_decode(session);
        tok = lfg_session_sample(session);
        r.n_generated++;
    }
    auto tg1 = Clock::now();
    r.gen_total_ms = std::chrono::duration<double, std::milli>(tg1 - tg0).count();
    r.gen_per_tok_ms = r.n_generated > 0 ? r.gen_total_ms / r.n_generated : 0;

    lfg_session_free(session);
    return r;
}

static BenchResult run_ranked(lfg_model *model, const lfg_vocab *vocab,
                               const std::string &prompt, int gen_tokens,
                               int32_t top_k, bool cold_cache) {
    BenchResult r = {};
    lfg_session_config sc = lfg_session_default_config();
    sc.n_ctx = 2048; sc.sampling.temp = 0.0f;
    auto *session = lfg_session_create(model, &sc);

    auto t_reg0 = Clock::now();
    lfg_session_register_tools(session, TOOLS, N_TOOLS, top_k);
    auto t_reg1 = Clock::now();
    r.register_ms = std::chrono::duration<double, std::milli>(t_reg1 - t_reg0).count();

    if (!cold_cache) {
        // Re-register to get cache-hit timing
        auto t_re0 = Clock::now();
        lfg_session_register_tools(session, TOOLS, N_TOOLS, top_k);
        auto t_re1 = Clock::now();
        r.register_ms = std::chrono::duration<double, std::milli>(t_re1 - t_re0).count();
    }

    auto prompt_toks = tokenize(vocab, prompt, true);
    r.prompt_tokens = (int)prompt_toks.size();

    lfg_session_ingest_tokens(session, prompt_toks.data(), prompt_toks.size(), true);

    auto t0 = Clock::now();
    lfg_session_decode(session);
    lfg_token tok = lfg_session_sample(session);
    auto t1 = Clock::now();
    r.ttft_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Generate
    auto tg0 = Clock::now();
    for (int i = 0; i < gen_tokens; i++) {
        if (lfg_vocab_is_eog(vocab, tok)) break;
        lfg_session_ingest_tokens(session, &tok, 1, false);
        lfg_session_decode(session);
        tok = lfg_session_sample(session);
        r.n_generated++;
    }
    auto tg1 = Clock::now();
    r.gen_total_ms = std::chrono::duration<double, std::milli>(tg1 - tg0).count();
    r.gen_per_tok_ms = r.n_generated > 0 ? r.gen_total_ms / r.n_generated : 0;

    lfg_session_free(session);
    return r;
}

static void print_result(const char *label, const BenchResult &r) {
    printf("  %-38s | reg %7.1f ms | TTFT %7.1f ms | gen %6.1f ms/tok (%d tok) | ctx: %d prompt + %d tools\n",
           label, r.register_ms, r.ttft_ms, r.gen_per_tok_ms, r.n_generated,
           r.prompt_tokens, r.tool_tokens);
}

int main() {
    lfg_backend_init();
    lfg_model_load_config cfg = lfg_model_load_default_config();
    cfg.model_path = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
    cfg.n_gpu_layers = 0;
    auto *model = lfg_load_model(&cfg);
    if (!model) { printf("FAIL: model not found\n"); return 1; }
    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    const int GEN = 50;

    const char *prompts[] = {
        "What is the weather like in San Francisco today?\n",
        "Search the web for recent news about artificial intelligence.\n",
        "Send an email to bob@example.com about the quarterly report.\n",
    };
    const char *labels[] = { "weather", "search", "email" };

    printf("E2E Tool Call Benchmark — 1.2B model, CPU, n_ctx=2048, gen=%d tokens\n", GEN);
    printf("=%.120s\n", "============================================================================================================================");

    for (int p = 0; p < 3; p++) {
        std::string prompt = prompts[p];
        printf("\nQuery: \"%s\"\n", labels[p]);
        printf("--%.120s\n", "----------------------------------------------------------------------------------------------------------------------------");

        auto r0 = run_no_tools(model, vocab, prompt, GEN);
        print_result("(A) No tools (baseline)", r0);

        auto r1 = run_manual_inject(model, vocab, prompt, GEN);
        print_result("(B) Manual inject all 5 tools", r1);

        auto r2 = run_ranked(model, vocab, prompt, GEN, 5, false);
        print_result("(C) Ranked top_k=5 (cache hit)", r2);

        auto r3 = run_ranked(model, vocab, prompt, GEN, 3, false);
        print_result("(D) Ranked top_k=3 (cache hit)", r3);

        auto r4 = run_ranked(model, vocab, prompt, GEN, 2, false);
        print_result("(E) Ranked top_k=2 (cache hit)", r4);

        auto r5 = run_ranked(model, vocab, prompt, GEN, 1, false);
        print_result("(F) Ranked top_k=1 (cache hit)", r5);

        // One cold-cache run per prompt set
        if (p == 0) {
            auto r6 = run_ranked(model, vocab, prompt, GEN, 3, true);
            print_result("(G) Ranked top_k=3 (cold cache)", r6);
        }

        printf("\n  TTFT overhead vs baseline:  B=%+.0f ms  C=%+.0f ms  D=%+.0f ms  E=%+.0f ms  F=%+.0f ms\n",
               r1.ttft_ms - r0.ttft_ms, r2.ttft_ms - r0.ttft_ms,
               r3.ttft_ms - r0.ttft_ms, r4.ttft_ms - r0.ttft_ms,
               r5.ttft_ms - r0.ttft_ms);
        printf("  Gen overhead vs baseline:   B=%+.1f%%  C=%+.1f%%  D=%+.1f%%  E=%+.1f%%  F=%+.1f%%\n",
               r0.gen_per_tok_ms > 0 ? 100.0 * (r1.gen_per_tok_ms - r0.gen_per_tok_ms) / r0.gen_per_tok_ms : 0,
               r0.gen_per_tok_ms > 0 ? 100.0 * (r2.gen_per_tok_ms - r0.gen_per_tok_ms) / r0.gen_per_tok_ms : 0,
               r0.gen_per_tok_ms > 0 ? 100.0 * (r3.gen_per_tok_ms - r0.gen_per_tok_ms) / r0.gen_per_tok_ms : 0,
               r0.gen_per_tok_ms > 0 ? 100.0 * (r4.gen_per_tok_ms - r0.gen_per_tok_ms) / r0.gen_per_tok_ms : 0,
               r0.gen_per_tok_ms > 0 ? 100.0 * (r5.gen_per_tok_ms - r0.gen_per_tok_ms) / r0.gen_per_tok_ms : 0);
    }

    lfg_model_free(model);
    return 0;
}
