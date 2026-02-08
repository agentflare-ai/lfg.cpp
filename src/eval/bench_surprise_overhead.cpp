#include "../inference/lfg_api.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::vector<lfg_token> tokenize(const lfg_vocab *vocab, const char *text, bool add_bos) {
    int32_t len = (int32_t)std::strlen(text);
    std::vector<lfg_token> tokens(len + 16);
    int32_t n = lfg_tokenize(vocab, text, len, tokens.data(), (int32_t)tokens.size(), add_bos, false);
    if (n < 0) {
        tokens.resize(-n);
        n = lfg_tokenize(vocab, text, len, tokens.data(), (int32_t)tokens.size(), add_bos, false);
    }
    tokens.resize(n);
    return tokens;
}

struct bench_result {
    double ingest_ms;
    double generate_ms;
    int32_t n_prompt;
    int32_t n_generated;
    double prompt_tps;
    double gen_tps;
};

static bench_result run_trial(lfg_model *model, const lfg_vocab *vocab,
                               const lfg_token *prompt, int32_t n_prompt,
                               int32_t gen_tokens, bool enable_surprise) {
    bench_result r{};
    r.n_prompt = n_prompt;

    lfg_session_config cfg = lfg_session_default_config();
    cfg.n_ctx = 2048;
    cfg.n_batch = 512;
    cfg.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &cfg);

    if (enable_surprise) {
        lfg_surprise_monitor_config scfg = lfg_surprise_monitor_default_config();
        scfg.threshold = 0.3f;
        lfg_session_configure_surprise_monitor(session, &scfg);
    }

    // Benchmark prompt ingestion
    auto t0 = std::chrono::high_resolution_clock::now();
    lfg_session_ingest_tokens(session, prompt, n_prompt, true);
    auto t1 = std::chrono::high_resolution_clock::now();
    r.ingest_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.prompt_tps = (double)n_prompt / (r.ingest_ms / 1000.0);

    // Pop surprise event so it doesn't interfere with generation timing
    if (enable_surprise) {
        lfg_surprise_event sev;
        lfg_session_surprise_pop(session, &sev, nullptr, 0);
    }

    // Benchmark generation
    auto t2 = std::chrono::high_resolution_clock::now();
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = gen_tokens;
    lfg_generate_result gr = lfg_session_generate(session, gc);
    auto t3 = std::chrono::high_resolution_clock::now();
    r.generate_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    r.n_generated = gr.n_tokens;
    r.gen_tps = (r.n_generated > 0) ? (double)r.n_generated / (r.generate_ms / 1000.0) : 0;

    lfg_session_free(session);
    return r;
}

int main() {
    lfg_backend_init();

    lfg_model_load_config mcfg = lfg_model_load_default_config();
    const char *model_path = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
    mcfg.model_path = model_path;
    mcfg.n_gpu_layers = 0;
    lfg_model *model = lfg_load_model(&mcfg);
    if (!model) {
        std::fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    // Build prompts of varying lengths
    const char *base_text =
        "The transformer architecture was introduced in the paper Attention Is All You Need. "
        "It uses self-attention mechanisms to process sequences in parallel rather than sequentially. "
        "This enables much faster training on modern hardware like GPUs and TPUs. "
        "The key innovation is the multi-head attention mechanism which allows the model to attend "
        "to information from different representation subspaces at different positions. ";

    // Repeat to build longer prompts
    std::string short_text(base_text);
    std::string medium_text;
    std::string long_text;
    for (int i = 0; i < 4; i++) medium_text += base_text;
    for (int i = 0; i < 8; i++) long_text += base_text;

    struct test_case {
        const char *label;
        std::vector<lfg_token> tokens;
    };

    test_case cases[] = {
        {"short", tokenize(vocab, short_text.c_str(), true)},
        {"medium", tokenize(vocab, medium_text.c_str(), true)},
        {"long", tokenize(vocab, long_text.c_str(), true)},
    };

    const int warmup = 1;
    const int trials = 3;
    const int gen_tokens = 32;

    std::printf("%-8s %6s | %12s %12s %8s | %12s %12s %8s | %8s\n",
                "prompt", "tokens",
                "base_ingest", "surp_ingest", "overhead",
                "base_gen", "surp_gen", "gen_ovhd",
                "events");
    std::printf("%-8s %6s | %12s %12s %8s | %12s %12s %8s | %8s\n",
                "", "",
                "(tok/s)", "(tok/s)", "(%)",
                "(tok/s)", "(tok/s)", "(%)",
                "");
    std::printf("---------|--------|--------------|--------------|----------|--------------|--------------|----------|---------\n");

    for (auto &tc : cases) {
        const lfg_token *toks = tc.tokens.data();
        int32_t n = (int32_t)tc.tokens.size();

        // Warmup
        for (int w = 0; w < warmup; w++) {
            run_trial(model, vocab, toks, n, gen_tokens, false);
            run_trial(model, vocab, toks, n, gen_tokens, true);
        }

        // Trials
        double base_ingest_sum = 0, surp_ingest_sum = 0;
        double base_gen_sum = 0, surp_gen_sum = 0;
        int32_t surprise_events = 0;

        for (int t = 0; t < trials; t++) {
            auto base = run_trial(model, vocab, toks, n, gen_tokens, false);
            auto surp = run_trial(model, vocab, toks, n, gen_tokens, true);
            base_ingest_sum += base.prompt_tps;
            surp_ingest_sum += surp.prompt_tps;
            base_gen_sum += base.gen_tps;
            surp_gen_sum += surp.gen_tps;

            // Count events on last trial
            if (t == trials - 1) {
                lfg_session_config cfg = lfg_session_default_config();
                cfg.n_ctx = 2048;
                cfg.sampling.temp = 0.0f;
                lfg_session *s = lfg_session_create(model, &cfg);
                lfg_surprise_monitor_config scfg = lfg_surprise_monitor_default_config();
                scfg.threshold = 0.3f;
                lfg_session_configure_surprise_monitor(s, &scfg);
                lfg_session_ingest_tokens(s, toks, n, true);
                lfg_surprise_event sev;
                surprise_events = lfg_session_surprise_pop(s, &sev, nullptr, 0) ? 1 : 0;
                lfg_session_free(s);
            }
        }

        double base_ingest = base_ingest_sum / trials;
        double surp_ingest = surp_ingest_sum / trials;
        double base_gen = base_gen_sum / trials;
        double surp_gen = surp_gen_sum / trials;
        double ingest_overhead = ((base_ingest - surp_ingest) / base_ingest) * 100.0;
        double gen_overhead = ((base_gen - surp_gen) / base_gen) * 100.0;

        std::printf("%-8s %6d | %10.0f   %10.0f   %6.1f%%  | %10.0f   %10.0f   %6.1f%%  | %6d\n",
                    tc.label, n,
                    base_ingest, surp_ingest, ingest_overhead,
                    base_gen, surp_gen, gen_overhead,
                    surprise_events);
    }

    lfg_model_free(model);
    return 0;
}
