// Benchmark: KV cache context-length scaling
// Measures tok/s degradation as context fills from 512 to 8192 tokens.

#include "lfg_api.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

struct bench_cfg {
    std::string model_path;
    int n_predict = 64;
    int n_threads = 4;
    int warmup    = 2;
    int iters     = 5;
};

static bench_cfg parse_args(int argc, char **argv) {
    bench_cfg cfg;
    cfg.model_path = (argc > 1) ? argv[1] : "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--n-predict" && i + 1 < argc) cfg.n_predict = std::atoi(argv[++i]);
        else if (arg == "--n-threads" && i + 1 < argc) cfg.n_threads = std::atoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) cfg.warmup = std::atoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc) cfg.iters = std::atoi(argv[++i]);
    }
    return cfg;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? (v[n/2-1] + v[n/2]) / 2.0 : v[n/2];
}

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

// Build a prompt of approximately target_tokens length by repeating base text
static std::vector<lfg_token> build_prompt(const lfg_vocab *vocab, const char *base_text, int32_t target_tokens) {
    std::string text;
    // Over-generate text, tokenize, then truncate to exact count
    int repeats = (target_tokens / 20) + 2; // rough estimate: ~20 tokens per sentence
    for (int i = 0; i < repeats; i++) {
        text += base_text;
        text += " ";
    }
    auto tokens = tokenize(vocab, text.c_str(), true); // add BOS
    if ((int32_t)tokens.size() > target_tokens) {
        tokens.resize(target_tokens);
    }
    return tokens;
}

int main(int argc, char **argv) {
    bench_cfg cfg = parse_args(argc, argv);

    lfg_backend_init();

    lfg_model_load_config lcfg = lfg_model_load_default_config();
    lcfg.model_path = cfg.model_path.c_str();
    lcfg.n_gpu_layers = 0;
    lfg_model *model = lfg_load_model(&lcfg);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", cfg.model_path.c_str());
        return 1;
    }

    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    printf("=== Long Context Scaling Benchmark ===\n");
    printf("Model: %s\n", cfg.model_path.c_str());
    printf("Generate: %d tokens, Threads: %d, Warmup: %d, Iters: %d\n\n",
           cfg.n_predict, cfg.n_threads, cfg.warmup, cfg.iters);

    const char *base_text =
        "The transformer architecture was introduced in the paper Attention Is All You Need. "
        "It uses self-attention mechanisms to process sequences in parallel rather than sequentially. "
        "This enables much faster training on modern hardware like GPUs and TPUs. "
        "The key innovation is the multi-head attention mechanism which allows the model to attend "
        "to information from different representation subspaces at different positions. ";

    int context_depths[] = {512, 1024, 2048, 4096, 8192};
    int n_depths = sizeof(context_depths) / sizeof(context_depths[0]);

    printf("%-12s %8s %10s %10s %10s\n",
           "ctx_depth", "prompt", "tok/s", "ms", "degrade%");
    printf("%-12s %8s %10s %10s %10s\n",
           "---", "---", "---", "---", "---");

    double baseline_tps = 0;

    for (int d = 0; d < n_depths; d++) {
        int depth = context_depths[d];
        auto prompt = build_prompt(vocab, base_text, depth);
        int32_t n_prompt = (int32_t)prompt.size();

        // n_ctx must fit prompt + generated tokens
        int32_t n_ctx = depth + cfg.n_predict + 64;

        std::vector<double> tps_samples;
        int last_n_tokens = 0;

        for (int run = 0; run < cfg.warmup + cfg.iters; ++run) {
            lfg_session_config scfg = lfg_session_default_config();
            scfg.n_ctx = n_ctx;
            scfg.n_threads = cfg.n_threads;
            scfg.sampling.temp = 0.0f;
            lfg_session *session = lfg_session_create(model, &scfg);
            if (!session) {
                fprintf(stderr, "Failed to create session for ctx=%d\n", depth);
                break;
            }

            lfg_session_ingest_tokens(session, prompt.data(), n_prompt, true);

            lfg_generate_config gc = lfg_generate_default_config();
            gc.max_tokens = cfg.n_predict;

            auto t0 = std::chrono::high_resolution_clock::now();
            lfg_generate_result result = lfg_session_generate(session, gc);
            auto t1 = std::chrono::high_resolution_clock::now();

            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (run >= cfg.warmup) {
                double tok_per_sec = (result.n_tokens > 0) ? (result.n_tokens / (ms / 1000.0)) : 0;
                tps_samples.push_back(tok_per_sec);
                last_n_tokens = result.n_tokens;
            }

            lfg_session_free(session);
        }

        if (tps_samples.empty()) continue;

        double med_tps = median(tps_samples);
        double med_ms = (last_n_tokens > 0) ? (last_n_tokens / med_tps * 1000.0) : 0;

        if (d == 0) baseline_tps = med_tps;

        double degrade_pct = baseline_tps > 0
            ? ((baseline_tps - med_tps) / baseline_tps * 100.0)
            : 0;

        printf("%-12d %8d %10.1f %10.1f %9.1f%%\n",
               depth, n_prompt, med_tps, med_ms, degrade_pct);
    }

    printf("\n");
    lfg_model_free(model);
    return 0;
}
