// Benchmark: confidence monitor overhead vs baseline generation
// Measures tok/s across 4 configs: baseline, entropy-only, confidence-only, both.

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
    int n_predict = 128;
    int n_threads = 4;
    int warmup    = 1;
    int iters     = 5;
};

static bench_cfg parse_args(int argc, char **argv) {
    bench_cfg cfg;
    cfg.model_path = (argc > 1) ? argv[1] : "models/lfm2-350M.gguf";
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

struct run_result {
    double median_tps;
    double median_ms;
    int n_tokens;
    int n_confidence_spans;
};

static run_result bench_generate(
    lfg_model *model, const bench_cfg &cfg,
    bool enable_entropy, bool enable_confidence,
    int warmup, int iters)
{
    const lfg_vocab *vocab = lfg_model_get_vocab(model);
    const char *prompt = "The history of artificial intelligence begins with ancient myths and stories of artificial beings endowed with intelligence. The modern field was founded in 1956 at a conference at Dartmouth College.";
    int32_t prompt_len = (int32_t)std::strlen(prompt);

    std::vector<double> tps_samples;
    int last_n_tokens = 0;
    int last_confidence_spans = 0;

    for (int run = 0; run < warmup + iters; ++run) {
        lfg_session_config scfg = lfg_session_default_config();
        scfg.n_ctx = 2048;
        scfg.n_threads = cfg.n_threads;
        scfg.sampling.temp = 0.0f;
        lfg_session *session = lfg_session_create(model, &scfg);
        if (!session) { fprintf(stderr, "Failed to create session\n"); exit(1); }

        if (enable_entropy) {
            lfg_entropy_monitor_config ecfg = lfg_entropy_monitor_default_config();
            ecfg.threshold = 0.7f;
            ecfg.cooldown_tokens = 16;
            ecfg.ring_size = 4;
            lfg_session_configure_entropy_monitor(session, &ecfg);
        }

        if (enable_confidence) {
            lfg_confidence_monitor_config ccfg = lfg_confidence_monitor_default_config();
            ccfg.threshold = 0.3f;
            ccfg.min_span = 5;
            ccfg.ring_size = 4;
            lfg_session_configure_confidence_monitor(session, &ccfg);
        }

        // Tokenize + ingest prompt
        int32_t tok_cap = prompt_len + 16;
        std::vector<lfg_token> tokens(tok_cap);
        int32_t n = lfg_tokenize(vocab, prompt, prompt_len,
                                  tokens.data(), tok_cap, true, false);
        if (n <= 0) { fprintf(stderr, "Tokenization failed\n"); exit(1); }
        tokens.resize(n);
        lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true);

        // Time the generation
        lfg_generate_config gc = lfg_generate_default_config();
        gc.max_tokens = cfg.n_predict;

        auto t0 = std::chrono::high_resolution_clock::now();
        lfg_generate_result result = lfg_session_generate(session, gc);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (run >= warmup) {
            double tok_per_sec = (result.n_tokens > 0) ? (result.n_tokens / (ms / 1000.0)) : 0;
            tps_samples.push_back(tok_per_sec);
            last_n_tokens = result.n_tokens;
            last_confidence_spans = result.n_confidence_spans;
        }

        lfg_session_free(session);
    }

    run_result r;
    r.median_tps = median(tps_samples);
    r.median_ms = (last_n_tokens > 0) ? (last_n_tokens / r.median_tps * 1000.0) : 0;
    r.n_tokens = last_n_tokens;
    r.n_confidence_spans = last_confidence_spans;
    return r;
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

    printf("Model: %s\n", cfg.model_path.c_str());
    printf("Predict: %d tokens, Threads: %d, Warmup: %d, Iters: %d\n\n",
           cfg.n_predict, cfg.n_threads, cfg.warmup, cfg.iters);

    struct {
        const char *name;
        bool entropy;
        bool confidence;
    } configs[] = {
        {"Baseline (no monitors)",  false, false},
        {"Entropy only",            true,  false},
        {"Confidence only",         false, true},
        {"Entropy + Confidence",    true,  true},
    };

    printf("%-28s %8s %8s %8s %s\n",
           "Config", "tok/s", "ms", "tokens", "conf_spans");
    printf("%-28s %8s %8s %8s %s\n",
           "---", "---", "---", "---", "---");

    double baseline_tps = 0;

    for (auto &c : configs) {
        run_result r = bench_generate(model, cfg, c.entropy, c.confidence,
                                       cfg.warmup, cfg.iters);
        if (baseline_tps == 0) baseline_tps = r.median_tps;

        double overhead_pct = baseline_tps > 0
            ? ((baseline_tps - r.median_tps) / baseline_tps * 100.0)
            : 0;

        printf("%-28s %8.1f %8.1f %8d %6d",
               c.name, r.median_tps, r.median_ms, r.n_tokens, r.n_confidence_spans);
        if (c.entropy || c.confidence) {
            printf("    (%+.1f%%)", -overhead_pct);
        }
        printf("\n");
    }

    printf("\n");

    lfg_model_free(model);
    return 0;
}
