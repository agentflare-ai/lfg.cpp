// Benchmark: rewind-heavy entropy retrieval throughput
// Measures effective tok/s under different entropy thresholds that trigger rewinds.

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

// Fixed retrieval context returned on every entropy trigger
static const char *RETRIEVAL_CONTEXT =
    "Additional context: The attention mechanism computes a weighted sum of values "
    "where weights are determined by the compatibility of queries and keys. "
    "Multi-head attention runs multiple attention functions in parallel.";

struct rewind_config {
    const char *name;
    bool        enable;
    float       threshold;
    int32_t     cooldown;
};

struct run_result {
    double median_tps;
    double median_wall_ms;
    int    n_tokens;
    int    n_retrievals;
};

static run_result bench_rewind(
    lfg_model *model, const bench_cfg &cfg,
    const rewind_config &rcfg,
    int warmup, int iters)
{
    const lfg_vocab *vocab = lfg_model_get_vocab(model);
    const char *prompt = "The history of artificial intelligence begins with ancient myths and stories "
        "of artificial beings endowed with intelligence. The modern field was founded in 1956 at a "
        "conference at Dartmouth College.";
    int32_t prompt_len = (int32_t)std::strlen(prompt);

    std::vector<double> tps_samples;
    std::vector<double> wall_samples;
    int last_n_tokens = 0;
    int last_n_retrievals = 0;

    for (int run = 0; run < warmup + iters; ++run) {
        lfg_session_config scfg = lfg_session_default_config();
        scfg.n_ctx = 4096; // extra room for rewind injections
        scfg.n_threads = cfg.n_threads;
        scfg.sampling.temp = 0.0f;
        lfg_session *session = lfg_session_create(model, &scfg);
        if (!session) { fprintf(stderr, "Failed to create session\n"); exit(1); }

        if (rcfg.enable) {
            lfg_entropy_monitor_config ecfg = lfg_entropy_monitor_default_config();
            ecfg.threshold = rcfg.threshold;
            ecfg.cooldown_tokens = rcfg.cooldown;
            ecfg.ring_size = 8;
            lfg_session_configure_entropy_monitor(session, &ecfg);
        }

        // Tokenize + ingest prompt
        int32_t tok_cap = prompt_len + 16;
        std::vector<lfg_token> tokens(tok_cap);
        int32_t n = lfg_tokenize(vocab, prompt, prompt_len,
                                  tokens.data(), tok_cap, true, false);
        if (n <= 0) { fprintf(stderr, "Tokenization failed\n"); exit(1); }
        tokens.resize(n);
        lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true);

        int retrieval_count = 0;

        auto t0 = std::chrono::high_resolution_clock::now();
        lfg_generate_result result{};
        if (!rcfg.enable) {
            lfg_generate_config gc = lfg_generate_default_config();
            gc.max_tokens = cfg.n_predict;
            result = lfg_session_generate(session, gc);
        } else {
            std::vector<float> embd_buf(lfg_model_n_embd_out(model));
            int generated = 0;

            for (int i = 0; i < cfg.n_predict; ++i) {
                lfg_session_decode(session);
                lfg_token tok = lfg_session_sample(session);
                if (lfg_vocab_is_eog(vocab, tok)) {
                    break;
                }

                lfg_entropy_event ev;
                if (lfg_session_entropy_pop(session, &ev, embd_buf.data(), (int32_t)embd_buf.size())) {
                    if (lfg_session_rewind(session, ev.checkpoint_id)) {
                        int32_t len = (int32_t)std::strlen(RETRIEVAL_CONTEXT);
                        int32_t cap = len + 16;
                        std::vector<lfg_token> inj_toks((size_t)cap);
                        int32_t n_inj = lfg_tokenize(vocab, RETRIEVAL_CONTEXT, len,
                                                     inj_toks.data(), cap, false, false);
                        if (n_inj < 0) {
                            cap = -n_inj;
                            inj_toks.resize((size_t)cap);
                            n_inj = lfg_tokenize(vocab, RETRIEVAL_CONTEXT, len,
                                                 inj_toks.data(), cap, false, false);
                        }
                        if (n_inj > 0) {
                            lfg_session_ingest_tokens(session, inj_toks.data(), (size_t)n_inj, false);
                        }
                        retrieval_count++;
                        lfg_session_entropy_flush(session);
                        continue;
                    }
                    lfg_session_entropy_flush(session);
                } else {
                    lfg_session_entropy_flush(session);
                }

                lfg_session_ingest_tokens(session, &tok, 1, false);
                generated++;
            }

            result.n_tokens = generated;
            result.n_retrievals = retrieval_count;
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (run >= warmup) {
            double tok_per_sec = (result.n_tokens > 0) ? (result.n_tokens / (ms / 1000.0)) : 0;
            tps_samples.push_back(tok_per_sec);
            wall_samples.push_back(ms);
            last_n_tokens = result.n_tokens;
            last_n_retrievals = retrieval_count;
        }

        lfg_session_free(session);
    }

    run_result r;
    r.median_tps = median(tps_samples);
    r.median_wall_ms = median(wall_samples);
    r.n_tokens = last_n_tokens;
    r.n_retrievals = last_n_retrievals;
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

    printf("=== Rewind Throughput Benchmark ===\n");
    printf("Model: %s\n", cfg.model_path.c_str());
    printf("Generate: %d tokens, Threads: %d, Warmup: %d, Iters: %d\n\n",
           cfg.n_predict, cfg.n_threads, cfg.warmup, cfg.iters);

    rewind_config configs[] = {
        {"Baseline (no entropy)",  false, 0.0f, 0},
        {"Light rewind (t=0.5)",   true,  0.5f, 32},
        {"Heavy rewind (t=0.2)",   true,  0.2f, 8},
    };

    printf("%-26s %8s %10s %12s %12s\n",
           "Config", "tok/s", "rewinds", "wall_ms", "eff_tok/s");
    printf("%-26s %8s %10s %12s %12s\n",
           "---", "---", "---", "---", "---");

    double baseline_tps = 0;

    for (auto &rc : configs) {
        run_result r = bench_rewind(model, cfg, rc, cfg.warmup, cfg.iters);

        if (baseline_tps == 0) baseline_tps = r.median_tps;

        // Effective tok/s: output tokens / total wall time (includes rewind overhead)
        double eff_tps = (r.n_tokens > 0) ? (r.n_tokens / (r.median_wall_ms / 1000.0)) : 0;

        double overhead_pct = baseline_tps > 0
            ? ((baseline_tps - r.median_tps) / baseline_tps * 100.0)
            : 0;

        printf("%-26s %8.1f %10d %12.1f %12.1f",
               rc.name, r.median_tps, r.n_retrievals, r.median_wall_ms, eff_tps);
        if (rc.enable) {
            printf("    (%+.1f%%)", -overhead_pct);
        }
        printf("\n");
    }

    printf("\n");
    lfg_model_free(model);
    return 0;
}
