// In-process performance comparison: lfg.cpp vs vendored llama.cpp
// Both engines compiled by the same Zig compiler, driven from a single binary.

#include "lfg_inference.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>

struct bench_config {
    std::string model_path;
    int n_predict   = 64;
    int n_threads   = 4;
    int warmup      = 1;
    int iters       = 3;
    int ngl         = 0;
    bool no_parity  = false;
    bool quiet      = false;
};

struct bench_result {
    std::vector<double> gen_tps;       // tokens/sec per iteration
    std::vector<double> total_ms;      // total wall ms per iteration
    std::vector<int32_t> tokens;       // token IDs from last iteration
    int n_generated = 0;
};

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 0) {
        return (v[n / 2 - 1] + v[n / 2]) / 2.0;
    }
    return v[n / 2];
}

static bench_config parse_args(int argc, char ** argv) {
    bench_config cfg;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path> [--n-predict N] [--n-threads N] [--warmup N] [--iters N] [--ngl N] [--no-parity] [--quiet]\n", argv[0]);
        exit(1);
    }

    cfg.model_path = argv[1];

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--n-predict" && i + 1 < argc) { cfg.n_predict = std::stoi(argv[++i]); }
        else if (arg == "--n-threads" && i + 1 < argc) { cfg.n_threads = std::stoi(argv[++i]); }
        else if (arg == "--warmup" && i + 1 < argc) { cfg.warmup = std::stoi(argv[++i]); }
        else if (arg == "--iters" && i + 1 < argc) { cfg.iters = std::stoi(argv[++i]); }
        else if (arg == "--ngl" && i + 1 < argc) { cfg.ngl = std::stoi(argv[++i]); }
        else if (arg == "--no-parity") { cfg.no_parity = true; }
        else if (arg == "--quiet") { cfg.quiet = true; }
        else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            exit(1);
        }
    }

    return cfg;
}

// ---------------------------------------------------------------------------
// run_llama: vendored llama.cpp path
// ---------------------------------------------------------------------------
static bench_result run_llama(const bench_config & cfg) {
    bench_result result;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = cfg.ngl;

    llama_model * model = llama_model_load_from_file(cfg.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "llama: failed to load model\n");
        exit(1);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = 2048;
    ctx_params.n_batch   = 512;
    ctx_params.n_threads = cfg.n_threads;
    ctx_params.no_perf   = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "llama: failed to create context\n");
        exit(1);
    }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_token bos = llama_vocab_bos(vocab);

    int total_iters = cfg.warmup + cfg.iters;

    for (int iter = 0; iter < total_iters; iter++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_sampler_reset(smpl);

        llama_batch batch = llama_batch_get_one(&bos, 1);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "llama: decode failed\n");
            exit(1);
        }

        std::vector<int32_t> iter_tokens;
        int n_gen = 0;

        auto t0 = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < cfg.n_predict; i++) {
            llama_token tok = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, tok)) break;

            iter_tokens.push_back(tok);
            n_gen++;

            batch = llama_batch_get_one(&tok, 1);
            if (llama_decode(ctx, batch)) break;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (iter >= cfg.warmup) {
            double tps = (n_gen > 0 && ms > 0) ? n_gen / (ms / 1000.0) : 0.0;
            result.gen_tps.push_back(tps);
            result.total_ms.push_back(ms);
            result.tokens = iter_tokens;
            result.n_generated = n_gen;
        }
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return result;
}

// ---------------------------------------------------------------------------
// run_lfg: lfg.cpp path
// ---------------------------------------------------------------------------
static bench_result run_lfg(const bench_config & cfg) {
    bench_result result;

    lfg_model_params model_params = lfg_model_default_params();
    model_params.n_gpu_layers = cfg.ngl;

    lfg_model * model = lfg_model_load_from_file(cfg.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "lfg: failed to load model\n");
        exit(1);
    }

    const lfg_vocab * vocab = lfg_model_get_vocab(model);

    lfg_context_params ctx_params = lfg_context_default_params();
    ctx_params.n_ctx     = 2048;
    ctx_params.n_batch   = 512;
    ctx_params.n_threads = cfg.n_threads;
    ctx_params.no_perf   = true;

    lfg_context * ctx = lfg_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "lfg: failed to create context\n");
        exit(1);
    }

    auto sparams = lfg_sampler_chain_default_params();
    lfg_sampler * smpl = lfg_sampler_chain_init(sparams);
    lfg_sampler_chain_add(smpl, lfg_sampler_init_greedy());

    lfg_token bos = lfg_vocab_bos(vocab);

    int total_iters = cfg.warmup + cfg.iters;

    for (int iter = 0; iter < total_iters; iter++) {
        lfg_memory_clear(lfg_get_memory(ctx), true);
        lfg_sampler_reset(smpl);

        lfg_batch batch = lfg_batch_get_one(&bos, 1);
        if (lfg_decode(ctx, batch)) {
            fprintf(stderr, "lfg: decode failed\n");
            exit(1);
        }

        std::vector<int32_t> iter_tokens;
        int n_gen = 0;

        auto t0 = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < cfg.n_predict; i++) {
            lfg_token tok = lfg_sampler_sample(smpl, ctx, -1);
            if (lfg_vocab_is_eog(vocab, tok)) break;

            iter_tokens.push_back(tok);
            n_gen++;

            batch = lfg_batch_get_one(&tok, 1);
            if (lfg_decode(ctx, batch)) break;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (iter >= cfg.warmup) {
            double tps = (n_gen > 0 && ms > 0) ? n_gen / (ms / 1000.0) : 0.0;
            result.gen_tps.push_back(tps);
            result.total_ms.push_back(ms);
            result.tokens = iter_tokens;
            result.n_generated = n_gen;
        }
    }

    lfg_sampler_free(smpl);
    lfg_free(ctx);
    lfg_model_free(model);

    return result;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    bench_config cfg = parse_args(argc, argv);

    ggml_backend_load_all();

    if (!cfg.quiet) {
        fprintf(stderr, "Running llama.cpp benchmark...\n");
    }
    bench_result llama_res = run_llama(cfg);

    if (!cfg.quiet) {
        fprintf(stderr, "Running lfg.cpp benchmark...\n");
    }
    bench_result lfg_res = run_lfg(cfg);

    // Extract file basename from model path
    std::string model_name = cfg.model_path;
    size_t slash = model_name.rfind('/');
    if (slash != std::string::npos) model_name = model_name.substr(slash + 1);

    double llama_tps_med = median(llama_res.gen_tps);
    double lfg_tps_med   = median(lfg_res.gen_tps);
    double llama_ms_med  = median(llama_res.total_ms);
    double lfg_ms_med    = median(lfg_res.total_ms);

    double pct_diff = (llama_ms_med > 0)
        ? ((lfg_ms_med - llama_ms_med) / llama_ms_med) * 100.0
        : 0.0;

    // Parity check
    bool parity_pass = true;
    int parity_match = 0;
    int parity_total = 0;

    if (!cfg.no_parity) {
        parity_total = std::min(llama_res.tokens.size(), lfg_res.tokens.size());
        for (int i = 0; i < parity_total; i++) {
            if (llama_res.tokens[i] == lfg_res.tokens[i]) {
                parity_match++;
            } else {
                parity_pass = false;
            }
        }
        if (llama_res.tokens.size() != lfg_res.tokens.size()) {
            parity_pass = false;
        }
    }

    // Report
    printf("=============================================================\n");
    printf("  lfg.cpp vs llama.cpp — same model, same Zig compiler\n");
    printf("  Model: %s | Predict: %d | Threads: %d\n", model_name.c_str(), cfg.n_predict, cfg.n_threads);
    printf("  Warmup: %d | Iterations: %d\n", cfg.warmup, cfg.iters);
    printf("=============================================================\n\n");
    printf("%-16s %18s %18s %12s\n", "Engine", "Gen t/s (median)", "Total ms (median)", "vs baseline");
    printf("-------------------------------------------------------------------\n");
    printf("%-16s %18.1f %18.1f %12s\n", "llama.cpp", llama_tps_med, llama_ms_med, "—");
    printf("%-16s %18.1f %18.1f %+11.2f%%\n", "lfg.cpp", lfg_tps_med, lfg_ms_med, pct_diff);

    if (!cfg.no_parity) {
        printf("\nToken parity: %s (%d/%d identical)\n",
            parity_pass ? "PASS" : "FAIL", parity_match, parity_total);
    }

    printf("=============================================================\n");

    return parity_pass ? 0 : 1;
}
