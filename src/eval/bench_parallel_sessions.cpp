// Benchmark: parallel session scaling
// Measures aggregate throughput with N concurrent sessions sharing one model.

#include "lfg_api.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>

struct bench_cfg {
    std::string model_path;
    int n_predict = 128;
    int n_threads = 1; // per-session thread count
    int warmup    = 1;
    int iters     = 3;
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

// Simple barrier for C++17 (no std::barrier until C++20)
struct Barrier {
    std::mutex              mtx;
    std::condition_variable cv;
    int                     count;
    int                     expected;
    int                     generation;

    explicit Barrier(int n) : count(0), expected(n), generation(0) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mtx);
        int gen = generation;
        if (++count == expected) {
            count = 0;
            generation++;
            cv.notify_all();
        } else {
            cv.wait(lock, [&] { return gen != generation; });
        }
    }
};

struct thread_result {
    int    n_tokens;
    double elapsed_ms;
    double tps;
};

static void worker_fn(
    lfg_model *model,
    const bench_cfg &cfg,
    Barrier &barrier,
    thread_result &result)
{
    const lfg_vocab *vocab = lfg_model_get_vocab(model);

    lfg_session_config scfg = lfg_session_default_config();
    scfg.n_ctx = 2048;
    scfg.n_threads = cfg.n_threads;
    scfg.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &scfg);
    if (!session) {
        result = {0, 0, 0};
        return;
    }

    // Ingest BOS only
    const char *prompt = "Hello";
    int32_t prompt_len = (int32_t)std::strlen(prompt);
    std::vector<lfg_token> tokens(prompt_len + 16);
    int32_t n = lfg_tokenize(vocab, prompt, prompt_len,
                              tokens.data(), (int32_t)tokens.size(), true, false);
    if (n <= 0) {
        lfg_session_free(session);
        result = {0, 0, 0};
        return;
    }
    tokens.resize(n);
    lfg_session_ingest_tokens(session, tokens.data(), tokens.size(), true);

    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = cfg.n_predict;

    // Wait for all threads to be ready
    barrier.arrive_and_wait();

    auto t0 = std::chrono::high_resolution_clock::now();
    lfg_generate_result gr = lfg_session_generate(session, gc);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.n_tokens = gr.n_tokens;
    result.elapsed_ms = ms;
    result.tps = (gr.n_tokens > 0) ? (gr.n_tokens / (ms / 1000.0)) : 0;

    lfg_session_free(session);
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

    printf("=== Parallel Sessions Scaling Benchmark ===\n");
    printf("Model: %s\n", cfg.model_path.c_str());
    printf("Generate: %d tokens/session, Threads/session: %d, Warmup: %d, Iters: %d\n\n",
           cfg.n_predict, cfg.n_threads, cfg.warmup, cfg.iters);

    int thread_counts[] = {1, 2, 4, 8};
    int n_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

    printf("%-10s %10s %12s %12s %12s\n",
           "sessions", "wall_ms", "per_thr_tps", "agg_tps", "efficiency");
    printf("%-10s %10s %12s %12s %12s\n",
           "---", "---", "---", "---", "---");

    double baseline_agg_tps = 0;

    for (int ci = 0; ci < n_configs; ci++) {
        int N = thread_counts[ci];

        std::vector<double> agg_tps_samples;
        std::vector<double> wall_samples;
        std::vector<double> per_thread_samples;

        for (int run = 0; run < cfg.warmup + cfg.iters; ++run) {
            Barrier barrier(N);
            std::vector<thread_result> results(N);
            std::vector<std::thread> threads;

            for (int t = 0; t < N; t++) {
                threads.emplace_back(worker_fn,
                    model, std::ref(cfg), std::ref(barrier), std::ref(results[t]));
            }

            for (auto &th : threads) th.join();

            // Wall time = max elapsed across all threads (from barrier release)
            double max_ms = 0;
            double total_tokens = 0;
            double tps_sum = 0;
            for (int t = 0; t < N; t++) {
                if (results[t].elapsed_ms > max_ms) max_ms = results[t].elapsed_ms;
                total_tokens += results[t].n_tokens;
                tps_sum += results[t].tps;
            }

            if (run >= cfg.warmup) {
                double agg = (total_tokens > 0) ? (total_tokens / (max_ms / 1000.0)) : 0;
                agg_tps_samples.push_back(agg);
                wall_samples.push_back(max_ms);
                per_thread_samples.push_back(tps_sum / N);
            }
        }

        double med_agg = median(agg_tps_samples);
        double med_wall = median(wall_samples);
        double med_per = median(per_thread_samples);

        if (ci == 0) baseline_agg_tps = med_agg;

        // Scaling efficiency: ideal = N * baseline, actual = med_agg
        double ideal = baseline_agg_tps * N;
        double efficiency = (ideal > 0) ? (med_agg / ideal * 100.0) : 0;

        printf("%-10d %10.1f %12.1f %12.1f %11.1f%%\n",
               N, med_wall, med_per, med_agg, efficiency);
    }

    printf("\n");
    lfg_model_free(model);
    return 0;
}
