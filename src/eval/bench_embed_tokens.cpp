// Benchmark: lfg_session_embed_tokens() throughput for ColBERT-style models
//
// Measures tokens/sec for per-token embedding computation across varying
// input lengths. Reports both first-call (includes lazy context creation)
// and steady-state throughput.

#include "lfg_api.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Corpus of varied-length texts for realistic throughput measurement
static const char *TEXTS[] = {
    // Short (< 20 tokens)
    "What is quantum computing?",
    "The mitochondria is the powerhouse of the cell.",

    // Medium (~30-50 tokens)
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
    "It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair.",

    "Transformers are a deep learning architecture based on self-attention mechanisms. "
    "They process sequences in parallel and have become the foundation of modern language models.",

    // Long (~60-100 tokens)
    "Photosynthesis is the process by which green plants and certain other organisms convert "
    "light energy into chemical energy. During this process, carbon dioxide and water are "
    "transformed into glucose and oxygen using sunlight absorbed by chlorophyll in the leaves. "
    "This process is fundamental to life on Earth as it produces the oxygen we breathe.",

    "The Great Wall of China is a series of fortifications built along the historical northern "
    "borders of ancient Chinese states and Imperial China. Construction began in the 7th century "
    "BC and continued for centuries under various dynasties. The wall spans thousands of kilometers "
    "and was built to protect against various nomadic groups from the Eurasian Steppe.",
};
static const int N_TEXTS = sizeof(TEXTS) / sizeof(TEXTS[0]);

int main(int argc, char **argv) {
    const char *model_path = (argc > 1) ? argv[1]
        : "models/LFM2-ColBERT-350M-Q4_K_M.gguf";
    int n_iters = (argc > 2) ? atoi(argv[2]) : 20;

    lfg_backend_init();

    lfg_model_load_config lcfg = lfg_model_load_default_config();
    lcfg.model_path = model_path;
    lcfg.n_gpu_layers = 99;
    struct lfg_model *model = lfg_load_model(&lcfg);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    int32_t n_embd = lfg_model_n_embd_out(model);
    printf("Model: %s\n", model_path);
    printf("Embedding dim: %d\n", n_embd);
    printf("Iterations per text: %d\n\n", n_iters);

    lfg_session_config scfg = lfg_session_default_config();
    scfg.n_ctx = 512;
    scfg.n_batch = 512;
    lfg_session *session = lfg_session_create(model, &scfg);
    if (!session) {
        fprintf(stderr, "Failed to create session\n");
        lfg_model_free(model);
        return 1;
    }

    // Output buffer (512 tokens * n_embd)
    int32_t out_cap = 512 * n_embd;
    std::vector<float> out(out_cap);

    // Warmup + measure first-call latency (includes lazy context creation)
    {
        int32_t len = (int32_t)std::strlen(TEXTS[0]);
        auto t0 = Clock::now();
        int32_t n_tok = lfg_session_embed_tokens(session, TEXTS[0], len, out.data(), out_cap);
        auto t1 = Clock::now();
        printf("First call (includes context creation): %d tokens in %.1f ms (%.0f tok/s)\n\n",
               n_tok, ms(t0, t1), n_tok / (ms(t0, t1) / 1000.0));
    }

    // Steady-state benchmark per text
    printf("%-60s  %6s  %8s  %10s\n", "Text (truncated)", "Tokens", "Avg (ms)", "Tok/s");
    printf("%-60s  %6s  %8s  %10s\n",
           "------------------------------------------------------------", "------", "--------", "----------");

    int64_t total_tokens = 0;
    double total_ms = 0.0;

    for (int t = 0; t < N_TEXTS; t++) {
        int32_t len = (int32_t)std::strlen(TEXTS[t]);
        int32_t n_tok = 0;
        double sum_ms = 0.0;

        for (int i = 0; i < n_iters; i++) {
            auto t0 = Clock::now();
            n_tok = lfg_session_embed_tokens(session, TEXTS[t], len, out.data(), out_cap);
            auto t1 = Clock::now();
            sum_ms += ms(t0, t1);
        }

        double avg_ms = sum_ms / n_iters;
        double tok_per_sec = n_tok / (avg_ms / 1000.0);
        total_tokens += (int64_t)n_tok * n_iters;
        total_ms += sum_ms;

        // Truncate text for display
        char trunc[57];
        std::snprintf(trunc, sizeof(trunc), "%.53s%s", TEXTS[t],
                      std::strlen(TEXTS[t]) > 53 ? "..." : "");

        printf("%-60s  %6d  %8.2f  %10.0f\n", trunc, n_tok, avg_ms, tok_per_sec);
    }

    double overall_tok_per_sec = total_tokens / (total_ms / 1000.0);
    printf("\nOverall: %lld tokens in %.1f ms = %.0f tok/s\n",
           (long long)total_tokens, total_ms, overall_tok_per_sec);

    lfg_session_free(session);
    lfg_model_free(model);
    return 0;
}
