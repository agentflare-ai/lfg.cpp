// eval_bert_parity.cpp — In-process embedding parity & throughput: lfg.cpp vs llama.cpp
//
// Links BOTH llama_core (vendored llama.cpp) and lfg_core (lfg.cpp).
// Same compiler, same optimization flags, same Metal backend.
// Computes embeddings via both APIs and compares vectors + throughput.
//
// Usage: eval_bert_parity <model.gguf> [iterations]

// llama.cpp API
#include "llama.h"

// lfg.cpp API
#include "lfg_api.h"
#include "lfg_inference.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---- helpers ----

static void l2_normalize(float * v, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += v[i] * v[i];
    if (sum <= 0.0f) return;
    float inv = 1.0f / sqrtf(sum);
    for (int i = 0; i < n; i++) v[i] *= inv;
}

static float cosine_sim(const float * a, const float * b, int n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na == 0.0f || nb == 0.0f) return 0.0f;
    return dot / (sqrtf(na) * sqrtf(nb));
}

static float max_abs_diff(const float * a, const float * b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static float mean_abs_diff(const float * a, const float * b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += fabsf(a[i] - b[i]);
    return sum / (float)n;
}

// ---- llama.cpp embedding (in-process, same compiler) ----

static int embed_llama(llama_context * ctx, const llama_vocab * vocab,
                        const char * text, float * out, int n_embd) {
    int tok_cap = (int)strlen(text) + 16;
    llama_token * toks = (llama_token *)malloc(tok_cap * sizeof(llama_token));
    int n_tok = llama_tokenize(vocab, text, (int)strlen(text), toks, tok_cap, true, false);
    if (n_tok < 0) {
        tok_cap = -n_tok;
        toks = (llama_token *)realloc(toks, tok_cap * sizeof(llama_token));
        n_tok = llama_tokenize(vocab, text, (int)strlen(text), toks, tok_cap, true, false);
    }
    if (n_tok <= 0) { free(toks); return 0; }

    int ctx_cap = (int)llama_n_ctx(ctx);
    llama_token * tok_ptr = toks;
    if (n_tok > ctx_cap) {
        tok_ptr = toks + (n_tok - ctx_cap);
        n_tok = ctx_cap;
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_get_one(tok_ptr, n_tok);
    if (llama_decode(ctx, batch) != 0) { free(toks); return 0; }

    float * embd = llama_get_embeddings_seq(ctx, 0);
    if (!embd) embd = llama_get_embeddings_ith(ctx, -1);
    if (!embd) { free(toks); return 0; }

    memcpy(out, embd, n_embd * sizeof(float));
    l2_normalize(out, n_embd);
    free(toks);
    return n_embd;
}

// ---- lfg.cpp embedding (in-process, same compiler) ----

static int embed_lfg(lfg_context * ctx, const lfg_vocab * vocab,
                      const char * text, float * out, int n_embd) {
    int tok_cap = (int)strlen(text) + 16;
    lfg_token * toks = (lfg_token *)malloc(tok_cap * sizeof(lfg_token));
    int n_tok = lfg_tokenize(vocab, text, (int)strlen(text), toks, tok_cap, true, false);
    if (n_tok < 0) {
        tok_cap = -n_tok;
        toks = (lfg_token *)realloc(toks, tok_cap * sizeof(lfg_token));
        n_tok = lfg_tokenize(vocab, text, (int)strlen(text), toks, tok_cap, true, false);
    }
    if (n_tok <= 0) { free(toks); return 0; }

    int ctx_cap = (int)lfg_n_ctx(ctx);
    lfg_token * tok_ptr = toks;
    if (n_tok > ctx_cap) {
        tok_ptr = toks + (n_tok - ctx_cap);
        n_tok = ctx_cap;
    }

    lfg_memory_clear(lfg_get_memory(ctx), true);
    lfg_batch batch = lfg_batch_get_one(tok_ptr, n_tok);
    if (lfg_decode(ctx, batch) != 0) { free(toks); return 0; }

    float * embd = lfg_get_embeddings_seq(ctx, 0);
    if (!embd) embd = lfg_get_embeddings_ith(ctx, -1);
    if (!embd) { free(toks); return 0; }

    memcpy(out, embd, n_embd * sizeof(float));
    l2_normalize(out, n_embd);
    free(toks);
    return n_embd;
}

// ---- main ----

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [iterations]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    int iterations = argc >= 3 ? atoi(argv[2]) : 200;
    if (iterations < 1) iterations = 200;

    const char * prompts[] = {
        "The cat sat on the mat",
        "A kitten was sitting on a rug",
        "Quantum mechanics describes subatomic particle behavior",
        "What is the weather like?",
        "The forecast calls for sunny skies and warm temperatures",
        "Machine learning models can classify images accurately",
        "It will rain heavily tomorrow afternoon",
        "The stock market rallied after the Federal Reserve announcement",
        "Artificial intelligence is transforming healthcare",
        "The quick brown fox jumps over the lazy dog",
    };
    const int n_prompts = sizeof(prompts) / sizeof(prompts[0]);

    // =============================================
    // Load model via LLAMA.CPP
    // =============================================
    printf("=== Loading model via llama.cpp ===\n");
    llama_backend_init();

    llama_model_params llama_mparams = llama_model_default_params();
    llama_mparams.n_gpu_layers = 99;
    llama_model * llama_model_ptr = llama_model_load_from_file(model_path, llama_mparams);
    if (!llama_model_ptr) {
        fprintf(stderr, "llama.cpp: failed to load model\n");
        return 1;
    }

    int n_embd_llama = llama_model_n_embd_out(llama_model_ptr);
    const llama_vocab * llama_vocab_ptr = llama_model_get_vocab(llama_model_ptr);
    printf("  n_embd: %d\n", n_embd_llama);

    llama_context_params llama_cparams = llama_context_default_params();
    llama_cparams.n_ctx = 512;
    llama_cparams.n_batch = 512;
    llama_cparams.n_threads = 4;
    llama_cparams.embeddings = true;
    llama_cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    llama_context * llama_ctx = llama_init_from_model(llama_model_ptr, llama_cparams);
    if (!llama_ctx) {
        fprintf(stderr, "llama.cpp: failed to create context\n");
        return 1;
    }

    // =============================================
    // Load model via LFG.CPP
    // =============================================
    printf("\n=== Loading model via lfg.cpp ===\n");
    lfg_backend_init();

    lfg_model_load_config lfg_lcfg = lfg_model_load_default_config();
    lfg_lcfg.model_path = model_path;
    lfg_lcfg.n_gpu_layers = 99;
    lfg_model * lfg_model_ptr = lfg_load_model(&lfg_lcfg);
    if (!lfg_model_ptr) {
        fprintf(stderr, "lfg.cpp: failed to load model\n");
        return 1;
    }

    int n_embd_lfg = lfg_model_n_embd_out(lfg_model_ptr);
    const lfg_vocab * lfg_vocab_ptr = lfg_model_get_vocab(lfg_model_ptr);
    printf("  n_embd: %d\n", n_embd_lfg);

    if (n_embd_llama != n_embd_lfg) {
        fprintf(stderr, "n_embd mismatch: llama=%d lfg=%d\n", n_embd_llama, n_embd_lfg);
        return 1;
    }
    int n_embd = n_embd_llama;

    lfg_context_params lfg_cparams = lfg_context_default_params();
    lfg_cparams.n_ctx = 512;
    lfg_cparams.n_batch = 512;
    lfg_cparams.n_threads = 4;
    lfg_cparams.embeddings = true;
    lfg_cparams.pooling_type = LFG_POOLING_TYPE_MEAN;
    lfg_context * lfg_ctx = lfg_init_from_model(lfg_model_ptr, lfg_cparams);
    if (!lfg_ctx) {
        fprintf(stderr, "lfg.cpp: failed to create context\n");
        return 1;
    }

    // =============================================
    // PARITY CHECK
    // =============================================
    printf("\n=== EMBEDDING PARITY: lfg.cpp vs llama.cpp (same compiler, in-process) ===\n");
    printf("%-50s %12s %12s %12s\n", "Prompt", "CosSim", "MaxDiff", "MeanDiff");
    printf("%-50s %12s %12s %12s\n",
           "--------------------------------------------------", "------------", "------------", "------------");

    std::vector<float> embd_llama_v(n_embd);
    std::vector<float> embd_lfg_v(n_embd);

    bool all_match = true;
    int total_exact = 0, total_close = 0, total_diverged = 0;

    for (int i = 0; i < n_prompts; i++) {
        int r1 = embed_llama(llama_ctx, llama_vocab_ptr, prompts[i], embd_llama_v.data(), n_embd);
        int r2 = embed_lfg(lfg_ctx, lfg_vocab_ptr, prompts[i], embd_lfg_v.data(), n_embd);

        if (r1 != n_embd || r2 != n_embd) {
            char sp[51]; snprintf(sp, sizeof(sp), "%.50s", prompts[i]);
            printf("%-50s  FAILED (r1=%d r2=%d)\n", sp, r1, r2);
            all_match = false;
            continue;
        }

        float cs = cosine_sim(embd_llama_v.data(), embd_lfg_v.data(), n_embd);
        float md = max_abs_diff(embd_llama_v.data(), embd_lfg_v.data(), n_embd);
        float ad = mean_abs_diff(embd_llama_v.data(), embd_lfg_v.data(), n_embd);

        char sp[51]; snprintf(sp, sizeof(sp), "%.50s", prompts[i]);
        printf("%-50s %12.10f %12.2e %12.2e\n", sp, cs, md, ad);

        if (cs < 0.9999f) all_match = false;

        for (int j = 0; j < n_embd; j++) {
            float d = fabsf(embd_llama_v[j] - embd_lfg_v[j]);
            if (d == 0.0f) total_exact++;
            else if (d < 1e-6f) total_close++;
            else total_diverged++;
        }
    }

    // Side-by-side sample
    printf("\n  Sample (prompt 0, first 8 dims):\n");
    embed_llama(llama_ctx, llama_vocab_ptr, prompts[0], embd_llama_v.data(), n_embd);
    embed_lfg(lfg_ctx, lfg_vocab_ptr, prompts[0], embd_lfg_v.data(), n_embd);
    printf("    llama: ");
    for (int i = 0; i < 8; i++) printf("%11.7f ", embd_llama_v[i]);
    printf("...\n");
    printf("    lfg:   ");
    for (int i = 0; i < 8; i++) printf("%11.7f ", embd_lfg_v[i]);
    printf("...\n");

    printf("\n  Element-wise (%d dims x %d prompts = %d total):\n",
           n_embd, n_prompts, n_embd * n_prompts);
    printf("    Exact (d=0):    %d\n", total_exact);
    printf("    Close (d<1e-6): %d\n", total_close);
    printf("    Diverged:       %d\n", total_diverged);
    printf("  Verdict: %s\n", all_match ? "MATCH" : "MISMATCH");

    // =============================================
    // THROUGHPUT BENCHMARK
    // =============================================
    printf("\n=== THROUGHPUT (%d iters x %d prompts = %d calls each) ===\n",
           iterations, n_prompts, iterations * n_prompts);

    // Warmup both
    for (int w = 0; w < 10; w++) {
        embed_llama(llama_ctx, llama_vocab_ptr, prompts[0], embd_llama_v.data(), n_embd);
        embed_lfg(lfg_ctx, lfg_vocab_ptr, prompts[0], embd_lfg_v.data(), n_embd);
    }

    // llama.cpp
    double secs_llama;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < iterations; iter++) {
            for (int i = 0; i < n_prompts; i++) {
                embed_llama(llama_ctx, llama_vocab_ptr, prompts[i], embd_llama_v.data(), n_embd);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        secs_llama = std::chrono::duration<double>(t1 - t0).count();
    }
    int total = iterations * n_prompts;
    printf("  llama.cpp:  %d embeddings in %7.3f s  ->  %7.1f embd/s  (%6.3f ms/embd)\n",
           total, secs_llama, total / secs_llama, 1000.0 * secs_llama / total);

    // lfg.cpp
    double secs_lfg;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < iterations; iter++) {
            for (int i = 0; i < n_prompts; i++) {
                embed_lfg(lfg_ctx, lfg_vocab_ptr, prompts[i], embd_lfg_v.data(), n_embd);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        secs_lfg = std::chrono::duration<double>(t1 - t0).count();
    }
    printf("  lfg.cpp:    %d embeddings in %7.3f s  ->  %7.1f embd/s  (%6.3f ms/embd)\n",
           total, secs_lfg, total / secs_lfg, 1000.0 * secs_lfg / total);

    double ratio = secs_llama / secs_lfg;
    printf("\n  Ratio: lfg.cpp is %.2fx %s than llama.cpp\n",
           ratio > 1.0 ? ratio : 1.0 / ratio,
           ratio > 1.0 ? "faster" : "slower");

    // =============================================
    // CLEANUP
    // =============================================
    llama_free(llama_ctx);
    llama_model_free(llama_model_ptr);

    lfg_free(lfg_ctx);
    lfg_model_free(lfg_model_ptr);

    lfg_backend_free();

    printf("\nDone.\n");
    return all_match ? 0 : 1;
}
