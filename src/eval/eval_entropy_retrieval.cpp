// eval_entropy_retrieval.cpp — Agentic entropy-triggered retrieval demo
//
// Demonstrates the full agentic pattern:
//   1. Build a knowledge base by embedding facts via lfg_session_embed()
//   2. For each query, run baseline (no monitor) and retrieval-augmented generation
//   3. Print a decision trace showing entropy events, KB matches, and rewind/inject
//   4. Compare outputs and timing
//
// Usage:
//   zig build install -Dall_targets=false -Doptimize=ReleaseFast
//   ./zig-out/bin/eval-entropy-retrieval

#include "lfg_api.h"
#include <chrono>
#include <cstdio>
#include <cstring>
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

static float cosine_similarity(const float *a, const float *b, int n) {
    float dot = 0.0f;
    for (int i = 0; i < n; ++i) dot += a[i] * b[i];
    return dot;  // both L2-normalized, so dot == cosine
}

using Clock = std::chrono::high_resolution_clock;

// ---------------------------------------------------------------------------
// Knowledge base
// ---------------------------------------------------------------------------

struct kb_entry {
    const char *text;
    std::vector<float> embedding;
};

static std::vector<kb_entry> build_kb(lfg_session *session, const char **texts, int n_texts, int32_t n_embd) {
    std::vector<kb_entry> kb;
    for (int i = 0; i < n_texts; ++i) {
        kb_entry entry;
        entry.text = texts[i];
        entry.embedding.resize(n_embd);
        int32_t got = lfg_session_embed(session, texts[i], (int32_t)std::strlen(texts[i]),
                                         entry.embedding.data(), n_embd);
        if (got != n_embd) {
            printf("  WARN: embed failed for KB[%d], zeroing\n", i);
            entry.embedding.assign(n_embd, 0.0f);
        }
        kb.push_back(std::move(entry));
    }
    return kb;
}

// ---------------------------------------------------------------------------
// Baseline generation (no entropy monitor)
// ---------------------------------------------------------------------------

struct run_result {
    std::string text;
    double      ms;
    int         n_tokens;
};

static run_result run_baseline(lfg_model *model, const lfg_vocab *vocab,
                               const std::string &prompt, int max_tokens) {
    run_result r = {};

    lfg_session_config sc = lfg_session_default_config();
    sc.n_ctx = 2048;
    sc.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &sc);
    if (!session) { printf("  ERROR: session_create failed\n"); return r; }

    auto toks = tokenize(vocab, prompt, true);
    lfg_session_ingest_tokens(session, toks.data(), toks.size(), true);

    auto t0 = Clock::now();
    for (int i = 0; i < max_tokens; ++i) {
        lfg_session_decode(session);
        lfg_token tok = lfg_session_sample(session);
        if (lfg_vocab_is_eog(vocab, tok)) break;
        r.text += token_to_string(vocab, tok);
        r.n_tokens++;
        lfg_session_ingest_tokens(session, &tok, 1, false);
    }
    auto t1 = Clock::now();
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    lfg_session_free(session);
    return r;
}

// ---------------------------------------------------------------------------
// Retrieval-augmented generation with entropy monitor
// ---------------------------------------------------------------------------

struct retrieval_result {
    std::string text;
    double      ms;
    int         n_tokens;
    int         n_retrievals;
};

static retrieval_result run_with_retrieval(
    lfg_model *model, const lfg_vocab *vocab,
    const std::string &prompt,
    const std::vector<kb_entry> &kb, int32_t n_embd,
    int max_tokens, int max_retrievals,
    float threshold, int32_t cooldown)
{
    retrieval_result r = {};

    lfg_session_config sc = lfg_session_default_config();
    sc.n_ctx = 2048;
    sc.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &sc);
    if (!session) { printf("  ERROR: session_create failed\n"); return r; }

    // Configure entropy monitor
    lfg_entropy_monitor_config ecfg = lfg_entropy_monitor_default_config();
    ecfg.threshold = threshold;
    ecfg.cooldown_tokens = cooldown;
    ecfg.ring_size = 8;
    lfg_session_configure_entropy_monitor(session, &ecfg);

    auto toks = tokenize(vocab, prompt, true);
    lfg_session_ingest_tokens(session, toks.data(), toks.size(), true);

    std::vector<float> embd_buf(n_embd);

    auto t0 = Clock::now();
    for (int i = 0; i < max_tokens; ++i) {
        lfg_session_decode(session);
        lfg_token tok = lfg_session_sample(session);
        if (lfg_vocab_is_eog(vocab, tok)) break;

        // Check for entropy events
        lfg_entropy_event ev;
        if (r.n_retrievals < max_retrievals &&
            lfg_session_entropy_pop(session, &ev, embd_buf.data(), n_embd)) {

            // Find best KB match
            int best_idx = -1;
            float best_score = -1.0f;
            for (int k = 0; k < (int)kb.size(); ++k) {
                float score = cosine_similarity(embd_buf.data(), kb[k].embedding.data(), n_embd);
                if (score > best_score) { best_score = score; best_idx = k; }
            }

            std::string tok_str = token_to_string(vocab, ev.token);
            printf("  Event @ pos=%d: H_norm=%.2f, token=\"%s\"\n",
                   ev.n_past, ev.normalized, tok_str.c_str());

            if (best_idx >= 0) {
                // Truncate KB text for display
                char preview[60];
                std::snprintf(preview, sizeof(preview), "%.56s", kb[best_idx].text);
                if (std::strlen(kb[best_idx].text) > 56) {
                    preview[56] = '.'; preview[57] = '.'; preview[58] = '.'; preview[59] = '\0';
                }
                printf("    -> matched KB[%d] \"%s\" (sim=%.3f)\n", best_idx, preview, best_score);
            }

            if (best_idx >= 0 && best_score > 0.0f) {
                if (lfg_session_rewind(session, ev.checkpoint_id)) {
                    std::string inject = " " + std::string(kb[best_idx].text) + " ";
                    auto inject_toks = tokenize(vocab, inject, false);
                    lfg_session_ingest_tokens(session, inject_toks.data(), inject_toks.size(), false);
                    r.n_retrievals++;
                    printf("    -> rewind to pos=%d, injected %d tokens\n", ev.n_past, (int)inject_toks.size());

                    lfg_session_entropy_flush(session);
                    continue;  // re-decode from new position
                } else {
                    printf("    -> rewind FAILED (snapshot expired?)\n");
                }
            }
        } else {
            lfg_session_entropy_flush(session);
        }

        r.text += token_to_string(vocab, tok);
        r.n_tokens++;
        lfg_session_ingest_tokens(session, &tok, 1, false);
    }
    auto t1 = Clock::now();
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    lfg_session_free(session);
    return r;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    const char *model_path = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
    const int MAX_TOKENS = 80;
    const int MAX_RETRIEVALS = 2;
    const float THRESHOLD = 0.05f;
    const int32_t COOLDOWN = 16;

    printf("=== Entropy-Triggered Retrieval Demo ===\n");
    printf("Model: %s\n", model_path);

    lfg_backend_init();
    lfg_model_load_config cfg = lfg_model_load_default_config();
    cfg.model_path = model_path;
    cfg.n_gpu_layers = 0;

    lfg_model *model = lfg_load_model(&cfg);
    if (!model) {
        printf("FAIL: could not load model at %s\n", model_path);
        return 1;
    }
    const lfg_vocab *vocab = lfg_model_get_vocab(model);
    int32_t n_embd = lfg_model_n_embd(model);

    // --- Knowledge base ---
    const char *facts[] = {
        "The Eiffel Tower is 330 meters tall and was completed in 1889 for the World's Fair in Paris, France.",
        "The speed of light in vacuum is exactly 299,792,458 meters per second.",
        "Mount Everest has an elevation of 8,849 meters above sea level, located in the Himalayas.",
        "The Pacific Ocean covers approximately 165.25 million square kilometers, making it the largest ocean.",
        "DNA was first identified by Friedrich Miescher in 1869, and its double helix structure was discovered by Watson and Crick in 1953.",
    };
    constexpr int N_FACTS = 5;

    printf("Building knowledge base: %d entries (n_embd=%d)\n", N_FACTS, n_embd);

    // Create a temporary session just for embedding the KB
    lfg_session_config embed_sc = lfg_session_default_config();
    embed_sc.n_ctx = 2048;
    embed_sc.sampling.temp = 0.0f;
    lfg_session *embed_session = lfg_session_create(model, &embed_sc);
    if (!embed_session) {
        printf("FAIL: could not create embedding session\n");
        lfg_model_free(model);
        return 1;
    }

    auto t_kb0 = Clock::now();
    auto kb = build_kb(embed_session, facts, N_FACTS, n_embd);
    auto t_kb1 = Clock::now();
    double kb_ms = std::chrono::duration<double, std::milli>(t_kb1 - t_kb0).count();
    printf("KB embedded in %.0f ms\n\n", kb_ms);

    lfg_session_free(embed_session);

    // --- Queries ---
    struct query {
        const char *text;
        const char *label;
    };

    query queries[] = {
        {"Tell me about the Eiffel Tower and how tall it is.\n", "Eiffel Tower"},
        {"What is DNA and who discovered it?\n", "DNA discovery"},
        {"How fast does light travel through space?\n", "Speed of light"},
    };
    constexpr int N_QUERIES = 3;

    int total_retrievals = 0;

    for (int q = 0; q < N_QUERIES; ++q) {
        printf("--- Query %d: \"%s\" ---\n\n", q + 1, queries[q].label);

        // Baseline
        auto baseline = run_baseline(model, vocab, queries[q].text, MAX_TOKENS);
        double baseline_tps = baseline.n_tokens > 0 ? 1000.0 * baseline.n_tokens / baseline.ms : 0;
        printf("[Baseline] %d tokens, %.0f ms (%.1f tok/s)\n", baseline.n_tokens, baseline.ms, baseline_tps);
        printf("  %s\n\n", baseline.text.c_str());

        // With retrieval
        printf("[Retrieval] threshold=%.2f, cooldown=%d, max_retrievals=%d\n",
               THRESHOLD, COOLDOWN, MAX_RETRIEVALS);
        auto retrieval = run_with_retrieval(model, vocab, queries[q].text, kb, n_embd,
                                            MAX_TOKENS, MAX_RETRIEVALS, THRESHOLD, COOLDOWN);
        double retrieval_tps = retrieval.n_tokens > 0 ? 1000.0 * retrieval.n_tokens / retrieval.ms : 0;
        printf("  %d tokens, %.0f ms (%.1f tok/s), %d retrieval(s)\n",
               retrieval.n_tokens, retrieval.ms, retrieval_tps, retrieval.n_retrievals);
        printf("  %s\n\n", retrieval.text.c_str());

        total_retrievals += retrieval.n_retrievals;

        // Comparison
        bool differs = (retrieval.text != baseline.text);
        printf("[Compare] outputs %s\n\n", differs ? "DIFFER (retrieval changed generation)" : "SAME (no retrieval impact)");
    }

    printf("=== Summary ===\n");
    printf("Queries: %d, Total retrievals: %d\n", N_QUERIES, total_retrievals);
    printf("KB: %d entries, n_embd=%d\n", N_FACTS, n_embd);

    lfg_model_free(model);
    return 0;
}
