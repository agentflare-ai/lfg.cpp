#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../inference/lfg_api.h"
#include "../inference/lfg_inference.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

static const char * MODEL_PATH = "models/all-MiniLM-L6-v2.Q8_0.gguf";

static float cosine_similarity(const float * a, const float * b, int n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na == 0.0f || nb == 0.0f) return 0.0f;
    return dot / (sqrtf(na) * sqrtf(nb));
}

static bool model_exists() {
    std::ifstream f(MODEL_PATH);
    return f.good();
}

// Shared model/session fixtures
static lfg_model * g_model   = nullptr;
static lfg_session * g_session = nullptr;

static void load_model() {
    if (g_model) return;
    lfg_backend_init();
    lfg_model_load_config cfg = lfg_model_load_default_config();
    cfg.model_path = MODEL_PATH;
    cfg.n_gpu_layers = 99;
    g_model = lfg_load_model(&cfg);
}

static void create_session() {
    if (g_session) return;
    load_model();
    if (!g_model) return;
    lfg_session_config scfg = lfg_session_default_config();
    scfg.n_ctx = 512;
    scfg.n_threads = 4;
    g_session = lfg_session_create(g_model, &scfg);
}

TEST_CASE("BERT model loads successfully") {
    if (!model_exists()) {
        MESSAGE("Skipping: " << MODEL_PATH << " not found");
        return;
    }
    load_model();
    REQUIRE(g_model != nullptr);

    lfg_model_stats stats = lfg_model_get_stats(g_model);
    MESSAGE("  n_params:    " << stats.n_params);
    MESSAGE("  size_bytes:  " << stats.size_bytes);
    MESSAGE("  n_vocab:     " << stats.n_vocab);
    MESSAGE("  n_ctx_train: " << stats.n_ctx_train);

    CHECK(stats.n_params > 0);
    CHECK(stats.n_vocab > 0);
    CHECK(stats.n_ctx_train > 0);
}

TEST_CASE("BERT session creation") {
    if (!model_exists()) { MESSAGE("Skipping"); return; }
    create_session();
    REQUIRE(g_session != nullptr);
}

TEST_CASE("BERT mean-pooled embedding produces non-zero vector") {
    if (!model_exists()) { MESSAGE("Skipping"); return; }
    create_session();
    REQUIRE(g_session != nullptr);

    const char * text = "Hello world";
    int32_t text_len = (int32_t)strlen(text);

    // all-MiniLM-L6-v2 has 384-dim embeddings
    std::vector<float> embd(384, 0.0f);
    int32_t n_embd = lfg_session_embed(g_session, text, text_len, embd.data(), (int32_t)embd.size());

    MESSAGE("  n_embd returned: " << n_embd);
    REQUIRE(n_embd > 0);
    CHECK(n_embd == 384);

    // Embedding should be non-zero
    float sum_sq = 0.0f;
    for (int i = 0; i < n_embd; i++) sum_sq += embd[i] * embd[i];
    MESSAGE("  L2 norm: " << sqrtf(sum_sq));
    CHECK(sum_sq > 0.5f);  // L2-normalized → ~1.0
    CHECK(sum_sq < 1.5f);
}

TEST_CASE("BERT embedding: similar sentences have high cosine similarity") {
    if (!model_exists()) { MESSAGE("Skipping"); return; }
    create_session();
    REQUIRE(g_session != nullptr);

    const char * s1 = "The cat sat on the mat";
    const char * s2 = "A kitten was sitting on a rug";
    const char * s3 = "Quantum mechanics describes subatomic particle behavior";

    std::vector<float> e1(384), e2(384), e3(384);

    int32_t r1 = lfg_session_embed(g_session, s1, (int32_t)strlen(s1), e1.data(), 384);
    int32_t r2 = lfg_session_embed(g_session, s2, (int32_t)strlen(s2), e2.data(), 384);
    int32_t r3 = lfg_session_embed(g_session, s3, (int32_t)strlen(s3), e3.data(), 384);

    REQUIRE(r1 == 384);
    REQUIRE(r2 == 384);
    REQUIRE(r3 == 384);

    float sim_similar   = cosine_similarity(e1.data(), e2.data(), 384);
    float sim_different = cosine_similarity(e1.data(), e3.data(), 384);

    MESSAGE("  sim(cat/kitten):    " << sim_similar);
    MESSAGE("  sim(cat/quantum):   " << sim_different);

    // Similar sentences should have higher cosine similarity than unrelated ones
    CHECK(sim_similar > sim_different);
    // Similar sentences: expect > 0.5
    CHECK(sim_similar > 0.5f);
    // Unrelated sentences: expect < 0.5
    CHECK(sim_different < 0.5f);
}

TEST_CASE("BERT embedding: same text produces identical embeddings") {
    if (!model_exists()) { MESSAGE("Skipping"); return; }
    create_session();
    REQUIRE(g_session != nullptr);

    const char * text = "Reproducibility test sentence";
    int32_t len = (int32_t)strlen(text);

    std::vector<float> e1(384), e2(384);
    REQUIRE(lfg_session_embed(g_session, text, len, e1.data(), 384) == 384);
    REQUIRE(lfg_session_embed(g_session, text, len, e2.data(), 384) == 384);

    float sim = cosine_similarity(e1.data(), e2.data(), 384);
    MESSAGE("  sim(same, same): " << sim);
    CHECK(sim > 0.9999f);  // Should be essentially 1.0
}

TEST_CASE("BERT per-token embeddings") {
    if (!model_exists()) { MESSAGE("Skipping"); return; }
    create_session();
    REQUIRE(g_session != nullptr);

    const char * text = "Hello world";
    int32_t text_len = (int32_t)strlen(text);

    // Allocate enough for many tokens
    int32_t max_tokens = 32;
    int32_t n_embd = 384;
    std::vector<float> embd(max_tokens * n_embd, 0.0f);

    int32_t n_tok = lfg_session_embed_tokens(g_session, text, text_len,
                                              embd.data(), max_tokens * n_embd);

    MESSAGE("  n_tok returned: " << n_tok);
    REQUIRE(n_tok > 0);
    CHECK(n_tok <= max_tokens);

    // Each per-token embedding should be L2-normalized (non-zero)
    for (int32_t t = 0; t < n_tok; t++) {
        float sum_sq = 0.0f;
        const float * tok_embd = embd.data() + t * n_embd;
        for (int i = 0; i < n_embd; i++) sum_sq += tok_embd[i] * tok_embd[i];
        CHECK(sum_sq > 0.5f);
        CHECK(sum_sq < 1.5f);
    }

    // Per-token embeddings should differ from each other
    if (n_tok >= 2) {
        float sim = cosine_similarity(embd.data(), embd.data() + n_embd, n_embd);
        MESSAGE("  sim(token0, token1): " << sim);
        CHECK(sim < 0.9999f);  // Different tokens → different embeddings
    }
}

TEST_CASE("BERT embedding: semantic search use case") {
    if (!model_exists()) { MESSAGE("Skipping"); return; }
    create_session();
    REQUIRE(g_session != nullptr);

    // Query
    const char * query = "What is the weather like?";

    // Documents
    const char * docs[] = {
        "The forecast calls for sunny skies and warm temperatures",
        "Machine learning models can classify images accurately",
        "It will rain heavily tomorrow afternoon",
        "The stock market rallied after the Federal Reserve announcement",
    };
    const int n_docs = 4;

    std::vector<float> q_embd(384);
    REQUIRE(lfg_session_embed(g_session, query, (int32_t)strlen(query), q_embd.data(), 384) == 384);

    float best_sim = -1.0f;
    int best_idx = -1;

    for (int i = 0; i < n_docs; i++) {
        std::vector<float> d_embd(384);
        REQUIRE(lfg_session_embed(g_session, docs[i], (int32_t)strlen(docs[i]), d_embd.data(), 384) == 384);

        float sim = cosine_similarity(q_embd.data(), d_embd.data(), 384);
        MESSAGE("  sim(query, doc[" << i << "]): " << sim << " — " << docs[i]);

        if (sim > best_sim) {
            best_sim = sim;
            best_idx = i;
        }
    }

    MESSAGE("  Best match: doc[" << best_idx << "] with sim=" << best_sim);
    // Weather-related documents (0 or 2) should rank highest
    CHECK((best_idx == 0 || best_idx == 2));
}

// Cleanup at end
TEST_CASE("BERT cleanup") {
    if (g_session) { lfg_session_free(g_session); g_session = nullptr; }
    if (g_model)   { lfg_model_free(g_model);     g_model = nullptr; }
}
