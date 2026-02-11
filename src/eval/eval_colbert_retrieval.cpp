// Example: ColBERT late-interaction retrieval with LFM2-ColBERT-350M
//
// ColBERT computes per-token embeddings for queries and documents, then scores
// via MaxSim: for each query token, find the max cosine similarity with any
// document token, then sum across all query tokens.
//
// This uses lfg_session_embed_tokens() which provides per-token embeddings
// via the session API (POOLING_TYPE_NONE internally).

#include "lfg_api.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

// ── helpers ─────────────────────────────────────────────────────────────────

// Per-token embedding matrix: n_tokens rows of n_embd floats, L2-normalized
struct token_embeddings {
    std::vector<float> data; // flat [n_tokens * n_embd]
    int32_t n_tokens;
    int32_t n_embd;

    const float *row(int32_t i) const { return data.data() + i * n_embd; }
};

// Encode text into per-token embeddings using session API
static token_embeddings encode(lfg_session *session, int32_t n_embd, const char *text) {
    int32_t text_len = (int32_t)std::strlen(text);

    // Allocate generous output buffer (max 512 tokens * n_embd)
    int32_t max_tokens = 512;
    int32_t out_cap = max_tokens * n_embd;
    std::vector<float> buf(out_cap);

    int32_t n_tok = lfg_session_embed_tokens(session, text, text_len,
                                              buf.data(), out_cap);
    if (n_tok <= 0) {
        fprintf(stderr, "embed_tokens failed for: %.40s...\n", text);
        return {{}, 0, n_embd};
    }

    token_embeddings emb;
    emb.n_tokens = n_tok;
    emb.n_embd = n_embd;
    emb.data.assign(buf.begin(), buf.begin() + n_tok * n_embd);
    return emb;
}

// ColBERT MaxSim: sum over query tokens of max similarity with any doc token
static float maxsim(const token_embeddings &query, const token_embeddings &doc) {
    float score = 0.0f;
    for (int32_t q = 0; q < query.n_tokens; q++) {
        float max_dot = -1e9f;
        for (int32_t d = 0; d < doc.n_tokens; d++) {
            float dot = 0.0f;
            for (int32_t j = 0; j < query.n_embd; j++) {
                dot += query.row(q)[j] * doc.row(d)[j];
            }
            if (dot > max_dot) max_dot = dot;
        }
        score += max_dot;
    }
    return score;
}

// ── main ────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    const char *model_path = (argc > 1) ? argv[1]
        : "models/LFM2-ColBERT-350M-Q4_K_M.gguf";

    lfg_backend_init();

    // Load model
    lfg_model_load_config lcfg = lfg_model_load_default_config();
    lcfg.model_path = model_path;
    lcfg.n_gpu_layers = 0;
    struct lfg_model *model = lfg_load_model(&lcfg);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    int32_t n_embd = lfg_model_n_embd_out(model);

    printf("Model: %s\n", model_path);
    printf("Embedding dim (n_embd_out): %d\n\n", n_embd);

    // Create session — embed_tokens will lazily create per-token context
    lfg_session_config scfg = lfg_session_default_config();
    scfg.n_ctx = 512;
    scfg.n_batch = 512;
    lfg_session *session = lfg_session_create(model, &scfg);
    if (!session) {
        fprintf(stderr, "Failed to create session\n");
        lfg_model_free(model);
        return 1;
    }

    // ── Document corpus ─────────────────────────────────────────────────

    const char *documents[] = {
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
        "It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair.",

        "Photosynthesis is the process by which green plants convert sunlight into chemical energy. "
        "Chlorophyll in the leaves absorbs light, driving the synthesis of glucose from CO2 and water.",

        "The Python programming language was created by Guido van Rossum and first released in 1991. "
        "It emphasizes code readability and supports multiple programming paradigms.",

        "The Great Wall of China is a series of fortifications built along the northern borders. "
        "Construction began in the 7th century BC and continued for centuries under various dynasties.",

        "Transformers are a deep learning architecture based on self-attention mechanisms. "
        "They process sequences in parallel and have become the foundation of modern language models.",

        "The mitochondria are membrane-bound organelles found in eukaryotic cells. "
        "They generate most of the cell's supply of ATP through oxidative phosphorylation.",
    };
    int n_docs = sizeof(documents) / sizeof(documents[0]);

    // Encode all documents
    printf("Encoding %d documents...\n", n_docs);
    std::vector<token_embeddings> doc_embs;
    for (int i = 0; i < n_docs; i++) {
        doc_embs.push_back(encode(session, n_embd, documents[i]));
        printf("  doc[%d]: %d tokens\n", i, doc_embs.back().n_tokens);
    }

    // ── Queries ─────────────────────────────────────────────────────────

    const char *queries[] = {
        "When was the Eiffel Tower built?",
        "How do plants make energy from sunlight?",
        "Who created Python?",
        "Tell me about the Great Wall",
        "What are transformers in deep learning?",
        "What do mitochondria do?",
    };
    int n_queries = sizeof(queries) / sizeof(queries[0]);

    printf("\n=== ColBERT MaxSim Retrieval ===\n\n");

    int correct = 0;

    for (int q = 0; q < n_queries; q++) {
        auto query_emb = encode(session, n_embd, queries[q]);

        // Score against all documents
        std::vector<std::pair<float, int>> scores;
        for (int d = 0; d < n_docs; d++) {
            float score = maxsim(query_emb, doc_embs[d]);
            scores.push_back({score, d});
        }

        // Sort by score descending
        std::sort(scores.begin(), scores.end(),
                  [](const auto &a, const auto &b) { return a.first > b.first; });

        int top_doc = scores[0].second;
        bool hit = (top_doc == q); // expected: query i matches doc i
        if (hit) correct++;

        printf("Query: \"%s\"\n", queries[q]);
        printf("  Rank  Score   Document\n");
        for (int r = 0; r < n_docs; r++) {
            int di = scores[r].second;
            printf("  #%-4d %6.1f   [%d] %.60s%s%s\n",
                   r + 1, scores[r].first, di,
                   documents[di],
                   std::strlen(documents[di]) > 60 ? "..." : "",
                   (r == 0 && hit) ? "  <-- correct" : (r == 0 ? "  <-- WRONG" : ""));
        }
        printf("\n");
    }

    printf("Accuracy: %d/%d (%.0f%%)\n\n", correct, n_queries,
           (double)correct / n_queries * 100.0);

    lfg_session_free(session);
    lfg_model_free(model);
    return 0;
}
