// Compare token embeddings, output_norm weights, and final RMSNorm activations
// between two models given the same prompt.
//
// Usage: compare_rmsnorm <model_a> <model_b> [prompt] [n_threads]

#include "lfg_inference.h"
#include "lfg_model.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

struct model_ctx {
    lfg_model   * model = nullptr;
    lfg_context * ctx   = nullptr;

    ~model_ctx() {
        if (ctx)   lfg_free(ctx);
        if (model) lfg_model_free(model);
    }
};

static bool init(model_ctx & mc, const char * path, int n_threads) {
    lfg_model_params mp = lfg_model_default_params();
    mp.n_gpu_layers = 0;

    mc.model = lfg_model_load_from_file(path, mp);
    if (!mc.model) {
        fprintf(stderr, "Failed to load model: %s\n", path);
        return false;
    }

    lfg_context_params cp = lfg_context_default_params();
    cp.n_ctx     = 2048;
    cp.n_batch   = 512;
    cp.n_threads = n_threads;
    cp.embeddings = true;

    mc.ctx = lfg_init_from_model(mc.model, cp);
    if (!mc.ctx) {
        fprintf(stderr, "Failed to create context for: %s\n", path);
        return false;
    }
    return true;
}

static std::vector<lfg_token> tokenize(const lfg_vocab * vocab, const std::string & text) {
    std::vector<lfg_token> tokens(text.size() + 16);
    int n = lfg_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                         tokens.data(), (int32_t)tokens.size(), true, true);
    if (n < 0) {
        tokens.resize(-n);
        n = lfg_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                         tokens.data(), (int32_t)tokens.size(), true, true);
    }
    tokens.resize(n);
    return tokens;
}

// Read tensor data into a float vector (handles f16/f32/quantized)
static std::vector<float> read_tensor_floats(ggml_tensor * t) {
    int64_t n = ggml_nelements(t);
    std::vector<float> data(n);

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));
    } else {
        // For non-f32 types, read raw bytes then dequantize
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());

        const auto * type_traits = ggml_get_type_traits(t->type);
        if (type_traits->to_float) {
            type_traits->to_float(raw.data(), data.data(), n);
        } else {
            fprintf(stderr, "Warning: cannot dequantize type %s, zeroing\n", ggml_type_name(t->type));
            std::fill(data.begin(), data.end(), 0.0f);
        }
    }
    return data;
}

struct compare_result {
    float cos_sim;
    float mae;
    float rmse;
    float max_abs_diff;
    int   max_diff_idx;
    float val_a_at_max;
    float val_b_at_max;
    bool  identical;
};

static compare_result compare_vectors(const float * a, int n_a, const float * b, int n_b) {
    compare_result r = {};
    int n = std::min(n_a, n_b);

    float dot = 0, na = 0, nb = 0, sad = 0, ssd = 0;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        float ad = fabsf(d);
        sad += ad;
        ssd += d * d;
        if (ad > r.max_abs_diff) {
            r.max_abs_diff = ad;
            r.max_diff_idx = i;
        }
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }

    r.cos_sim = (na > 0 && nb > 0) ? dot / (sqrtf(na) * sqrtf(nb)) : 0.0f;
    r.mae  = sad / n;
    r.rmse = sqrtf(ssd / n);
    r.val_a_at_max = a[r.max_diff_idx];
    r.val_b_at_max = b[r.max_diff_idx];
    r.identical = (n_a == n_b) && (r.max_abs_diff == 0.0f);
    return r;
}

static void print_compare(const char * section, const compare_result & r, int n) {
    printf("  Dims compared:     %d\n", n);
    printf("  Cosine similarity: %.8f\n", r.cos_sim);
    printf("  MAE:               %.8f\n", r.mae);
    printf("  RMSE:              %.8f\n", r.rmse);
    printf("  Max |diff|:        %.8f  (dim %d: A=%.6f, B=%.6f)\n",
           r.max_abs_diff, r.max_diff_idx, r.val_a_at_max, r.val_b_at_max);
    printf("  Identical:         %s\n", r.identical ? "YES" : "NO");
}

static void print_first_n(const char * label, const float * v, int n, int show) {
    printf("  [%s] first %d: ", label, show);
    for (int i = 0; i < show && i < n; i++) printf("%.6f ", v[i]);
    printf("\n");
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_a> <model_b> [prompt] [n_threads]\n", argv[0]);
        return 1;
    }

    const char * path_a    = argv[1];
    const char * path_b    = argv[2];
    std::string  prompt    = argc > 3 ? argv[3] : "The capital of France is";
    int          n_threads = argc > 4 ? std::stoi(argv[4]) : 4;

    fprintf(stderr, "Model A: %s\n", path_a);
    fprintf(stderr, "Model B: %s\n", path_b);
    fprintf(stderr, "Prompt:  \"%s\"\n", prompt.c_str());

    ggml_backend_load_all();

    model_ctx a, b;
    if (!init(a, path_a, n_threads)) return 1;
    if (!init(b, path_b, n_threads)) return 1;

    int n_embd_a  = lfg_model_n_embd(a.model);
    int n_embd_b  = lfg_model_n_embd(b.model);
    int n_layer_a = lfg_model_n_layer(a.model);
    int n_layer_b = lfg_model_n_layer(b.model);
    int n_vocab_a = lfg_vocab_n_tokens(lfg_model_get_vocab(a.model));
    int n_vocab_b = lfg_vocab_n_tokens(lfg_model_get_vocab(b.model));

    printf("\n=== Model Info ===\n");
    printf("  A: n_embd=%d  n_layer=%d  n_vocab=%d\n", n_embd_a, n_layer_a, n_vocab_a);
    printf("  B: n_embd=%d  n_layer=%d  n_vocab=%d\n", n_embd_b, n_layer_b, n_vocab_b);

    // ======================================================
    // 1. Compare tok_embd weights (embedding table)
    // ======================================================
    printf("\n=== 1. Token Embedding Weights (tok_embd) ===\n");
    {
        ggml_tensor * te_a = a.model->tok_embd;
        ggml_tensor * te_b = b.model->tok_embd;

        printf("  A: shape [%lld, %lld]  type=%s\n", (long long)te_a->ne[0], (long long)te_a->ne[1], ggml_type_name(te_a->type));
        printf("  B: shape [%lld, %lld]  type=%s\n", (long long)te_b->ne[0], (long long)te_b->ne[1], ggml_type_name(te_b->type));

        auto data_a = read_tensor_floats(te_a);
        auto data_b = read_tensor_floats(te_b);

        // Compare row-by-row for the first few tokens
        int n_rows = std::min((int)te_a->ne[1], (int)te_b->ne[1]);
        int n_cols = std::min((int)te_a->ne[0], (int)te_b->ne[0]);
        int show_rows = std::min(n_rows, 8);

        printf("\n  Per-token embedding comparison (first %d tokens):\n", show_rows);
        printf("  %6s  %12s  %12s  %12s\n", "token", "cos_sim", "mae", "max|diff|");
        for (int r = 0; r < show_rows; r++) {
            auto cr = compare_vectors(
                data_a.data() + r * te_a->ne[0], n_cols,
                data_b.data() + r * te_b->ne[0], n_cols);
            printf("  %6d  %12.8f  %12.8f  %12.8f\n", r, cr.cos_sim, cr.mae, cr.max_abs_diff);
        }

        // Full embedding table comparison (all shared elements)
        int n_shared = std::min((int)ggml_nelements(te_a), (int)ggml_nelements(te_b));
        auto full = compare_vectors(data_a.data(), n_shared, data_b.data(), n_shared);
        printf("\n  Full table comparison (%d elements):\n", n_shared);
        print_compare("tok_embd", full, n_shared);
    }

    // ======================================================
    // 2. Compare output_norm weights (final RMSNorm gamma)
    // ======================================================
    printf("\n=== 2. Output Norm Weights (output_norm) ===\n");
    {
        ggml_tensor * on_a = a.model->output_norm;
        ggml_tensor * on_b = b.model->output_norm;

        if (on_a && on_b) {
            printf("  A: shape [%lld]  type=%s\n", (long long)on_a->ne[0], ggml_type_name(on_a->type));
            printf("  B: shape [%lld]  type=%s\n", (long long)on_b->ne[0], ggml_type_name(on_b->type));

            auto data_a = read_tensor_floats(on_a);
            auto data_b = read_tensor_floats(on_b);

            print_first_n("A", data_a.data(), (int)data_a.size(), 16);
            print_first_n("B", data_b.data(), (int)data_b.size(), 16);

            int n = std::min((int)data_a.size(), (int)data_b.size());
            auto r = compare_vectors(data_a.data(), (int)data_a.size(), data_b.data(), (int)data_b.size());
            print_compare("output_norm", r, n);
        } else {
            printf("  (one or both models missing output_norm tensor)\n");
        }
    }

    // ======================================================
    // 3. Per-layer transformer weight comparison
    // ======================================================
    printf("\n=== 3. Per-Layer Weight Comparison ===\n");
    {
        int n_layers = std::min(n_layer_a, n_layer_b);

        struct tensor_pair {
            const char * name;
            ggml_tensor * lfg_layer::* field;
        };

        tensor_pair pairs[] = {
            { "attn_norm", &lfg_layer::attn_norm },
            { "wq",        &lfg_layer::wq },
            { "wk",        &lfg_layer::wk },
            { "wv",        &lfg_layer::wv },
            { "wo",        &lfg_layer::wo },
            { "ffn_norm",  &lfg_layer::ffn_norm },
            { "ffn_gate",  &lfg_layer::ffn_gate },
            { "ffn_down",  &lfg_layer::ffn_down },
            { "ffn_up",    &lfg_layer::ffn_up },
        };

        // Header
        printf("  %5s  %-10s  %12s  %12s  %12s\n", "layer", "tensor", "cos_sim", "mae", "max|diff|");
        printf("  %5s  %-10s  %12s  %12s  %12s\n", "-----", "----------", "------------", "------------", "------------");

        for (int il = 0; il < n_layers; il++) {
            for (auto & p : pairs) {
                ggml_tensor * ta = a.model->layers[il].*(p.field);
                ggml_tensor * tb = b.model->layers[il].*(p.field);

                if (!ta || !tb) continue;

                auto da = read_tensor_floats(ta);
                auto db = read_tensor_floats(tb);

                int n = std::min((int)da.size(), (int)db.size());
                auto r = compare_vectors(da.data(), (int)da.size(), db.data(), (int)db.size());

                printf("  %5d  %-10s  %12.8f  %12.8f  %12.8f\n",
                       il, p.name, r.cos_sim, r.mae, r.max_abs_diff);
            }
        }

        // Also check MoE layers if present
        bool has_moe = false;
        for (int il = 0; il < n_layers; il++) {
            if (a.model->layers[il].ffn_gate_inp && b.model->layers[il].ffn_gate_inp) {
                has_moe = true;
                break;
            }
        }
        if (has_moe) {
            printf("\n  MoE expert weights:\n");
            printf("  %5s  %-14s  %12s  %12s  %12s\n", "layer", "tensor", "cos_sim", "mae", "max|diff|");
            printf("  %5s  %-14s  %12s  %12s  %12s\n", "-----", "--------------", "------------", "------------", "------------");
            for (int il = 0; il < n_layers; il++) {
                tensor_pair moe_pairs[] = {
                    { "ffn_gate_inp",  &lfg_layer::ffn_gate_inp },
                    { "ffn_gate_exps", &lfg_layer::ffn_gate_exps },
                    { "ffn_down_exps", &lfg_layer::ffn_down_exps },
                    { "ffn_up_exps",   &lfg_layer::ffn_up_exps },
                };
                for (auto & p : moe_pairs) {
                    ggml_tensor * ta = a.model->layers[il].*(p.field);
                    ggml_tensor * tb = b.model->layers[il].*(p.field);
                    if (!ta || !tb) continue;

                    auto da = read_tensor_floats(ta);
                    auto db = read_tensor_floats(tb);
                    int n = std::min((int)da.size(), (int)db.size());
                    auto r = compare_vectors(da.data(), (int)da.size(), db.data(), (int)db.size());
                    printf("  %5d  %-14s  %12.8f  %12.8f  %12.8f\n",
                           il, p.name, r.cos_sim, r.mae, r.max_abs_diff);
                }
            }
        }
    }

    // ======================================================
    // 4. Compare final RMSNorm activations (result_norm)
    // ======================================================
    printf("\n=== 4. RMSNorm Activations (result_norm) — post-decode ===\n");
    {
        const lfg_vocab * vocab_a = lfg_model_get_vocab(a.model);
        const lfg_vocab * vocab_b = lfg_model_get_vocab(b.model);

        auto tokens_a = tokenize(vocab_a, prompt);
        auto tokens_b = tokenize(vocab_b, prompt);

        printf("  Tokens A: %zu  B: %zu  identical: %s\n",
               tokens_a.size(), tokens_b.size(),
               (tokens_a == tokens_b) ? "YES" : "NO");

        lfg_batch batch_a = lfg_batch_get_one(tokens_a.data(), (int32_t)tokens_a.size());
        lfg_batch batch_b = lfg_batch_get_one(tokens_b.data(), (int32_t)tokens_b.size());

        if (lfg_decode(a.ctx, batch_a)) { fprintf(stderr, "decode A failed\n"); return 1; }
        if (lfg_decode(b.ctx, batch_b)) { fprintf(stderr, "decode B failed\n"); return 1; }

        float * embd_a = lfg_get_embeddings_ith(a.ctx, -1);
        float * embd_b = lfg_get_embeddings_ith(b.ctx, -1);

        if (!embd_a || !embd_b) {
            fprintf(stderr, "Failed to get embeddings\n");
            return 1;
        }

        print_first_n("A", embd_a, n_embd_a, 16);
        print_first_n("B", embd_b, n_embd_b, 16);

        int n = std::min(n_embd_a, n_embd_b);
        auto r = compare_vectors(embd_a, n_embd_a, embd_b, n_embd_b);
        print_compare("result_norm", r, n);
    }

    return 0;
}
