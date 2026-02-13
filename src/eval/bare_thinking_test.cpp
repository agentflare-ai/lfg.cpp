// Bare test: does the 1.2B-Thinking model actually produce thinking text?
// No grammar, no budget, no healing — just raw generation.

#include "lfg_inference.h"
#include "ggml-backend.h"

#include <cstdio>
#include <string>
#include <vector>

static std::string tok2str(const lfg_vocab * v, lfg_token t) {
    char buf[512];
    int n = lfg_token_to_piece(v, t, buf, sizeof(buf), 0, true);
    return n > 0 ? std::string(buf, n) : "";
}

int main() {
    const char * model_path = "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf";

    ggml_backend_load_all();

    lfg_model_params mp = lfg_model_default_params();
    mp.n_gpu_layers = 0;
    lfg_model * model = lfg_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    const lfg_vocab * vocab = lfg_model_get_vocab(model);

    lfg_context_params cp = lfg_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    cp.n_threads = 4;

    lfg_context * ctx = lfg_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); lfg_model_free(model); return 1; }

    auto sparams = lfg_sampler_chain_default_params();
    lfg_sampler * smpl = lfg_sampler_chain_init(sparams);

    // Try multiple sampling configs
    // Config 1: greedy
    // Config 2: temp=0.6 + top_p
    // Config 3: temp=0.8 + top_k
    lfg_sampler_chain_add(smpl, lfg_sampler_init_penalties(64, 1.1f, 0.0f, 0.0f));
    lfg_sampler_chain_add(smpl, lfg_sampler_init_top_k(40));
    lfg_sampler_chain_add(smpl, lfg_sampler_init_top_p(0.9f, 1));
    lfg_sampler_chain_add(smpl, lfg_sampler_init_temp(0.6f));
    lfg_sampler_chain_add(smpl, lfg_sampler_init_dist(42));

    // Build prompt with chat template
    lfg_chat_message msgs[1];
    msgs[0].role = "user";
    msgs[0].content = "What is 2+2? Answer in JSON.";

    char tmpl_buf[2048];
    int32_t tmpl_len = lfg_chat_apply_template("bailing-think", msgs, 1, true, tmpl_buf, sizeof(tmpl_buf));

    printf("Prompt:\n%.*s\n\n", tmpl_len, tmpl_buf);

    // Tokenize
    std::vector<lfg_token> tokens(512);
    int n = lfg_tokenize(vocab, tmpl_buf, tmpl_len, tokens.data(), (int32_t)tokens.size(), true, true);
    tokens.resize(n);

    printf("Tokens: %d\nLast 3: ", n);
    for (int i = std::max(0, n - 3); i < n; i++)
        printf("[%d]'%s' ", tokens[i], tok2str(vocab, tokens[i]).c_str());
    printf("\n\n--- Generation (temp=0.6, rep_pen=1.1, 200 tokens) ---\n");

    // Decode prompt
    lfg_batch batch = lfg_batch_get_one(tokens.data(), n);
    lfg_decode(ctx, batch);

    // Generate
    for (int i = 0; i < 200; i++) {
        lfg_token t = lfg_sampler_sample(smpl, ctx, -1);

        std::string s = tok2str(vocab, t);
        printf("%s", s.c_str());
        fflush(stdout);

        if (lfg_vocab_is_eog(vocab, t)) {
            printf("[EOS]");
            break;
        }

        batch = lfg_batch_get_one(&t, 1);
        lfg_decode(ctx, batch);
    }
    printf("\n");

    lfg_sampler_free(smpl);
    lfg_free(ctx);
    lfg_model_free(model);
    return 0;
}
