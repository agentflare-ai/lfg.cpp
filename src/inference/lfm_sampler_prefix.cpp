#include "lfm_inference.h"
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

struct lfm_sampler_prefix_context {
    const struct lfm_vocab * vocab;
    std::string prefix;
};

static const char * lfm_sampler_prefix_name(const struct lfm_sampler * smpl) {
    (void)smpl;
    return "prefix";
}

static void lfm_sampler_prefix_apply(struct lfm_sampler * smpl, lfm_token_data_array * cur_p) {
    auto * ctx = (lfm_sampler_prefix_context *) smpl->ctx;
    
    if (ctx->prefix.empty()) return;

    for (size_t i = 0; i < cur_p->size; ++i) {
        // Optimization: skip already masked
        if (cur_p->data[i].logit == -INFINITY) continue;

        char buf[256];
        // Note: lstrip=0, special=false
        int n = lfm_token_to_piece(ctx->vocab, cur_p->data[i].id, buf, sizeof(buf), 0, false);
        
        if (n <= 0) {
            // If we have a required prefix, an empty/special token does not contribute to it.
            // So we reject it.
            if (!ctx->prefix.empty()) {
                cur_p->data[i].logit = -INFINITY;
            }
            continue;
        }

        std::string s(buf, n);
        
        // Check 1: Token starts with prefix (e.g. prefix="ht", token="http")
        // Check 2: Prefix starts with token (e.g. prefix="http", token="ht")
        
        bool match = false;
        if (s.size() >= ctx->prefix.size()) {
            if (s.compare(0, ctx->prefix.size(), ctx->prefix) == 0) match = true;
        } else {
            if (ctx->prefix.compare(0, s.size(), s) == 0) match = true;
        }

        if (!match) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
}

static struct lfm_sampler * lfm_sampler_prefix_clone(const struct lfm_sampler * smpl) {
    const auto * ctx = (const lfm_sampler_prefix_context *) smpl->ctx;
    return lfm_sampler_init_prefix(ctx->vocab, ctx->prefix.c_str());
}

static void lfm_sampler_prefix_free(struct lfm_sampler * smpl) {
    delete (lfm_sampler_prefix_context *) smpl->ctx;
}

static struct lfm_sampler_i lfm_sampler_prefix_i = {
    /* .name   = */ lfm_sampler_prefix_name,
    /* .accept = */ NULL,
    /* .apply  = */ lfm_sampler_prefix_apply,
    /* .reset  = */ NULL,
    /* .clone  = */ lfm_sampler_prefix_clone,
    /* .free   = */ lfm_sampler_prefix_free,
    /* .backend_init = */ NULL,
    /* .backend_accept = */ NULL,
    /* .backend_apply = */ NULL,
    /* .backend_set_input = */ NULL
};

struct lfm_sampler * lfm_sampler_init_prefix(const struct lfm_vocab * vocab, const char * prefix) {
    auto * ctx = new lfm_sampler_prefix_context();
    ctx->vocab = vocab;
    ctx->prefix = prefix ? prefix : "";
    
    return lfm_sampler_init(&lfm_sampler_prefix_i, ctx);
}

void lfm_sampler_prefix_set(struct lfm_sampler * smpl, const char * prefix) {
    auto * ctx = (lfm_sampler_prefix_context *) smpl->ctx;
    ctx->prefix = prefix ? prefix : "";
}
