#pragma once

// TODO: rename liquid-sampling.h/.cpp to liquid-sampler.h/.cpp ?

#include "lfm_inference.h"

#include <vector>

struct lfm_vocab;
struct lfm_grammar;

// sampler chain

struct lfm_sampler_chain {
    lfm_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    struct info {
        bool is_backend;

        lfm_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for lfm_sampler_sample to avoid repeated allocations
    std::vector<lfm_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

struct lfm_sampler * lfm_sampler_init_dry_testing(
        int32_t context_size,
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<lfm_token>> & seq_breakers);
