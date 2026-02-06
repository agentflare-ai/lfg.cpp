#pragma once

// TODO: rename liquid-sampling.h/.cpp to liquid-sampler.h/.cpp ?

#include "lfg_inference.h"

#include <vector>

struct lfg_vocab;
struct lfg_grammar;

// sampler chain

struct lfg_sampler_chain {
    lfg_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    struct info {
        bool is_backend;

        lfg_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for lfg_sampler_sample to avoid repeated allocations
    std::vector<lfg_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

struct lfg_sampler * lfg_sampler_init_dry_testing(
        int32_t context_size,
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<lfg_token>> & seq_breakers);
