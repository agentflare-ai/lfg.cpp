#pragma once

#include "lfm_inference.h"
#include <vector>
#include <memory>
#include <string>

namespace liquid {

class InferenceCore {
public:
    // Sampling parameters (mirrors llama.cpp common_params_sampling)
    struct SamplingConfig {
        uint32_t seed              = 0xFFFFFFFF; // LFM_DEFAULT_SEED
        int32_t  n_prev            = 64;         // number of previous tokens to remember
        int32_t  top_k             = 40;         // <= 0 to use vocab size
        float    top_p             = 0.95f;      // 1.0 = disabled
        float    min_p             = 0.05f;      // 0.0 = disabled
        float    typ_p             = 1.00f;      // typical_p, 1.0 = disabled
        float    temp              = 0.80f;      // <= 0.0 to sample greedily
        int32_t  penalty_last_n    = 64;         // last n tokens to penalize (0 = disable penalty)
        float    penalty_repeat    = 1.00f;      // 1.0 = disabled
        float    penalty_freq      = 0.00f;      // 0.0 = disabled
        float    penalty_present   = 0.00f;      // 0.0 = disabled
        int32_t  mirostat          = 0;          // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
        float    mirostat_tau      = 5.00f;      // target entropy
        float    mirostat_eta      = 0.10f;      // learning rate
        std::string grammar;                     // optional BNF-like grammar to constrain sampling
    };

    struct Config {
        int n_threads = 4;
        int n_ctx = 2048;
        int n_batch = 512;
        bool enable_healing = false; // Enable checkpointing for token healing
        bool structured_checkpointing = true; // Snapshot sampler state for structured decoding
        int reasoning_budget = 0; // 0 = disabled. Number of tokens allowed for reasoning.
        SamplingConfig sampling;
    };

    struct RestoreOptions {
        bool restore_sampler_state = true;
        bool restore_grammar = true;
    };

    struct Checkpoint {
        int n_past = 0;
        size_t token_count = 0;
        size_t sampler_count = 0;
        uint64_t rng_state = 0; // Placeholder if we need to save RNG state later
        std::vector<uint8_t> state_data; // Full model state snapshot
        std::string grammar_str;
        std::string grammar_root;
        std::shared_ptr<lfm_sampler> sampler_state;
        size_t reasoning_token_count = 0;
    };

    InferenceCore(lfm_model* model, const Config& config);
    ~InferenceCore();

    // Reset context state
    void Reset();

    // Create a lightweight snapshot of the current state
    Checkpoint CreateCheckpoint() const;

    // Restore state to a previous checkpoint
    // Returns true on success
    bool RestoreCheckpoint(const Checkpoint& cp);
    bool RestoreCheckpoint(const Checkpoint& cp, const RestoreOptions& options);

    // Ingest embeddings (e.g. from an external encoder)
    // returns true on success
    bool IngestEmbeddings(const std::vector<float>& embeddings, int n_tokens);

    // Ingest tokens (e.g. prompt or previous generation)
    bool IngestTokens(const std::vector<lfm_token>& tokens, bool update_sampler = true);

    // Run forward pass / Decode
    // Returns true if verification/decode succeeded
    bool Decode();

    // Sample next token
    // Returns the sampled token id
    lfm_token Sample();

    // Get logits for the last token
    std::vector<float> GetLogits() const;

    // Configure structured decoding (GBNF grammar)
    // Pass empty string to disable
    void ConfigureStructuredDecoding(const std::string& grammar, const std::string& root_rule = "root");

    // Configure tokens that delimit a reasoning/thinking block
    // Constraints are suspended while inside these blocks
    void ConfigureReasoning(const std::vector<lfm_token>& start_tokens, const std::vector<lfm_token>& end_tokens);

    // Attempts to heal the token boundary by backtracking the last token and resampling
    // with a prefix constraint. Returns true if healing occurred (token changed).
    bool HealLastToken();

private:
    lfm_model* model_;
    lfm_context* ctx_;
    Config config_;
    int n_past_ = 0;
    
    // State tracking
    std::vector<lfm_token> token_history_;
    
    // Reasoning state
    std::vector<lfm_token> reasoning_start_tokens_;
    std::vector<lfm_token> reasoning_end_tokens_;
    bool in_reasoning_ = false;
    
    // Reasoning Budget State
    size_t reasoning_token_count_ = 0;
    int forcing_reasoning_end_index_ = -1; // -1 if not forcing, >= 0 is index into reasoning_end_tokens_

    // Sampling state
    struct lfm_sampler* sampler_ = nullptr;
    struct lfm_sampler* prefix_sampler_ = nullptr;
    std::string grammar_str_;
    std::string grammar_root_;

    // Sampler history tracking (only tokens accepted by sampler)
    std::vector<lfm_token> sampler_history_;
    std::vector<size_t> sampler_offsets_;
    size_t pending_sampler_accepts_ = 0;
    bool sampler_recording_enabled_ = true;
    
    // Healing state
    std::vector<uint8_t> healing_state_buffer_;
    int healing_n_past_ = -1;
    std::shared_ptr<lfm_sampler> healing_sampler_snapshot_;

    // Scratch buffers to avoid heap allocations in hot paths
    std::vector<lfm_pos> pos_buf_;
    std::vector<int8_t> logits_buf_;

    bool IngestInternal(const std::vector<lfm_token>& tokens, bool update_sampler);
    void RebuildSampler();
    void RecordSamplerToken(lfm_token token, bool pending);
    bool ShouldRecordSamplerToken(lfm_token token) const;
    void RebuildSamplerFromHistory(bool full_reingest);
    void TruncateHistory(size_t new_size);
    std::shared_ptr<lfm_sampler> SnapshotSampler() const;
    bool SetSamplerFromSnapshot(const std::shared_ptr<lfm_sampler>& snapshot);
    void UpdatePrefixSampler();
    struct lfm_sampler* FindSamplerByName(const char * name) const;
};

} // namespace liquid
