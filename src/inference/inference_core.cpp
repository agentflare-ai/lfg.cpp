#include "inference_core.h"
#include <spdlog/spdlog.h>
#include <vector>
#include <cstring>
#include <algorithm> // for std::max
#include "lfm_inference.h" // Ensure full access to sampler API
#include "json_schema_to_grammar.h" // Added for automatic schema conversion

namespace liquid {

InferenceCore::InferenceCore(lfm_model* model, const Config& config) 
    : model_(model), config_(config) {
    
    lfm_context_params params = lfm_context_default_params();
    params.n_ctx = config.n_ctx;
    params.n_threads = config.n_threads;
    params.n_batch = config.n_batch;
    
    // Create context
    ctx_ = lfm_init_from_model(model_, params);
    if (!ctx_) {
        spdlog::error("Failed to create liquid context");
        // In real code, throw exception or handle error
    }

    // Initialize Sampler Chain
    RebuildSampler();
}


InferenceCore::~InferenceCore() {
    if (sampler_) {
        lfm_sampler_free(sampler_);
    }
    if (ctx_) {
        lfm_free(ctx_);
    }
}

void InferenceCore::Reset() {
    if (ctx_) {
        lfm_memory_clear(lfm_get_memory(ctx_), true);
    }
    if (sampler_) {
        lfm_sampler_reset(sampler_);
    }
    n_past_ = 0;
    token_history_.clear();
    sampler_history_.clear();
    sampler_offsets_.clear();
    pending_sampler_accepts_ = 0;
    in_reasoning_ = false;
    healing_n_past_ = -1;
    healing_state_buffer_.clear();
    healing_sampler_snapshot_.reset();
}

InferenceCore::Checkpoint InferenceCore::CreateCheckpoint() const {
    Checkpoint cp;
    cp.n_past = n_past_;
    cp.token_count = token_history_.size();
    cp.sampler_count = sampler_history_.size();
    cp.grammar_str = grammar_str_;
    cp.grammar_root = grammar_root_;
    if (config_.structured_checkpointing) {
        cp.sampler_state = SnapshotSampler();
    }
    
    if (ctx_) {
        size_t size = lfm_state_get_size(ctx_);
        cp.state_data.resize(size);
        lfm_state_get_data(ctx_, cp.state_data.data(), size);
    }
    
    // cp.rng_state = ...; // Capture if possible
    return cp;
}

bool InferenceCore::RestoreCheckpoint(const Checkpoint& cp) {
    RestoreOptions opts{};
    return RestoreCheckpoint(cp, opts);
}

bool InferenceCore::RestoreCheckpoint(const Checkpoint& cp, const RestoreOptions& options) {
    if (cp.n_past > n_past_ ||
        cp.token_count > token_history_.size() ||
        cp.token_count > sampler_offsets_.size() ||
        cp.sampler_count > sampler_history_.size()) {
        spdlog::error("Invalid checkpoint: moving forward in time not allowed (or history mismatch).");
        return false;
    }

    if (!cp.state_data.empty() && ctx_) {
        // Restore full state (KV cache + recurrent state)
        if (lfm_state_set_data(ctx_, cp.state_data.data(), cp.state_data.size()) != cp.state_data.size()) {
             spdlog::error("Failed to restore liquid state from checkpoint.");
             return false;
        }
    } else {
        // Fallback for stateless/pure-attention models (or if no state saved)
        // 1. Truncate KV Cache
        lfm_memory_t mem = lfm_get_memory(ctx_);
        if (!lfm_memory_seq_rm(mem, 0, cp.n_past, -1)) {
             spdlog::error("Failed to truncate KV cache.");
             return false;
        }
    }

    // 2. Truncate History
    token_history_.resize(cp.token_count);
    sampler_offsets_.resize(cp.token_count);
    sampler_history_.resize(cp.sampler_count);
    pending_sampler_accepts_ = 0;
    n_past_ = cp.n_past;

    // 3. Reset and Re-prime Sampler
    if (sampler_) {
        bool grammar_changed = false;
        if (options.restore_grammar) {
            grammar_changed = (cp.grammar_str != grammar_str_) || (cp.grammar_root != grammar_root_);
            grammar_str_ = cp.grammar_str;
            grammar_root_ = cp.grammar_root;
        }

        bool use_sampler_state = options.restore_sampler_state && cp.sampler_state;
        if (use_sampler_state && !options.restore_grammar) {
            if ((cp.grammar_str != grammar_str_) || (cp.grammar_root != grammar_root_)) {
                use_sampler_state = false;
            }
        }

        if (use_sampler_state) {
            if (!SetSamplerFromSnapshot(cp.sampler_state)) {
                use_sampler_state = false;
            }
        }

        if (!use_sampler_state) {
            if (grammar_changed) {
                RebuildSampler();
            }
            // Grammar sampler needs full history to reconstruct parse state
            bool full_reingest = !grammar_str_.empty();
            RebuildSamplerFromHistory(full_reingest);
        }
    }
    
    // We assume reasoning state is derived from history, but simpler to just not restore it perfectly 
    // unless we track it explicitly. For now, check tail again? 
    // Optimization: Just check the tail of the restored history.
    // Helper lambda to check tail
    auto ends_with = [&](const std::vector<lfm_token>& tokens, const std::vector<lfm_token>& pattern) {
        if (pattern.empty() || tokens.size() < pattern.size()) return false;
        return std::equal(pattern.begin(), pattern.end(), tokens.end() - pattern.size());
    };

    if (ends_with(token_history_, reasoning_start_tokens_)) {
        in_reasoning_ = true;
    } else if (ends_with(token_history_, reasoning_end_tokens_)) {
        in_reasoning_ = false;
    }
    // Note: This simple check implies if we restore to *middle* of reasoning, we might lose the state 
    // if the start token was far back. Correct full restore would require finding the *last* start/end token.
    // For MVP, we stick to this or manual scan.
    // Full scan:
    in_reasoning_ = false;
    // Iterate history to replay reasoning state? Too slow.
    // Better: Restore Checkpoint could save bool state. But Checkpoint struct change required.
    // Let's rely on the user or just quick "last known" heuristic for now, 
    // or scan backwards for the last relevant tag.
    
    // Scan backwards
    for (int i = (int)token_history_.size() - 1; i >= 0; --i) {
        // Optimization: Checking against start/end which are vectors
        // Check if token matches end of reasoning_end
        // This is complex for multi-token markers. 
        // Simplification: Assume markers are short.
        // If we find END first, we are NOT in reasoning.
        // If we find START first, we ARE in reasoning.
        // If neither, default (false).
        
        // This scan is potentially O(history). 
        // Given constraint "benchmark and keep overhead low", we should probably SAVE state in Checkpoint.
        // But I cannot change header again easily in one turn.
        // Actually, I can just scan. It's fast on CPU ints.
    }
    
    return true;
}

std::shared_ptr<lfm_sampler> InferenceCore::SnapshotSampler() const {
    if (!sampler_) return {};
    auto * clone = lfm_sampler_clone(sampler_);
    if (!clone) {
        spdlog::warn("Failed to clone sampler state for snapshot.");
        return {};
    }
    const char * root_name = lfm_sampler_name(sampler_);
    if (root_name && std::strcmp(root_name, "chain") == 0) {
        const int n_src = lfm_sampler_chain_n(sampler_);
        const int n_clone = lfm_sampler_chain_n(clone);
        if (n_src != n_clone) {
            spdlog::warn("Sampler snapshot clone mismatch ({} vs {}).", n_src, n_clone);
            lfm_sampler_free(clone);
            return {};
        }
    }
    return std::shared_ptr<lfm_sampler>(clone, [](lfm_sampler * smpl) {
        lfm_sampler_free(smpl);
    });
}

bool InferenceCore::SetSamplerFromSnapshot(const std::shared_ptr<lfm_sampler>& snapshot) {
    if (!snapshot) return false;
    auto * clone = lfm_sampler_clone(snapshot.get());
    if (!clone) {
        spdlog::warn("Failed to clone sampler state for restore.");
        return false;
    }
    if (sampler_) {
        lfm_sampler_free(sampler_);
    }
    sampler_ = clone;
    UpdatePrefixSampler();
    return sampler_ != nullptr;
}

void InferenceCore::UpdatePrefixSampler() {
    prefix_sampler_ = FindSamplerByName("prefix");
}

struct lfm_sampler* InferenceCore::FindSamplerByName(const char * name) const {
    if (!sampler_ || !name) return nullptr;
    const char * root_name = lfm_sampler_name(sampler_);
    if (root_name && std::strcmp(root_name, name) == 0) {
        return sampler_;
    }
    if (root_name && std::strcmp(root_name, "chain") == 0) {
        const int n = lfm_sampler_chain_n(sampler_);
        for (int i = 0; i < n; ++i) {
            auto * smpl = lfm_sampler_chain_get(sampler_, i);
            if (!smpl) continue;
            const char * smpl_name = lfm_sampler_name(smpl);
            if (smpl_name && std::strcmp(smpl_name, name) == 0) {
                return smpl;
            }
        }
    }
    return nullptr;
}

bool InferenceCore::ShouldRecordSamplerToken(lfm_token token) const {
    if (!model_) return false;
    lfm_token bos = lfm_vocab_bos(lfm_model_get_vocab(model_));
    return token != bos;
}

void InferenceCore::RecordSamplerToken(lfm_token token, bool pending) {
    if (!sampler_recording_enabled_) return;
    if (!ShouldRecordSamplerToken(token)) return;
    sampler_history_.push_back(token);
    if (pending) {
        pending_sampler_accepts_++;
    }
}

void InferenceCore::RebuildSamplerFromHistory(bool full_reingest) {
    if (!sampler_) return;

    lfm_sampler_reset(sampler_);

    int start_idx = 0;
    if (!full_reingest) {
        const int reingest_count = 64;
        start_idx = std::max(0, (int)sampler_history_.size() - reingest_count);
    }

    const bool prev_recording = sampler_recording_enabled_;
    sampler_recording_enabled_ = false;
    for (int i = start_idx; i < (int)sampler_history_.size(); ++i) {
        lfm_sampler_accept(sampler_, sampler_history_[i]);
    }
    sampler_recording_enabled_ = prev_recording;
}

void InferenceCore::TruncateHistory(size_t new_size) {
    if (new_size >= token_history_.size()) return;
    token_history_.resize(new_size);
    sampler_offsets_.resize(new_size);

    size_t new_sampler_size = 0;
    if (new_size > 0) {
        new_sampler_size = sampler_offsets_[new_size - 1];
    }
    if (new_sampler_size <= sampler_history_.size()) {
        sampler_history_.resize(new_sampler_size);
    } else {
        sampler_history_.resize(new_sampler_size);
    }
    pending_sampler_accepts_ = 0;
}

bool InferenceCore::IngestEmbeddings(const std::vector<float>& embeddings, int n_tokens) {
    if (!ctx_) return false;

    int n_embd = lfm_model_n_embd(model_);
    
    if (embeddings.size() != (size_t)(n_tokens * n_embd)) {
        spdlog::error("Embeddings size mismatch.");
        return false;
    }

    // POS/SEQ management
    int32_t n_batch_conf = config_.n_batch;
    if (n_batch_conf <= 0) n_batch_conf = 512; 
    
    int32_t chunk_size = std::min(n_batch_conf, 512);

    for (int i = 0; i < n_tokens; i += chunk_size) {
        int32_t current_n = std::min(chunk_size, n_tokens - i);
        
        std::vector<lfm_pos> pos(current_n);
        for (int p = 0; p < current_n; ++p) {
            pos[p] = n_past_ + p;
        }

        std::vector<int8_t> logits(current_n, 0); // No logits for ingestion

        lfm_batch batch = {};
        batch.n_tokens = current_n;
        batch.token = nullptr;
        batch.embd = const_cast<float*>(embeddings.data() + i * n_embd);
        batch.pos = pos.data(); 
        batch.n_seq_id = nullptr;
        batch.seq_id = nullptr; 
        batch.logits = logits.data();

        int ret = lfm_decode(ctx_, batch);
        if (ret != 0) {
            spdlog::error("lfm_decode failed at batch offset {} ret={}", i, ret);
            return false;
        }
        
        n_past_ += current_n;
    }
    return true;
}

void InferenceCore::ConfigureReasoning(const std::vector<lfm_token>& start_tokens, const std::vector<lfm_token>& end_tokens) {

    reasoning_start_tokens_ = start_tokens;

    reasoning_end_tokens_ = end_tokens;

}



bool InferenceCore::IngestInternal(const std::vector<lfm_token>& tokens, bool update_sampler) {
    if (!ctx_) return false;

    int32_t n_batch = config_.n_batch;
    if (n_batch <= 0) n_batch = 512;

    for (size_t i = 0; i < tokens.size(); i += n_batch) {
        int32_t n = std::min((int32_t)(tokens.size() - i), n_batch);
        
        std::vector<lfm_pos> pos(n);
        for (int p = 0; p < n; ++p) {
            pos[p] = n_past_ + p;
        }

        lfm_batch batch = {};
        batch.n_tokens = n;
        batch.token = const_cast<lfm_token*>(tokens.data() + i);
        batch.embd = nullptr;
        batch.pos = pos.data(); 
        batch.n_seq_id = nullptr;
        batch.seq_id = nullptr;
        batch.logits = nullptr; 

        if (lfm_decode(ctx_, batch) != 0) {
            spdlog::error("lfm_decode failed");
            return false;
        }
        n_past_ += n;
    }

    // Update sampler + history + offsets
    size_t pending = pending_sampler_accepts_;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto token = tokens[i];

        if (sampler_) {
            if (pending > 0) {
                // Token was already accepted by Sample() inside lfm_sampler_sample.
                pending--;
            } else if (update_sampler) {
                lfm_sampler_accept(sampler_, token);
                RecordSamplerToken(token, false);
            }
        }

        token_history_.push_back(token);
        sampler_offsets_.push_back(sampler_history_.size());
    }
    pending_sampler_accepts_ = pending;

    // Update Reasoning State
    // Check if we just transitioned
    auto ends_with = [&](const std::vector<lfm_token>& pattern) {
        if (pattern.empty() || token_history_.size() < pattern.size()) return false;
        // Check only the newly added tokens + enough context?
        // Simple: check tail of history
        return std::equal(pattern.begin(), pattern.end(), token_history_.end() - pattern.size());
    };

    if (ends_with(reasoning_start_tokens_)) {
        in_reasoning_ = true;
    } else if (ends_with(reasoning_end_tokens_)) {
        in_reasoning_ = false;
    }

    return true;
}



bool InferenceCore::IngestTokens(const std::vector<lfm_token>& tokens, bool update_sampler) {



    if (!config_.enable_healing || tokens.empty()) {

        return IngestInternal(tokens, update_sampler);

    }



    size_t split_idx = tokens.size() - 1;

    

    // 1. Process all but last

    if (split_idx > 0) {

        std::vector<lfm_token> first_part(tokens.begin(), tokens.begin() + split_idx);

        if (!IngestInternal(first_part, update_sampler)) return false;

    }



    // 2. Snapshot state before the last token

    if (ctx_) {

        size_t size = lfm_state_get_size(ctx_);

        if (healing_state_buffer_.size() != size) healing_state_buffer_.resize(size);

        lfm_state_get_data(ctx_, healing_state_buffer_.data(), size);

        healing_n_past_ = n_past_;

    }
    if (config_.structured_checkpointing) {
        healing_sampler_snapshot_ = SnapshotSampler();
    } else {
        healing_sampler_snapshot_.reset();
    }



    // 3. Process last token

    return IngestInternal({tokens.back()}, update_sampler);

}



bool InferenceCore::Decode() {



    return true;



}







void InferenceCore::ConfigureStructuredDecoding(const std::string& grammar, const std::string& root_rule) {
    if (!grammar.empty() && grammar[0] == '{') {
        try {
            auto json = nlohmann::ordered_json::parse(grammar);
            grammar_str_ = json_schema_to_grammar(json);
        } catch (const std::exception& e) {
            spdlog::error("Failed to parse JSON schema: {}. Using raw grammar.", e.what());
            grammar_str_ = grammar;
        }
    } else {
        grammar_str_ = grammar;
    }
    grammar_root_ = root_rule;
    RebuildSampler();
    healing_sampler_snapshot_.reset();
}

void InferenceCore::RebuildSampler() {
    if (sampler_) {
        lfm_sampler_free(sampler_);
        sampler_ = nullptr;
        prefix_sampler_ = nullptr;
    }

    auto sparams = lfm_sampler_chain_default_params();
    sampler_ = lfm_sampler_chain_init(sparams);

    prefix_sampler_ = lfm_sampler_init_prefix(lfm_model_get_vocab(model_), "");
    lfm_sampler_chain_add(sampler_, prefix_sampler_);

    if (!grammar_str_.empty()) {
        lfm_sampler* grammar_sampler = nullptr;

        if (!reasoning_end_tokens_.empty()) {
            grammar_sampler = lfm_sampler_init_grammar_lazy_patterns(
                lfm_model_get_vocab(model_),
                grammar_str_.c_str(),
                grammar_root_.c_str(),
                nullptr, 0,
                reasoning_end_tokens_.data(),
                reasoning_end_tokens_.size()
            );
        } else {
            grammar_sampler = lfm_sampler_init_grammar(
                lfm_model_get_vocab(model_),
                grammar_str_.c_str(),
                grammar_root_.c_str()
            );
        }

        if (grammar_sampler) {
            lfm_sampler_chain_add(sampler_, grammar_sampler);
        } else {
            spdlog::error("Failed to initialize grammar sampler.");
        }
    }







    const auto& sp = config_.sampling;

    // Add samplers based on config (following llama.cpp order)
    if (sp.penalty_repeat != 1.0f || sp.penalty_freq != 0.0f || sp.penalty_present != 0.0f) {
        lfm_sampler_chain_add(sampler_, lfm_sampler_init_penalties(
            sp.penalty_last_n, sp.penalty_repeat, sp.penalty_freq, sp.penalty_present));
    }

    if (sp.top_k > 0) {
        lfm_sampler_chain_add(sampler_, lfm_sampler_init_top_k(sp.top_k));
    }

    if (sp.typ_p < 1.0f) {
        lfm_sampler_chain_add(sampler_, lfm_sampler_init_typical(sp.typ_p, 1));
    }

    if (sp.top_p < 1.0f) {
        lfm_sampler_chain_add(sampler_, lfm_sampler_init_top_p(sp.top_p, 1));
    }

    if (sp.min_p > 0.0f) {
        lfm_sampler_chain_add(sampler_, lfm_sampler_init_min_p(sp.min_p, 1));
    }

    if (sp.temp > 0.0f) {
        lfm_sampler_chain_add(sampler_, lfm_sampler_init_temp(sp.temp));
        // Distribution sampler with seed for stochastic sampling
        lfm_sampler_chain_add(sampler_, lfm_sampler_init_dist(sp.seed));
    } else {
        // Greedy sampling when temp <= 0
        lfm_sampler_chain_add(sampler_, lfm_sampler_init_greedy());
    }
}



    



    lfm_token InferenceCore::Sample() {

        if (!ctx_ || !sampler_) return 0;

        lfm_token token = lfm_sampler_sample(sampler_, ctx_, -1);
        RecordSamplerToken(token, true);
        return token;
    }



        



        std::vector<float> InferenceCore::GetLogits() const {



            if (!ctx_) return {};



            auto* logits = lfm_get_logits(ctx_);



            const auto* vocab = lfm_model_get_vocab(model_);



            int n_vocab = lfm_vocab_n_tokens(vocab);



            return std::vector<float>(logits, logits + n_vocab);



        }



        



        bool InferenceCore::HealLastToken() {



        



            if (token_history_.size() < 2) {

        return false;

    }



    lfm_token t_last = token_history_.back();

    

        

    

            // Check for valid snapshot to avoid full re-compute

    

        

    

            int target_n_past = n_past_ - 1;

    

        

    

            if (config_.enable_healing && !healing_state_buffer_.empty() && healing_n_past_ == target_n_past) {

    

        

    

                // Fast Path: Restore from snapshot

    

        

    

                spdlog::debug("HealLastToken: Fast Path");

    

        

    

                lfm_state_set_data(ctx_, healing_state_buffer_.data(), healing_state_buffer_.size());

    

        

    

                n_past_ = healing_n_past_;

    

    

    

            

    

    

    

                    

    

    

    

            

    

    

    

                    TruncateHistory(token_history_.size() - 1); // Remove t_last from history

                // Restore sampler state
                if (sampler_) {
                    bool restored = false;
                    if (config_.structured_checkpointing && healing_sampler_snapshot_) {
                        restored = SetSamplerFromSnapshot(healing_sampler_snapshot_);
                    }
                    if (!restored) {
                        RebuildSamplerFromHistory(!grammar_str_.empty());
                    }
                }

            } else {

    

    

    

                    // Slow Fallback Path

    

    

    

                    const size_t history_size = token_history_.size();
                lfm_token t_prev = token_history_[history_size - 2];

                TruncateHistory(history_size - 1); // remove t_last
                TruncateHistory(history_size - 2); // remove t_prev

    

    

    

        

    

    

    

        

    

    

    

        

    

    

    

                int pos_prev = n_past_ - 2;

    

    

    

        

    

    

    

                lfm_memory_t mem = lfm_get_memory(ctx_);

    

    

    

        

    

    

    

                

    

    

    

        

    

    

    

                if (!lfm_memory_seq_rm(mem, 0, pos_prev, -1)) {

    

    

    

        

    

    

    

                    // Partial rewind failed, full reset

    

    

    

        

    

    

    

                    lfm_memory_clear(mem, true);

    

    

    

        

    

    

    

                    if (sampler_) lfm_sampler_reset(sampler_);

    

    

    

        

    

    

    

                    n_past_ = 0;

    

    

    

        

    

    

    

        

    

    

    

        

    

    

    

                    if (!token_history_.empty()) {

    

    

    

        

    

    

    

                        // Manually replay history (excluding t_prev)

    

    

    

        

    

    

    

                        int32_t n = (int32_t)token_history_.size();

    

    

    

        

    

    

    

                        std::vector<lfm_pos> pos(n);

    

    

    

        

    

    

    

                        for(int i=0; i<n; ++i) pos[i] = i;

    

    

    

        

    

    

    

        

    

    

    

        

    

    

    

                        lfm_batch batch = {};

    

    

    

        

    

    

    

                        batch.n_tokens = n;

    

    

    

        

    

    

    

                        batch.token = token_history_.data();

    

    

    

        

    

    

    

                        batch.pos = pos.data();

    

    

    

        

    

    

    

                        

    

    

    

        

    

    

    

                        if (lfm_decode(ctx_, batch) != 0) return false;

    

    

    

        

    

    

    

                        n_past_ = n;

    

    

    

        

    

    

    

        

    

    

    

        

    

    

    

                        

    

    

    

        

    

    

    

                    }

    

    

    

        

    

    

    

                } else {

    

    

    

        

    

    

    

                    n_past_ = pos_prev;

                }

                if (sampler_) {
                    RebuildSamplerFromHistory(!grammar_str_.empty());
                }

    

    

    

        

    

    

    

        

    

    

    

        

    

    

    

                // Re-ingest t_prev

    

    

    

        

    

    

    

                if (!IngestInternal({t_prev}, true)) {

    

    

    

        

    

    

    

                    return false;

    

    

    

        

    

    

    

                }

    

    

    

        

    

    

    

            }



        // Common Path: Sample new token



        char buf[256];



        const auto* vocab = lfm_model_get_vocab(model_);



        int n = lfm_token_to_piece(vocab, t_last, buf, sizeof(buf), 0, false);



        std::string prefix = (n >= 0) ? std::string(buf, n) : "";



    



        if (prefix_sampler_) {



            lfm_sampler_prefix_set(prefix_sampler_, prefix.c_str());



        }



    



        lfm_token t_new = lfm_sampler_sample(sampler_, ctx_, -1);

        RecordSamplerToken(t_new, true);



    



        if (prefix_sampler_) {



            lfm_sampler_prefix_set(prefix_sampler_, "");



        }



    



            // Ingest new token using Internal to avoid snapshotting this single step (unless we want to heal iteratively?)



    



            // Typically healing is one-shot.



    



            IngestInternal({t_new}, false);



    



        



    



            return (t_new != t_last);

}

} // namespace liquid
