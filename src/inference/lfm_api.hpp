#pragma once

#include "lfm_api.h"

#include <stdexcept>
#include <vector>

namespace liquid {

struct SamplingConfig {
    lfm_sampling_config raw = lfm_sampling_default_config();
};

struct SessionConfig {
    lfm_session_config raw = lfm_session_default_config();
};

class Checkpoint {
public:
    Checkpoint() = default;
    explicit Checkpoint(lfm_checkpoint * handle) : handle_(handle) {}
    ~Checkpoint() { reset(); }

    Checkpoint(const Checkpoint &) = delete;
    Checkpoint & operator=(const Checkpoint &) = delete;

    Checkpoint(Checkpoint && other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    Checkpoint & operator=(Checkpoint && other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    lfm_checkpoint * get() const { return handle_; }
    void reset() {
        if (handle_) {
            lfm_checkpoint_free(handle_);
            handle_ = nullptr;
        }
    }

private:
    lfm_checkpoint * handle_ = nullptr;
};

class Session {
public:
    Session(lfm_model * model, const SessionConfig & config = {}) {
        handle_ = lfm_session_create(model, &config.raw);
        if (!handle_) {
            throw std::runtime_error("Failed to create liquid session");
        }
    }

    ~Session() { reset(); }

    Session(const Session &) = delete;
    Session & operator=(const Session &) = delete;

    Session(Session && other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    Session & operator=(Session && other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    void Reset() { lfm_session_reset(handle_); }

    void ConfigureStructured(const char * grammar_or_schema, const char * root_rule = "root") {
        if (!lfm_session_configure_structured(handle_, grammar_or_schema, root_rule)) {
            throw std::runtime_error("Failed to configure structured decoding");
        }
    }

    void IngestTokens(const std::vector<lfm_token> & tokens, bool update_sampler = true) {
        if (!tokens.empty() && !lfm_session_ingest_tokens(handle_, tokens.data(), tokens.size(), update_sampler)) {
            throw std::runtime_error("Failed to ingest tokens");
        }
    }

    bool Decode() { return lfm_session_decode(handle_); }
    lfm_token Sample() { return lfm_session_sample(handle_); }
    bool HealLastToken() { return lfm_session_heal_last_token(handle_); }

    Checkpoint CreateCheckpoint() { return Checkpoint(lfm_session_create_checkpoint(handle_)); }
    struct RestoreOptions {
        bool restore_sampler_state = true;
        bool restore_grammar = true;
    };
    bool RestoreCheckpoint(const Checkpoint & ck, const RestoreOptions & opts = {}) {
        lfm_checkpoint_restore_options raw = lfm_checkpoint_restore_default_options();
        raw.restore_sampler_state = opts.restore_sampler_state;
        raw.restore_grammar = opts.restore_grammar;
        return lfm_session_restore_checkpoint_ex(handle_, ck.get(), &raw);
    }

    std::vector<float> GetLogits() {
        const int32_t required = lfm_session_get_logits(handle_, nullptr, 0);
        if (required <= 0) return {};
        std::vector<float> logits(static_cast<size_t>(required));
        lfm_session_get_logits(handle_, logits.data(), required);
        return logits;
    }

    int32_t VocabSize() { return lfm_session_get_vocab_size(handle_); }

    lfm_session * get() const { return handle_; }

private:
    void reset() {
        if (handle_) {
            lfm_session_free(handle_);
            handle_ = nullptr;
        }
    }

    lfm_session * handle_ = nullptr;
};

} // namespace liquid
