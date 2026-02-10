#pragma once

#include "lfg_api.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace liquid {

struct SamplingConfig {
    lfg_sampling_config raw = lfg_sampling_default_config();
};

struct SessionConfig {
    lfg_session_config raw = lfg_session_default_config();
};

class Checkpoint {
public:
    Checkpoint() = default;
    explicit Checkpoint(lfg_checkpoint * handle) : handle_(handle) {}
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

    lfg_checkpoint * get() const { return handle_; }
    void reset() {
        if (handle_) {
            lfg_checkpoint_free(handle_);
            handle_ = nullptr;
        }
    }

private:
    lfg_checkpoint * handle_ = nullptr;
};

class Session {
public:
    Session(lfg_model * model, const SessionConfig & config = {}) {
        handle_ = lfg_session_create(model, &config.raw);
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

    void Reset() { lfg_session_reset(handle_); }

    void ConfigureStructured(const char * grammar_or_schema, const char * root_rule = "root") {
        if (!lfg_session_configure_structured(handle_, grammar_or_schema, root_rule)) {
            throw std::runtime_error("Failed to configure structured decoding");
        }
    }

    void IngestTokens(const std::vector<lfg_token> & tokens, bool update_sampler = true) {
        if (!tokens.empty() && !lfg_session_ingest_tokens(handle_, tokens.data(), tokens.size(), update_sampler)) {
            throw std::runtime_error("Failed to ingest tokens");
        }
    }

    bool Decode() { return lfg_session_decode(handle_); }
    lfg_token Sample() { return lfg_session_sample(handle_); }
    bool HealLastToken() { return lfg_session_heal_last_token(handle_); }

    void ConfigureReasoning(const std::vector<lfg_token> & start_tokens,
                            const std::vector<lfg_token> & end_tokens) {
        lfg_session_configure_reasoning(handle_,
            start_tokens.data(), start_tokens.size(),
            end_tokens.data(), end_tokens.size());
    }

    Checkpoint CreateCheckpoint() { return Checkpoint(lfg_session_create_checkpoint(handle_)); }
    struct RestoreOptions {
        bool restore_sampler_state = true;
        bool restore_grammar = true;
    };
    bool RestoreCheckpoint(const Checkpoint & ck, const RestoreOptions & opts = {}) {
        lfg_checkpoint_restore_options raw = lfg_checkpoint_restore_default_options();
        raw.restore_sampler_state = opts.restore_sampler_state;
        raw.restore_grammar = opts.restore_grammar;
        return lfg_session_restore_checkpoint_ex(handle_, ck.get(), &raw);
    }

    std::vector<float> GetLogits() {
        const int32_t required = lfg_session_get_logits(handle_, nullptr, 0);
        if (required <= 0) return {};
        std::vector<float> logits(static_cast<size_t>(required));
        lfg_session_get_logits(handle_, logits.data(), required);
        return logits;
    }

    int32_t VocabSize() { return lfg_session_get_vocab_size(handle_); }

    // Structured tool call accessors
    std::pair<const lfg_tool_call *, int32_t> GetToolCalls() {
        int32_t n = 0;
        const lfg_tool_call *calls = lfg_session_get_tool_calls(handle_, &n);
        return {calls, n};
    }

    std::pair<const char *, int32_t> GetLastOutput() {
        int32_t len = 0;
        const char *out = lfg_session_get_last_output(handle_, &len);
        return {out, len};
    }

    void SetToolCallFormat(lfg_tool_call_format format) {
        lfg_session_set_tool_call_format(handle_, format);
    }

    lfg_session * get() const { return handle_; }

private:
    void reset() {
        if (handle_) {
            lfg_session_free(handle_);
            handle_ = nullptr;
        }
    }

    lfg_session * handle_ = nullptr;
};

} // namespace liquid
