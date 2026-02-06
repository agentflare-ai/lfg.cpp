#include "lfm_api.h"

#include "inference_core.h"
#include "json_schema_to_grammar.h"
#include "lfm_impl.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

struct lfm_session {
    lfm_model * model = nullptr;
    liquid::InferenceCore * core = nullptr;
};

struct lfm_checkpoint {
    liquid::InferenceCore::Checkpoint cp;
};

static liquid::InferenceCore::SamplingConfig ToSampling(const lfm_sampling_config & cfg) {
    liquid::InferenceCore::SamplingConfig out;
    out.seed = cfg.seed;
    out.n_prev = cfg.n_prev;
    out.top_k = cfg.top_k;
    out.top_p = cfg.top_p;
    out.min_p = cfg.min_p;
    out.typ_p = cfg.typ_p;
    out.temp = cfg.temp;
    out.penalty_last_n = cfg.penalty_last_n;
    out.penalty_repeat = cfg.penalty_repeat;
    out.penalty_freq = cfg.penalty_freq;
    out.penalty_present = cfg.penalty_present;
    out.mirostat = cfg.mirostat;
    out.mirostat_tau = cfg.mirostat_tau;
    out.mirostat_eta = cfg.mirostat_eta;
    return out;
}

LFM_API lfm_sampling_config lfm_sampling_default_config(void) {
    lfm_sampling_config cfg{};
    cfg.seed = 0xFFFFFFFF;
    cfg.n_prev = 64;
    cfg.top_k = 40;
    cfg.top_p = 0.95f;
    cfg.min_p = 0.05f;
    cfg.typ_p = 1.0f;
    cfg.temp = 0.80f;
    cfg.penalty_last_n = 64;
    cfg.penalty_repeat = 1.0f;
    cfg.penalty_freq = 0.0f;
    cfg.penalty_present = 0.0f;
    cfg.mirostat = 0;
    cfg.mirostat_tau = 5.0f;
    cfg.mirostat_eta = 0.10f;
    return cfg;
}

LFM_API lfm_session_config lfm_session_default_config(void) {
    lfm_session_config cfg{};
    cfg.n_threads = 4;
    cfg.n_ctx = 2048;
    cfg.n_batch = 512;
    cfg.enable_healing = false;
    cfg.structured_checkpointing = true;
    cfg.sampling = lfm_sampling_default_config();
    return cfg;
}

LFM_API lfm_session * lfm_session_create(lfm_model * model, const lfm_session_config * config) {
    if (!model) return nullptr;
    lfm_session_config cfg = config ? *config : lfm_session_default_config();

    liquid::InferenceCore::Config core_cfg;
    core_cfg.n_threads = cfg.n_threads;
    core_cfg.n_ctx = cfg.n_ctx;
    core_cfg.n_batch = cfg.n_batch;
    core_cfg.enable_healing = cfg.enable_healing;
    core_cfg.structured_checkpointing = cfg.structured_checkpointing;
    core_cfg.sampling = ToSampling(cfg.sampling);

    auto * session = new lfm_session();
    session->model = model;
    session->core = new liquid::InferenceCore(model, core_cfg);
    return session;
}

LFM_API void lfm_session_free(lfm_session * session) {
    if (!session) return;
    delete session->core;
    delete session;
}

LFM_API void lfm_session_reset(lfm_session * session) {
    if (!session || !session->core) return;
    session->core->Reset();
}

LFM_API bool lfm_session_configure_structured(lfm_session * session,
                                                    const char * grammar_or_schema,
                                                    const char * root_rule) {
    if (!session || !session->core || !grammar_or_schema) return false;
    session->core->ConfigureStructuredDecoding(grammar_or_schema, root_rule ? root_rule : "root");
    return true;
}

LFM_API int32_t lfm_json_schema_to_grammar(const char * json_schema,
                                                 bool force_gbnf,
                                                 char * buf,
                                                 size_t buf_size) {
    if (!json_schema) {
        lfm_set_last_error(LFM_ERROR_INVALID_ARGUMENT, "%s: json_schema is NULL", __func__);
        return -1;
    }
    if (!buf && buf_size > 0) {
        lfm_set_last_error(LFM_ERROR_INVALID_ARGUMENT, "%s: buf is NULL", __func__);
        return -1;
    }

    try {
        const auto schema = nlohmann::ordered_json::parse(json_schema);
        const std::string grammar = json_schema_to_grammar(schema, force_gbnf);
        if (!buf || buf_size == 0) {
            return static_cast<int32_t>(grammar.size());
        }
        return std::snprintf(buf, buf_size, "%s", grammar.c_str());
    } catch (const std::exception & err) {
        lfm_set_last_error(LFM_ERROR_INVALID_ARGUMENT,
                              "%s: failed to convert JSON schema: %s",
                              __func__,
                              err.what());
        if (buf && buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
}

LFM_API bool lfm_session_ingest_tokens(lfm_session * session,
                                             const lfm_token * tokens,
                                             size_t n_tokens,
                                             bool update_sampler) {
    if (!session || !session->core || !tokens) return false;
    std::vector<lfm_token> vec(tokens, tokens + n_tokens);
    return session->core->IngestTokens(vec, update_sampler);
}

LFM_API bool lfm_session_decode(lfm_session * session) {
    if (!session || !session->core) return false;
    return session->core->Decode();
}

LFM_API lfm_token lfm_session_sample(lfm_session * session) {
    if (!session || !session->core) return 0;
    return session->core->Sample();
}

LFM_API bool lfm_session_heal_last_token(lfm_session * session) {
    if (!session || !session->core) return false;
    return session->core->HealLastToken();
}

LFM_API int32_t lfm_session_get_logits(lfm_session * session, float * out, int32_t max_out) {
    if (!session || !session->core) return -1;
    auto logits = session->core->GetLogits();
    if (!out || max_out <= 0) {
        return static_cast<int32_t>(logits.size());
    }
    const int32_t n = std::min<int32_t>(max_out, static_cast<int32_t>(logits.size()));
    if (n > 0) {
        std::memcpy(out, logits.data(), sizeof(float) * n);
    }
    return n;
}

LFM_API int32_t lfm_session_get_vocab_size(lfm_session * session) {
    if (!session || !session->model) return -1;
    const auto * vocab = lfm_model_get_vocab(session->model);
    if (!vocab) return -1;
    return lfm_vocab_n_tokens(vocab);
}

LFM_API lfm_checkpoint * lfm_session_create_checkpoint(lfm_session * session) {
    if (!session || !session->core) return nullptr;
    auto * ck = new lfm_checkpoint();
    ck->cp = session->core->CreateCheckpoint();
    return ck;
}

LFM_API lfm_checkpoint_restore_options lfm_checkpoint_restore_default_options(void) {
    lfm_checkpoint_restore_options opts{};
    opts.restore_sampler_state = true;
    opts.restore_grammar = true;
    return opts;
}

LFM_API bool lfm_session_restore_checkpoint_ex(lfm_session * session,
                                                     const lfm_checkpoint * checkpoint,
                                                     const lfm_checkpoint_restore_options * options) {
    if (!session || !session->core || !checkpoint) return false;
    liquid::InferenceCore::RestoreOptions opts{};
    if (options) {
        opts.restore_sampler_state = options->restore_sampler_state;
        opts.restore_grammar = options->restore_grammar;
    }
    return session->core->RestoreCheckpoint(checkpoint->cp, opts);
}

LFM_API bool lfm_session_restore_checkpoint(lfm_session * session, const lfm_checkpoint * checkpoint) {
    return lfm_session_restore_checkpoint_ex(session, checkpoint, nullptr);
}

LFM_API void lfm_checkpoint_free(lfm_checkpoint * checkpoint) {
    delete checkpoint;
}
