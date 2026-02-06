#pragma once

#include "lfm_inference.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles for the new session-centric API.
typedef struct lfm_session lfm_session;
typedef struct lfm_checkpoint lfm_checkpoint;

// Sampling parameters for a session. Mirrors InferenceCore::SamplingConfig.
typedef struct lfm_sampling_config {
    uint32_t seed;
    int32_t  n_prev;
    int32_t  top_k;
    float    top_p;
    float    min_p;
    float    typ_p;
    float    temp;
    int32_t  penalty_last_n;
    float    penalty_repeat;
    float    penalty_freq;
    float    penalty_present;
    int32_t  mirostat;
    float    mirostat_tau;
    float    mirostat_eta;
} lfm_sampling_config;

// Session configuration. Owns decoding + sampling behavior.
typedef struct lfm_session_config {
    int n_threads;
    int n_ctx;
    int n_batch;
    bool enable_healing;
    bool structured_checkpointing; // Snapshot sampler state for structured decoding.
    lfm_sampling_config sampling;
} lfm_session_config;

// Configuration helpers.
LFM_API lfm_sampling_config lfm_sampling_default_config(void);
LFM_API lfm_session_config lfm_session_default_config(void);

// Session lifecycle.
LFM_API lfm_session * lfm_session_create(lfm_model * model, const lfm_session_config * config);
LFM_API void lfm_session_free(lfm_session * session);
LFM_API void lfm_session_reset(lfm_session * session);

// Structured decoding. If grammar_or_schema begins with '{', it is treated as JSON schema.
LFM_API bool lfm_session_configure_structured(lfm_session * session,
                                                    const char * grammar_or_schema,
                                                    const char * root_rule);

// Convert a JSON schema (as a string) to a grammar. Returns number of bytes written
// (excluding the null terminator), or the required size if buf is null or buf_size is 0.
// Returns -1 on error; use lfm_get_last_error for details.
LFM_API int32_t lfm_json_schema_to_grammar(const char * json_schema,
                                                 bool force_gbnf,
                                                 char * buf,
                                                 size_t buf_size);

// Token ingestion / decoding.
LFM_API bool lfm_session_ingest_tokens(lfm_session * session,
                                             const lfm_token * tokens,
                                             size_t n_tokens,
                                             bool update_sampler);
LFM_API bool lfm_session_decode(lfm_session * session);
LFM_API lfm_token lfm_session_sample(lfm_session * session);
LFM_API bool lfm_session_heal_last_token(lfm_session * session);

// Logits access. Returns number of logits copied or required size when out == nullptr.
LFM_API int32_t lfm_session_get_logits(lfm_session * session, float * out, int32_t max_out);
LFM_API int32_t lfm_session_get_vocab_size(lfm_session * session);

// Checkpointing.
LFM_API lfm_checkpoint * lfm_session_create_checkpoint(lfm_session * session);
typedef struct lfm_checkpoint_restore_options {
    bool restore_sampler_state;
    bool restore_grammar;
} lfm_checkpoint_restore_options;
// Default restore options: restore sampler state + grammar.
LFM_API lfm_checkpoint_restore_options lfm_checkpoint_restore_default_options(void);
LFM_API bool lfm_session_restore_checkpoint_ex(lfm_session * session,
                                                     const lfm_checkpoint * checkpoint,
                                                     const lfm_checkpoint_restore_options * options);
LFM_API bool lfm_session_restore_checkpoint(lfm_session * session, const lfm_checkpoint * checkpoint);
LFM_API void lfm_checkpoint_free(lfm_checkpoint * checkpoint);

#ifdef __cplusplus
} // extern "C"
#endif
