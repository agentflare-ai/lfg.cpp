#pragma once

#include "lfg_inference.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles for the session-centric API.
typedef struct lfg_session lfg_session;
typedef struct lfg_checkpoint lfg_checkpoint;

// Sampling parameters for a session.
typedef struct lfg_sampling_config {
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
} lfg_sampling_config;

// Session configuration. Owns decoding + sampling behavior.
typedef struct lfg_session_config {
    int n_threads;
    int n_ctx;
    int n_batch;
    bool enable_healing;
    bool structured_checkpointing; // Snapshot sampler state for structured decoding.
    int reasoning_budget;          // 0 = disabled. Number of tokens allowed for reasoning.
    int32_t max_tokens;            // 0 = unlimited. Max tokens to generate per reset cycle.
    lfg_sampling_config sampling;
} lfg_session_config;

// Configuration helpers.
LFG_API lfg_sampling_config lfg_sampling_default_config(void);
LFG_API lfg_session_config lfg_session_default_config(void);

// Session lifecycle.
LFG_API lfg_session * lfg_session_create(struct lfg_model * model, const lfg_session_config * config);
LFG_API void lfg_session_free(lfg_session * session);
LFG_API void lfg_session_reset(lfg_session * session);

// Structured decoding. If grammar_or_schema begins with '{', it is treated as JSON schema.
LFG_API bool lfg_session_configure_structured(lfg_session * session,
                                                    const char * grammar_or_schema,
                                                    const char * root_rule);

// Configure tokens that delimit a reasoning/thinking block.
// Constraints are suspended while inside these blocks.
LFG_API void lfg_session_configure_reasoning(lfg_session * session,
                                                   const lfg_token * start_tokens, size_t n_start,
                                                   const lfg_token * end_tokens,   size_t n_end);

// Convert a JSON schema (as a string) to a grammar. Returns number of bytes written
// (excluding the null terminator), or the required size if buf is null or buf_size is 0.
// Returns -1 on error; use lfg_get_last_error for details.
LFG_API int32_t lfg_json_schema_to_grammar(const char * json_schema,
                                                 bool force_gbnf,
                                                 char * buf,
                                                 size_t buf_size);

// Configure stop sequences. Generation returns EOS when any sequence matches.
// Each sequence is an array of tokens; pass arrays of pointers and lengths.
// Pass n_sequences == 0 to clear. Returns false on invalid arguments.
LFG_API bool lfg_session_configure_stop_sequences(
    lfg_session * session,
    const lfg_token * const * sequences,
    const size_t * sequence_lengths,
    size_t n_sequences);

// Token ingestion / decoding.
LFG_API bool lfg_session_ingest_tokens(lfg_session * session,
                                             const lfg_token * tokens,
                                             size_t n_tokens,
                                             bool update_sampler);
LFG_API bool lfg_session_decode(lfg_session * session);
LFG_API lfg_token lfg_session_sample(lfg_session * session);
LFG_API bool lfg_session_heal_last_token(lfg_session * session);

// Logits access. Returns number of logits copied or required size when out == nullptr.
LFG_API int32_t lfg_session_get_logits(lfg_session * session, float * out, int32_t max_out);
LFG_API int32_t lfg_session_get_vocab_size(lfg_session * session);

// Checkpointing.
LFG_API lfg_checkpoint * lfg_session_create_checkpoint(lfg_session * session);
typedef struct lfg_checkpoint_restore_options {
    bool restore_sampler_state;
    bool restore_grammar;
} lfg_checkpoint_restore_options;
// Default restore options: restore sampler state + grammar.
LFG_API lfg_checkpoint_restore_options lfg_checkpoint_restore_default_options(void);
LFG_API bool lfg_session_restore_checkpoint_ex(lfg_session * session,
                                                     const lfg_checkpoint * checkpoint,
                                                     const lfg_checkpoint_restore_options * options);
LFG_API bool lfg_session_restore_checkpoint(lfg_session * session, const lfg_checkpoint * checkpoint);
LFG_API void lfg_checkpoint_free(lfg_checkpoint * checkpoint);

// --- Tool Ranking API ---

typedef struct lfg_tool_desc {
    const char * name;
    const char * description;
    const char * json_schema;  // nullable
} lfg_tool_desc;

// Register tools with the session. Computes & caches embeddings internally.
// top_k: number of highest-ranked tools to inject into context. 0 = disabled.
// Returns number of tools registered, or -1 on error.
LFG_API int32_t lfg_session_register_tools(lfg_session * session,
                                           const lfg_tool_desc * tools, int32_t n_tools,
                                           int32_t top_k);

// Clear registered tools and free tool context.
LFG_API void lfg_session_clear_tools(lfg_session * session);

// --- Entropy Monitor API ---

// Event read from the ring buffer via lfg_session_entropy_pop().
typedef struct lfg_entropy_event {
    float       entropy;        // Raw Shannon entropy: H = -sum p_i log(p_i)
    float       normalized;     // entropy / log(n_vocab), range [0,1]
    float       top_logprob;    // Log probability of the sampled token
    lfg_token   token;          // The sampled token
    int32_t     n_past;         // Token position when event fired
    int32_t     checkpoint_id;  // Opaque ID for lfg_session_rewind()
    int32_t     n_embd;         // Embedding dimension (for embd_out sizing)
} lfg_entropy_event;

typedef struct lfg_entropy_monitor_config {
    float    threshold;          // Normalized entropy threshold (0,1]. 0 = disabled.
    int32_t  cooldown_tokens;    // Min tokens between events.
    int32_t  ring_size;          // Ring buffer slots. 0 = default (4).
} lfg_entropy_monitor_config;

LFG_API lfg_entropy_monitor_config lfg_entropy_monitor_default_config(void);

// Configure. Allocates ring buffer + embedding context. Pass NULL to disable.
// Returns n_embd (> 0) on success — use this to size your entropy_pop embedding buffer.
// Returns 0 on failure or when passing NULL (disable).
LFG_API int32_t lfg_session_configure_entropy_monitor(
    lfg_session * session, const lfg_entropy_monitor_config * config);

// Pop next pending event. Copies embedding into embd_out (must be >= n_embd floats).
// Pass NULL for embd_out to skip embedding copy. Returns false if no events pending.
LFG_API bool lfg_session_entropy_pop(lfg_session * session,
                                      lfg_entropy_event * event_out,
                                      float * embd_out, int32_t embd_cap);

// Number of pending (unread) entropy events.
LFG_API int32_t lfg_session_entropy_pending(lfg_session * session);

// Discard all pending entropy events without reading them. O(1).
LFG_API void lfg_session_entropy_flush(lfg_session * session);

// Pointer to atomic write counter. Observer can poll or use platform wait.
// Incremented each time an event is written to the ring.
LFG_API volatile int32_t * lfg_session_entropy_counter(lfg_session * session);

// Rewind to entropy checkpoint. Truncates KV cache, resets sampler. Zero-alloc.
LFG_API bool lfg_session_rewind(lfg_session * session, int32_t checkpoint_id);

// Normalized entropy from last sample(). -1 if no sample performed.
LFG_API float lfg_session_get_last_entropy(lfg_session * session);

// --- Embedding API ---

// Compute a mean-pooled, L2-normalized embedding for the given text.
// Writes n_embd floats into out. Returns n_embd on success, 0 on failure.
// Allocates an embedding context on first call (reused across calls).
LFG_API int32_t lfg_session_embed(lfg_session * session,
                                   const char * text, int32_t text_len,
                                   float * out, int32_t out_cap);

// --- Generate Loop API ---

// Token callback return value: continue or stop generation.
typedef enum { LFG_GENERATE_CONTINUE = 0, LFG_GENERATE_STOP = 1 } lfg_generate_action;

// Called per generated token. Return LFG_GENERATE_CONTINUE or LFG_GENERATE_STOP.
typedef lfg_generate_action (*lfg_generate_token_cb)(
    lfg_token token, const char * piece, int32_t piece_len, void * user_data);

// Called when entropy exceeds threshold. Receives event + embedding for KB matching.
// embedding is n_embd floats (event->n_embd), or NULL if embed failed.
// Return a C string to inject (generate loop handles rewind + tokenize + ingest),
// or NULL to skip this event and keep generating. String must be valid until callback returns.
typedef const char * (*lfg_generate_entropy_cb)(
    const lfg_entropy_event * event, const float * embedding, void * user_data);

typedef struct lfg_generate_config {
    int32_t  max_tokens;          // Hard token limit. 0 = use session config.

    // Callbacks (nullable — NULL means no callback)
    lfg_generate_token_cb    token_cb;
    void                   * token_cb_data;
    lfg_generate_entropy_cb  entropy_cb;
    void                   * entropy_cb_data;
} lfg_generate_config;

// Why generation stopped.
typedef enum {
    LFG_STOP_EOS        = 0,  // End-of-generation token
    LFG_STOP_MAX_TOKENS = 1,  // Hit max_tokens limit
    LFG_STOP_CALLBACK   = 2,  // Token callback returned STOP
} lfg_stop_reason;

typedef struct lfg_generate_result {
    int32_t          n_tokens;       // Tokens generated
    int32_t          n_retrievals;   // Number of entropy-triggered rewind+inject cycles
    lfg_stop_reason  stop_reason;    // Why generation stopped
} lfg_generate_result;

// Default generate config (max_tokens=0, all callbacks NULL).
LFG_API lfg_generate_config lfg_generate_default_config(void);

// Generate from current session state (prompt already ingested).
// Runs decode+sample+ingest loop on C side. Returns when stopped.
LFG_API lfg_generate_result lfg_session_generate(
    lfg_session * session, const lfg_generate_config * config);

// Prompt-level: tokenize raw text, ingest, generate.
// One FFI call for instruction/completion-style generation.
// add_bos controls whether a BOS token is prepended during tokenization.
LFG_API lfg_generate_result lfg_session_prompt_generate(
    lfg_session * session,
    const char * prompt, int32_t prompt_len,
    bool add_bos,
    const lfg_generate_config * config);

// Chat-level: format messages with model's chat template, tokenize, ingest, generate.
// One FFI call for the entire chat->generation pipeline.
LFG_API lfg_generate_result lfg_session_chat_generate(
    lfg_session * session,
    const lfg_chat_message * messages, size_t n_messages,
    const lfg_generate_config * config);

// --- Model Loader C API (replaces liquid::ModelLoader) ---

typedef struct lfg_model_load_config {
    const char * model_path;
    bool use_mmap;
    bool use_mlock;
    int  n_gpu_layers;
} lfg_model_load_config;

typedef struct lfg_model_stats {
    uint64_t n_params;
    uint64_t size_bytes;
    int32_t  n_vocab;
    int32_t  n_ctx_train;
} lfg_model_stats;

LFG_API lfg_model_load_config lfg_model_load_default_config(void);
LFG_API struct lfg_model * lfg_load_model(const lfg_model_load_config * config);
LFG_API lfg_model_stats lfg_model_get_stats(const struct lfg_model * model);
LFG_API int32_t lfg_model_get_metadata_str(const struct lfg_model * model,
                                                 const char * key,
                                                 char * buf, size_t buf_size);

#ifdef __cplusplus
} // extern "C"
#endif
