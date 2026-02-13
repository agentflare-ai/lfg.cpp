#pragma once

#include "lfg_inference.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles for the session-centric API.
typedef struct lfg_session lfg_session;
typedef struct lfg_checkpoint lfg_checkpoint;

// --- Structured Tool Call Types (OpenAI-compatible) ---

typedef enum {
    LFG_TOOL_CALL_FORMAT_PYTHONIC = 0,  // [func(key='val', key2=123)]
    LFG_TOOL_CALL_FORMAT_JSON     = 1,  // {"name":"func","arguments":{...}}
} lfg_tool_call_format;

typedef enum {
    LFG_TOOL_SCORE_OFF   = 0,  // Always inject tools (default, backward compat)
    LFG_TOOL_SCORE_AUTO  = 1,  // Skip if top score doesn't exceed mean by threshold
    LFG_TOOL_SCORE_FIXED = 2,  // Skip if top score < threshold
} lfg_tool_score_mode;

typedef struct lfg_tool_call {
    const char * id;         // "call_0", "call_1", ...
    const char * name;       // Function name
    const char * arguments;  // JSON string: '{"expression": "1 + 1"}'
} lfg_tool_call;

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
    lfg_tool_score_mode tool_score_mode;  // Tool injection gating. 0 = OFF (always inject).
    float tool_min_score;                  // Threshold value. AUTO: gap above mean. FIXED: absolute minimum.
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

// Configure text-based stop strings. The generate loop matches accumulated
// output text against these strings and stops when any suffix matches.
// Unlike token-level stop sequences, text stops are encoding-independent
// (same text always matches regardless of how the tokenizer splits it).
// Pass n_strings == 0 to clear. Returns false on invalid arguments.
LFG_API bool lfg_session_configure_stop_strings(
    lfg_session * session,
    const char * const * strings,
    int32_t n_strings);

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

// Tool function pointer for auto-execution. Takes JSON arguments string,
// returns a malloc'd result string (engine calls free()). Return NULL for error.
typedef const char * (*lfg_tool_fn)(const char *arguments, void *user_data);

// Observation callback — fired after each tool execution during auto-execution.
typedef void (*lfg_tool_call_cb)(
    const lfg_tool_call *call, const char *result, int32_t result_len,
    int32_t round, void *user_data);

typedef struct lfg_tool_desc {
    const char * name;
    const char * description;
    const char * parameters;   // JSON Schema object, nullable
    lfg_tool_fn  fn;           // NULL = consumer handles (LFG_STOP_TOOL_CALL)
    void *       fn_user_data;
} lfg_tool_desc;

// Register tools with the session. Computes & caches embeddings internally.
// top_k: number of highest-ranked tools to inject into context. 0 = disabled.
// Returns number of tools registered, or -1 on error.
LFG_API int32_t lfg_session_register_tools(lfg_session * session,
                                           const lfg_tool_desc * tools, int32_t n_tools,
                                           int32_t top_k);

// Clear registered tools and free tool context.
LFG_API void lfg_session_clear_tools(lfg_session * session);

// Rank registered tools against query text. Writes formatted tool list to buf.
// Returns bytes written (excluding NUL), or required size if buf is NULL/buf_size is 0.
// Returns -1 on error or if no tools are registered.
LFG_API int32_t lfg_session_rank_tools(lfg_session * session,
                                       const char * query, int32_t query_len,
                                       char * buf, int32_t buf_size);

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

typedef enum {
    LFG_ENTROPY_GATE_OFF   = 0,  // Disabled (no entropy events)
    LFG_ENTROPY_GATE_FIXED = 1,  // Fire when norm >= threshold (default)
    LFG_ENTROPY_GATE_AUTO  = 2,  // Fire when norm >= running_mean + threshold
} lfg_entropy_gate_mode;

typedef struct lfg_entropy_monitor_config {
    float    threshold;          // Normalized entropy threshold (0,1]. 0 = disabled.
    int32_t  cooldown_tokens;    // Min tokens between events.
    int32_t  ring_size;          // Ring buffer slots. 0 = default (4).
    lfg_entropy_gate_mode gate_mode;  // Gating mode. 0 = OFF, 1 = FIXED (default), 2 = AUTO.
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

// --- Confidence Monitor API (inverse entropy — sustained low-entropy span detection) ---

// Event emitted when a sustained low-entropy span ends.
typedef struct lfg_confidence_event {
    float       mean_entropy;    // Average normalized entropy over the span
    float       min_entropy;     // Minimum normalized entropy in the span
    int32_t     span_length;     // Number of consecutive low-entropy tokens
    int32_t     start_pos;       // n_past at span start
    int32_t     end_pos;         // n_past at span end
    int32_t     n_embd;          // Embedding dimension (for embd_out sizing)
    const char *span_text;       // Detokenized span text (valid until next pop). NULL if unavailable.
    int32_t     span_text_len;   // Length in bytes (excludes NUL terminator).
} lfg_confidence_event;

typedef enum {
    LFG_CONFIDENCE_GATE_OFF   = 0,  // Disabled (no confidence events)
    LFG_CONFIDENCE_GATE_FIXED = 1,  // Confident when norm <= threshold (default)
    LFG_CONFIDENCE_GATE_AUTO  = 2,  // Confident when norm <= running_mean - threshold
} lfg_confidence_gate_mode;

typedef struct lfg_confidence_monitor_config {
    float    threshold;          // Normalized entropy ceiling (0,1]. Tokens below this are "confident".
    int32_t  min_span;           // Minimum consecutive tokens to emit an event. 0 = default (5).
    int32_t  ring_size;          // Ring buffer slots. 0 = default (4).
    bool     include_reasoning;  // false (default) = skip reasoning tokens; true = include them.
    lfg_confidence_gate_mode gate_mode;  // Gating mode. 0 = OFF, 1 = FIXED (default), 2 = AUTO.
} lfg_confidence_monitor_config;

LFG_API lfg_confidence_monitor_config lfg_confidence_monitor_default_config(void);

// Configure. Allocates ring buffer + embedding context. Pass NULL to disable.
// Returns n_embd (> 0) on success, 0 on failure or disable.
LFG_API int32_t lfg_session_configure_confidence_monitor(
    lfg_session * session, const lfg_confidence_monitor_config * config);

// Pop next pending event. Copies embedding into embd_out. Returns false if none pending.
LFG_API bool lfg_session_confidence_pop(lfg_session * session,
                                         lfg_confidence_event * event_out,
                                         float * embd_out, int32_t embd_cap);

LFG_API int32_t lfg_session_confidence_pending(lfg_session * session);
LFG_API void    lfg_session_confidence_flush(lfg_session * session);
LFG_API volatile int32_t * lfg_session_confidence_counter(lfg_session * session);

// --- Surprise Monitor API (input novelty — aggregate surprise during ingestion) ---

// Single aggregate event produced after prompt ingestion.
// Summarizes how surprising the entire input was to the model.
typedef struct lfg_surprise_event {
    float       mean_surprise;       // Average normalized surprise across above-threshold tokens
    float       max_surprise;        // Maximum normalized surprise
    int32_t     n_above_threshold;   // Count of tokens above threshold
    int32_t     n_tokens_evaluated;  // Total tokens evaluated (prompt minus BOS)
    int32_t     n_embd;              // Embedding dimension (for embd_out sizing)
} lfg_surprise_event;

typedef enum {
    LFG_SURPRISE_GATE_OFF   = 0,  // Disabled (no surprise events)
    LFG_SURPRISE_GATE_FIXED = 1,  // Token surprising when surprise >= threshold (default)
    LFG_SURPRISE_GATE_AUTO  = 2,  // Token surprising when surprise >= prompt_mean + threshold
} lfg_surprise_gate_mode;

typedef struct lfg_surprise_monitor_config {
    float    threshold;          // Normalized surprise floor (0,1]. Above = surprising.
    int32_t  ring_size;          // Ring buffer slots. 0 = default (4).
    bool     include_reasoning;  // false (default) = skip reasoning tokens; true = include them.
    lfg_surprise_gate_mode gate_mode;  // Gating mode. 0 = OFF, 1 = FIXED (default), 2 = AUTO.
} lfg_surprise_monitor_config;

LFG_API lfg_surprise_monitor_config lfg_surprise_monitor_default_config(void);

// Configure. Allocates ring buffer + embedding context. Pass NULL to disable.
// Returns n_embd (> 0) on success, 0 on failure or disable.
LFG_API int32_t lfg_session_configure_surprise_monitor(
    lfg_session * session, const lfg_surprise_monitor_config * config);

// Pop next pending event. Copies embedding into embd_out.
// Pass NULL for embd_out to skip embedding copy. Returns false if none pending.
LFG_API bool lfg_session_surprise_pop(lfg_session * session,
                                       lfg_surprise_event * event_out,
                                       float * embd_out, int32_t embd_cap);
LFG_API int32_t lfg_session_surprise_pending(lfg_session * session);
LFG_API void    lfg_session_surprise_flush(lfg_session * session);
LFG_API volatile int32_t * lfg_session_surprise_counter(lfg_session * session);

// --- Embedding API ---

// Compute a mean-pooled, L2-normalized embedding for the given text.
// Writes n_embd floats into out. Returns n_embd on success, 0 on failure.
// Allocates an embedding context on first call (reused across calls).
LFG_API int32_t lfg_session_embed(lfg_session * session,
                                   const char * text, int32_t text_len,
                                   float * out, int32_t out_cap);

// Compute per-token, L2-normalized embeddings for the given text.
// Writes n_tok * n_embd floats into out (out_cap must be >= n_tok * n_embd).
// Returns n_tok on success, 0 on failure. Use lfg_model_n_embd_out() to get n_embd.
// Allocates a per-token embedding context on first call (reused across calls).
LFG_API int32_t lfg_session_embed_tokens(lfg_session * session,
                                          const char * text, int32_t text_len,
                                          float * out, int32_t out_cap);

// --- Generate Loop API ---

// Token callback return value: continue or stop generation.
typedef enum { LFG_GENERATE_CONTINUE = 0, LFG_GENERATE_STOP = 1 } lfg_generate_action;

// Called per generated token. Return LFG_GENERATE_CONTINUE or LFG_GENERATE_STOP.
typedef lfg_generate_action (*lfg_generate_token_cb)(
    lfg_token token, const char * piece, int32_t piece_len, void * user_data);

typedef struct lfg_generate_config {
    int32_t  max_tokens;          // Hard token limit. 0 = use session config.

    // Chat history options (only used by lfg_session_chat_generate)
    bool     include_history_reasoning;  // false (default) = strip <think>...</think>
                                         // from assistant messages in history.
                                         // Saves context for multi-turn chat.

    // Streaming callback (nullable — NULL means no callback)
    lfg_generate_token_cb      token_cb;
    void                     * token_cb_data;

    // Auto tool execution (observational callback + round limit)
    lfg_tool_call_cb           tool_call_cb;       // nullable, fired after each auto-executed tool call
    void                     * tool_call_cb_data;
    int32_t                    max_tool_rounds;     // 0 = default (5)
} lfg_generate_config;

// Why generation stopped.
typedef enum {
    LFG_STOP_EOS        = 0,  // End-of-generation token
    LFG_STOP_MAX_TOKENS = 1,  // Hit max_tokens limit
    LFG_STOP_CALLBACK   = 2,  // Token callback returned STOP
    LFG_STOP_TOOL_CALL  = 3,  // Model emitted <|tool_call_end|>
} lfg_stop_reason;

typedef struct lfg_generate_result {
    int32_t          n_tokens;            // Tokens generated
    int32_t          n_retrievals;        // Reserved (always 0). Retrieval is caller-orchestrated via entropy_pop.
    int32_t          n_confidence_spans;  // Reserved (always 0). Consume confidence events via confidence_pop.
    int32_t          n_surprise_events;   // Reserved (always 0). Consume surprise events via surprise_pop.
    int32_t          n_tool_calls;        // Number of parsed tool calls (0 if none or non-tool-call stop)
    int32_t          n_tool_rounds;       // Auto-execution rounds completed (0 if no auto-execution)
    lfg_stop_reason  stop_reason;         // Why generation stopped
} lfg_generate_result;

// Default generate config (max_tokens=0, token/tool callbacks NULL).
LFG_API lfg_generate_config lfg_generate_default_config(void);

// Generate from current session state (prompt already ingested).
// Runs decode+sample+ingest loop on C side. Returns when stopped.
LFG_API lfg_generate_result lfg_session_generate(
    lfg_session * session, lfg_generate_config config);

// Prompt-level: tokenize raw text, ingest, generate.
// One FFI call for instruction/completion-style generation.
// add_bos controls whether a BOS token is prepended during tokenization.
LFG_API lfg_generate_result lfg_session_prompt_generate(
    lfg_session * session,
    const char * prompt, int32_t prompt_len,
    bool add_bos,
    lfg_generate_config config);

// Chat-level: format messages with model's chat template, tokenize, ingest, generate.
// One FFI call for the entire chat->generation pipeline.
LFG_API lfg_generate_result lfg_session_chat_generate(
    lfg_session * session,
    const lfg_chat_message * messages, size_t n_messages,
    lfg_generate_config config);

// --- Last Formatted Prompt ---

// Returns the exact text sent to the tokenizer by the last chat_generate or
// prompt_generate call. Useful for debugging template application, tool injection,
// and thinking-strip behavior. Returns NULL if no generation has occurred.
// The pointer is valid until the next generate call or session_reset/free.
LFG_API const char * lfg_session_get_last_prompt(lfg_session * session, int32_t * len_out);

// --- Structured Tool Call Accessors ---

// Get parsed tool calls from the last generation (valid after LFG_STOP_TOOL_CALL).
// Writes count to *n_out. Returns pointer to array (owned by session, valid until next generate/reset/free).
LFG_API const lfg_tool_call * lfg_session_get_tool_calls(lfg_session * session, int32_t * n_out);

// Get the raw detokenized output from the last generation (with special tokens).
// Writes length to *len_out. Returns pointer (owned by session).
LFG_API const char * lfg_session_get_last_output(lfg_session * session, int32_t * len_out);

// Set the tool call format for parsing. Default is LFG_TOOL_CALL_FORMAT_PYTHONIC.
LFG_API void lfg_session_set_tool_call_format(lfg_session * session, lfg_tool_call_format format);

// Parse Pythonic-format tool calls from text (no session required).
// Input: text like "[func(key='val')]" — may contain multiple calls.
// Fills `out` array (up to `out_cap` entries). Each entry's name and arguments are
// malloc'd strings; caller must free them. The `id` field is set to NULL.
// Returns number of tool calls parsed.
LFG_API int32_t lfg_parse_pythonic_tool_calls(const char * text, int32_t text_len,
                                               lfg_tool_call * out, int32_t out_cap);

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
