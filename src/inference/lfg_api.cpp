#include "lfg_api.h"

#include "json_schema_to_grammar.h"
#include "lfg_impl.h"
#include "peg-parser.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <spdlog/spdlog.h>

// ---------------------------------------------------------------------------
// Growable buffer helpers (power-of-2 doubling, pre-allocated at session create)
// ---------------------------------------------------------------------------

typedef struct { lfg_token *data; size_t size, capacity; } lfg_buf_tokens;
typedef struct { uint8_t   *data; size_t size, capacity; } lfg_buf_u8;
typedef struct { size_t    *data; size_t size, capacity; } lfg_buf_size;

static void lfg_buf_tokens_init(lfg_buf_tokens *b, size_t cap) {
    b->capacity = cap ? cap : 64;
    b->data = (lfg_token *)malloc(b->capacity * sizeof(lfg_token));
    b->size = 0;
}
static void lfg_buf_tokens_ensure(lfg_buf_tokens *b, size_t needed) {
    if (needed <= b->capacity) return;
    size_t cap = b->capacity ? b->capacity : 64;
    while (cap < needed) cap *= 2;
    b->data = (lfg_token *)realloc(b->data, cap * sizeof(lfg_token));
    b->capacity = cap;
}
static void lfg_buf_tokens_push(lfg_buf_tokens *b, lfg_token v) {
    lfg_buf_tokens_ensure(b, b->size + 1);
    b->data[b->size++] = v;
}
static void lfg_buf_tokens_free(lfg_buf_tokens *b) {
    free(b->data); b->data = nullptr; b->size = b->capacity = 0;
}

static void lfg_buf_u8_init(lfg_buf_u8 *b, size_t cap) {
    b->capacity = cap ? cap : 64;
    b->data = (uint8_t *)malloc(b->capacity);
    b->size = 0;
}
static void lfg_buf_u8_ensure(lfg_buf_u8 *b, size_t needed) {
    if (needed <= b->capacity) return;
    size_t cap = b->capacity ? b->capacity : 64;
    while (cap < needed) cap *= 2;
    b->data = (uint8_t *)realloc(b->data, cap);
    b->capacity = cap;
}
static void lfg_buf_u8_free(lfg_buf_u8 *b) {
    free(b->data); b->data = nullptr; b->size = b->capacity = 0;
}

static void lfg_buf_size_init(lfg_buf_size *b, size_t cap) {
    b->capacity = cap ? cap : 64;
    b->data = (size_t *)malloc(b->capacity * sizeof(size_t));
    b->size = 0;
}
static void lfg_buf_size_ensure(lfg_buf_size *b, size_t needed) {
    if (needed <= b->capacity) return;
    size_t cap = b->capacity ? b->capacity : 64;
    while (cap < needed) cap *= 2;
    b->data = (size_t *)realloc(b->data, cap * sizeof(size_t));
    b->capacity = cap;
}
static void lfg_buf_size_push(lfg_buf_size *b, size_t v) {
    lfg_buf_size_ensure(b, b->size + 1);
    b->data[b->size++] = v;
}
static void lfg_buf_size_free(lfg_buf_size *b) {
    free(b->data); b->data = nullptr; b->size = b->capacity = 0;
}

// ---------------------------------------------------------------------------
// Internal entropy monitor types (pre-allocated at configure time)
// ---------------------------------------------------------------------------

struct lfg_entropy_ring_slot {
    lfg_entropy_event event;
    float            *embedding;  // Points into entropy_embd_pool
};

struct lfg_entropy_snap {
    int32_t  n_past;
    size_t   token_count;
    size_t   sampler_count;
    int32_t  generated_count;
    size_t   reasoning_token_count;
    int32_t  id;
    bool     valid;
};

// ---------------------------------------------------------------------------
// Internal confidence monitor types (pre-allocated at configure time)
// ---------------------------------------------------------------------------

struct lfg_confidence_ring_slot {
    lfg_confidence_event event;
    float               *embedding;  // Points into confidence_embd_pool
};

// ---------------------------------------------------------------------------
// Internal surprise monitor types (pre-allocated at configure time)
// ---------------------------------------------------------------------------

struct lfg_surprise_ring_slot {
    lfg_surprise_event event;
    float             *embedding;  // Points into surprise_embd_pool
};

// ---------------------------------------------------------------------------
// Opaque session struct (flat C, pre-allocated buffers)
// ---------------------------------------------------------------------------

struct lfg_session {
    lfg_model   *model;
    lfg_context *ctx;
    lfg_session_config config;

    int n_past;

    // History (pre-allocated to n_ctx)
    lfg_buf_tokens token_history;
    lfg_buf_tokens sampler_history;
    lfg_buf_size   sampler_offsets;
    size_t         pending_sampler_accepts;
    bool           sampler_recording_enabled;

    // Reasoning tokens (allocated at configure time)
    lfg_token *reasoning_start_tokens;  size_t reasoning_start_count;
    lfg_token *reasoning_end_tokens;    size_t reasoning_end_count;

    // Sampler (owned)
    lfg_sampler *sampler;
    lfg_sampler *prefix_sampler;  // non-owning pointer into the chain

    // Grammar (strdup'd at configure time)
    char *grammar_str;
    char *grammar_root;

    // Healing
    lfg_buf_u8   healing_state_buffer;
    int          healing_n_past;
    lfg_sampler *healing_sampler_snapshot;  // owned, NULL if none

    // Generation counting (for max_tokens)
    int32_t generated_count;

    // Stop sequences (flat token-level storage)
    lfg_token *stop_flat;          // all sequences concatenated
    size_t    *stop_offsets;       // start offset of each sequence
    size_t    *stop_lengths;       // length of each sequence
    size_t     stop_count;         // number of sequences
    size_t     stop_max_len;       // max sequence length (for lookback)
    int32_t    last_stop_len;      // length of last matched stop seq (set by sample)

    // Text-level stop strings (encoding-independent matching)
    char     **stop_texts;         // malloc'd array of strdup'd strings
    int32_t   *stop_text_lens;     // length of each string
    int32_t    stop_text_count;    // number of strings
    int32_t    stop_text_max_len;  // max string length (for buffer sizing)

    // Reasoning state tracking (mirrors old InferenceCore fields)
    bool   in_reasoning;
    size_t reasoning_token_count;

    // Scratch (pre-allocated to n_batch)
    lfg_pos *pos_buf;      size_t pos_buf_cap;
    int8_t  *logits_buf;   size_t logits_buf_cap;

    // Tool ranking state (all buffers pre-allocated at register time)
    struct lfg_tool_entry {
        char    *name;                 // strdup'd
        char    *json_text;            // malloc'd pre-formatted JSON
        int32_t  json_text_len;        // strlen of json_text
        int32_t  token_cost;           // token count of json_text
        float   *embedding;           // malloc'd, L2-normalized, size = n_embd
        lfg_tool_fn  fn;              // auto-execution callback (NULL = consumer handles)
        void        *fn_user_data;
    };

    lfg_context     *tool_ctx;         // Separate context for computing tool embeddings (POOLING_TYPE_MEAN)
    lfg_context     *embed_none_ctx;   // Lazily created with POOLING_TYPE_NONE for per-token embeddings
    lfg_tool_entry  *tool_entries;     // malloc'd array, size = tool_count
    int32_t          tool_count;
    int32_t          tool_n_embd;      // cached n_embd for dot products

    // Embedding cache (parallel arrays, sized to tool_count)
    uint64_t        *tool_cache_hashes;   // malloc'd
    float           *tool_cache_embeds;   // malloc'd, flat [cache_count * n_embd]
    int32_t          tool_cache_count;
    int32_t          tool_cache_cap;

    // Decode-time scratch (pre-allocated at register time, zero allocs in decode)
    float           *tool_query_embd;     // malloc'd, size = n_embd
    int32_t         *tool_score_indices;  // malloc'd, size = tool_count (sorted by score)
    float           *tool_scores;         // malloc'd, size = tool_count
    char            *tool_json_buf;        // malloc'd, pre-computed full JSON block
    int32_t          tool_json_buf_cap;
    lfg_token       *tool_token_buf;      // malloc'd, for tokenized tool block
    int32_t          tool_token_buf_cap;

    int32_t              tool_top_k;            // 0 = disabled
    lfg_tool_score_mode  tool_score_mode;
    float                tool_min_score;
    bool             tools_injected;       // Reset on session_reset()

    // Entropy monitor config
    float                entropy_threshold;
    int32_t              entropy_cooldown;
    int32_t              entropy_tokens_since;
    float                entropy_last;
    float                entropy_last_norm;
    bool                 entropy_active;
    lfg_entropy_gate_mode entropy_gate_mode;
    float                entropy_running_sum;
    int32_t              entropy_running_count;

    // SPSC ring buffer (pre-allocated at configure time)
    lfg_entropy_ring_slot *entropy_slots;
    lfg_entropy_snap      *entropy_snaps;
    float                 *entropy_embd_pool;
    int32_t                entropy_ring_cap;
    volatile int32_t       entropy_write_idx;
    int32_t                entropy_read_idx;
    int32_t                entropy_next_id;
    int32_t                entropy_n_embd;

    // Confidence monitor config (inverse entropy — sustained low-entropy span detection)
    float                confidence_threshold;
    int32_t              confidence_min_span;
    bool                 confidence_active;
    lfg_confidence_gate_mode confidence_gate_mode;
    float                confidence_running_sum;
    int32_t              confidence_running_count;

    // Run tracker (zero-alloc hot path state)
    int32_t              confidence_run_count;
    float                confidence_run_entropy_sum;
    float                confidence_run_min_entropy;
    int32_t              confidence_run_start_pos;

    // SPSC ring buffer (pre-allocated at configure time)
    lfg_confidence_ring_slot *confidence_slots;
    float                    *confidence_embd_pool;
    int32_t                   confidence_ring_cap;
    volatile int32_t          confidence_write_idx;
    int32_t                   confidence_read_idx;
    int32_t                   confidence_n_embd;

    // Confidence reasoning gating
    bool                 confidence_include_reasoning;

    // Confidence span text buffer (realloc'd, valid until next pop)
    char                *confidence_text_buf;
    int32_t              confidence_text_cap;

    // Surprise monitor (input novelty — single aggregate event per ingestion)
    float                surprise_threshold;
    bool                 surprise_active;
    bool                 surprise_include_reasoning;
    int32_t              surprise_skip_tokens;  // tokens to skip at start of next ingestion (chat context)
    lfg_surprise_gate_mode surprise_gate_mode;
    float               *surprise_per_token;     // malloc'd scratch for AUTO two-pass
    int32_t              surprise_per_token_cap;  // capacity of above

    // Accumulator (filled during ingestion, read via pop)
    int32_t              surprise_count;        // tokens above threshold
    float                surprise_sum;          // sum of surprises (for mean)
    float                surprise_max;          // max surprise
    int32_t              surprise_n_evaluated;  // total tokens evaluated

    // SPSC ring buffer (pre-allocated at configure time)
    lfg_surprise_ring_slot *surprise_slots;
    float                  *surprise_embd_pool;
    int32_t                 surprise_ring_cap;
    volatile int32_t        surprise_write_idx;
    int32_t                 surprise_read_idx;
    int32_t              surprise_n_embd;

    // Last formatted prompt (captured by chat_generate / prompt_generate)
    char                *last_formatted_prompt;      // malloc'd via strdup
    int32_t              last_formatted_prompt_len;

    // Structured tool call parsing (OpenAI-compatible)
    lfg_tool_call       *parsed_tool_calls;          // malloc'd array
    int32_t              parsed_tool_call_count;
    int32_t              parsed_tool_call_cap;
    char                *last_raw_output;             // detokenized with special=true
    int32_t              last_raw_output_len;
    int32_t              last_raw_output_cap;
    char                *tool_call_text_buf;           // accumulator during generation
    int32_t              tool_call_text_len;
    int32_t              tool_call_text_cap;
    bool                 in_tool_call;                 // flag for generate loop
    lfg_token            tool_call_start_token;        // cached special token ID
    lfg_token            tool_call_end_token;          // cached special token ID
    bool                 tool_call_tokens_cached;
    lfg_tool_call_format tool_call_format;             // default PYTHONIC (0)
    int32_t              tool_call_id_counter;          // for generating unique call_N IDs

    // Guardrails (throughput/latency monitoring)
    bool                 guardrail_active;
    lfg_guardrail_config guardrail_cfg;
    int32_t              guardrail_window_cap;
    int32_t              guardrail_sample_count;
    int32_t              guardrail_write_idx;
    float                guardrail_p50_tps;
    float                guardrail_p95_latency_ms;
    lfg_guardrail_level  guardrail_level;
    float               *guardrail_tps_samples;
    float               *guardrail_latency_samples;
    float               *guardrail_tmp_tps;
    float               *guardrail_tmp_latency;
};

// ---------------------------------------------------------------------------
// Opaque checkpoint struct
// ---------------------------------------------------------------------------

struct lfg_checkpoint {
    int      n_past;
    size_t   token_count;
    size_t   sampler_count;
    uint64_t rng_state;
    int32_t  generated_count;
    size_t   reasoning_token_count;
    uint8_t *state_data;      size_t state_data_size;
    char    *grammar_str;
    char    *grammar_root;
    lfg_sampler *sampler_state;  // owned clone
};

// ---------------------------------------------------------------------------
// Tool ranking helpers (used by both register_tools and decode)
// ---------------------------------------------------------------------------

static uint64_t fnv1a_hash(const char *data, size_t len) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= (uint8_t)data[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

// Append a JSON-escaped version of src to dst. Returns bytes written.
static int32_t json_escape_append(char *dst, const char *src) {
    int32_t w = 0;
    for (const char *p = src; *p; ++p) {
        switch (*p) {
            case '"':  dst[w++] = '\\'; dst[w++] = '"';  break;
            case '\\': dst[w++] = '\\'; dst[w++] = '\\'; break;
            case '\n': dst[w++] = '\\'; dst[w++] = 'n';  break;
            case '\r': dst[w++] = '\\'; dst[w++] = 'r';  break;
            case '\t': dst[w++] = '\\'; dst[w++] = 't';  break;
            default:   dst[w++] = *p; break;
        }
    }
    return w;
}

// Returns malloc'd string. Caller must free.
static char * tool_format_json(const lfg_tool_desc *tool, int32_t *out_len) {
    const char *name = tool->name ? tool->name : "";
    const char *desc = tool->description ? tool->description : "";
    const char *params = tool->parameters;
    bool has_params = params && params[0] != '\0';

    // Worst-case size: every char in name/desc could double from escaping
    size_t name_len = std::strlen(name);
    size_t desc_len = std::strlen(desc);
    size_t params_len = has_params ? std::strlen(params) : 0;
    // {"name": "...", "description": "...", "parameters": ...}
    size_t cap = 64 + name_len * 2 + desc_len * 2 + params_len + 1;

    char *buf = (char *)malloc(cap);
    int32_t w = 0;

    // {"name": "
    std::memcpy(buf + w, "{\"name\": \"", 10); w += 10;
    w += json_escape_append(buf + w, name);
    // ", "description": "
    std::memcpy(buf + w, "\", \"description\": \"", 19); w += 19;
    w += json_escape_append(buf + w, desc);
    buf[w++] = '"';

    if (has_params) {
        // , "parameters": <raw JSON>
        std::memcpy(buf + w, ", \"parameters\": ", 16); w += 16;
        std::memcpy(buf + w, params, params_len); w += (int32_t)params_len;
    }

    buf[w++] = '}';
    buf[w] = '\0';

    if (out_len) *out_len = w;
    return buf;
}

// Build semantic ranking text for a tool.
// We intentionally exclude parameters/schema to reduce JSON boilerplate noise.
static std::string tool_format_rank_text(const lfg_tool_desc *tool) {
    const char *name = tool && tool->name ? tool->name : "";
    const char *desc = tool && tool->description ? tool->description : "";
    std::string out;
    out.reserve(std::strlen(name) + std::strlen(desc) + 32);
    out += "name: ";
    out += name;
    out += "\ndescription: ";
    out += desc;
    return out;
}

static void l2_normalize(float *vec, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += vec[i] * vec[i];
    float inv_norm = (sum > 0.0f) ? (1.0f / std::sqrt(sum)) : 0.0f;
    for (int i = 0; i < n; ++i) vec[i] *= inv_norm;
}

// ---------------------------------------------------------------------------
// Shared embedding context helpers (used by tool ranking + entropy monitor)
// ---------------------------------------------------------------------------

// Ensure session has a tool_ctx for embeddings. Returns false on failure.
static bool session_ensure_embed_ctx(lfg_session *s) {
    if (s->tool_ctx) return true;
    lfg_context_params tparams = lfg_context_default_params();
    tparams.n_ctx = 512;
    tparams.n_batch = 512;
    // Embedding decode is batched; keep batch-thread count aligned with session threads.
    tparams.n_threads = s->config.n_threads;
    tparams.n_threads_batch = s->config.n_threads;
    tparams.embeddings = true;
    tparams.pooling_type = LFG_POOLING_TYPE_MEAN;
    s->tool_ctx = lfg_init_from_model(s->model, tparams);
    return s->tool_ctx != nullptr;
}

// Ensure session has an embed_none_ctx for per-token embeddings. Returns false on failure.
static bool session_ensure_embed_none_ctx(lfg_session *s) {
    if (s->embed_none_ctx) return true;
    lfg_context_params tparams = lfg_context_default_params();
    tparams.n_ctx = 512;
    tparams.n_batch = 512;
    // Per-token embedding path also uses batched decode internally.
    tparams.n_threads = s->config.n_threads;
    tparams.n_threads_batch = s->config.n_threads;
    tparams.embeddings = true;
    tparams.pooling_type = LFG_POOLING_TYPE_NONE;
    s->embed_none_ctx = lfg_init_from_model(s->model, tparams);
    return s->embed_none_ctx != nullptr;
}

// Compute mean-pooled embedding of the current prompt into out_embd.
// Returns true on success. Zero-alloc (uses existing tool_ctx).
static bool compute_query_embedding(lfg_session *s, float *out_embd) {
    if (!s->tool_ctx || s->token_history.size == 0) return false;

    int32_t n_embd = lfg_model_n_embd_out(s->model);
    lfg_memory_clear(lfg_get_memory(s->tool_ctx), true);

    lfg_token *prompt_tokens = s->token_history.data;
    int32_t n_prompt = (int32_t)s->token_history.size;
    int32_t ctx_cap = (int32_t)lfg_n_ctx(s->tool_ctx);
    if (n_prompt > ctx_cap) {
        prompt_tokens = s->token_history.data + (n_prompt - ctx_cap);
        n_prompt = ctx_cap;
    }

    lfg_batch batch = lfg_batch_get_one(prompt_tokens, n_prompt);
    if (lfg_decode(s->tool_ctx, batch) != 0) return false;

    float *embd_ptr = lfg_get_embeddings_seq(s->tool_ctx, 0);
    if (!embd_ptr) embd_ptr = lfg_get_embeddings_ith(s->tool_ctx, -1);
    if (!embd_ptr) return false;

    std::memcpy(out_embd, embd_ptr, n_embd * sizeof(float));
    l2_normalize(out_embd, n_embd);
    return true;
}

// ---------------------------------------------------------------------------
// Internal helpers — ported from InferenceCore methods
// ---------------------------------------------------------------------------

static lfg_sampler * session_find_sampler_by_name(const lfg_session *s, const char *name) {
    if (!s->sampler || !name) return nullptr;
    const char *root_name = lfg_sampler_name(s->sampler);
    if (root_name && std::strcmp(root_name, name) == 0) return s->sampler;
    if (root_name && std::strcmp(root_name, "chain") == 0) {
        const int n = lfg_sampler_chain_n(s->sampler);
        for (int i = 0; i < n; ++i) {
            auto *smpl = lfg_sampler_chain_get(s->sampler, i);
            if (!smpl) continue;
            const char *sn = lfg_sampler_name(smpl);
            if (sn && std::strcmp(sn, name) == 0) return smpl;
        }
    }
    return nullptr;
}

static void session_update_prefix_sampler(lfg_session *s) {
    s->prefix_sampler = session_find_sampler_by_name(s, "prefix");
}

static bool tokenize_literal(const lfg_vocab *vocab, const char *text, bool special, std::vector<lfg_token> &out) {
    if (!vocab || !text) return false;
    const int32_t len = (int32_t)std::strlen(text);
    int32_t cap = len + 8;
    out.resize(cap);
    int32_t n = lfg_tokenize(vocab, text, len, out.data(), cap, false, special);
    if (n < 0) {
        cap = -n;
        out.resize(cap);
        n = lfg_tokenize(vocab, text, len, out.data(), cap, false, special);
    }
    if (n <= 0) {
        out.clear();
        return false;
    }
    out.resize(n);
    return true;
}

static bool session_auto_configure_reasoning_from_markers(lfg_session *s, const char *tmpl_str) {
    if (!s) return false;
    if (s->reasoning_start_count > 0 || s->reasoning_end_count > 0) return false;
    const lfg_vocab *vocab = lfg_model_get_vocab(s->model);
    if (!vocab) return false;

    std::vector<lfg_token> start_tokens;
    std::vector<lfg_token> end_tokens;
    if (!tokenize_literal(vocab, "<think>", true, start_tokens)) return false;
    if (!tokenize_literal(vocab, "</think>", true, end_tokens)) return false;

    // Prefer explicit signals (template or metadata), but fall back to marker presence
    // so reasoning markers work with models that emit <think> without templating.
    bool enable = false;
    if (tmpl_str && std::strstr(tmpl_str, "<think>") != nullptr) {
        enable = true;
    } else {
        char name_buf[256];
        const int32_t n = lfg_model_get_metadata_str(s->model, "general.name", name_buf, sizeof(name_buf));
        if (n > 0) {
            std::string name(name_buf, (size_t)n);
            std::transform(name.begin(), name.end(), name.begin(),
                           [] (const std::string::value_type x) { return std::tolower(x); });
            if (name.find("thinking") != std::string::npos || name.find("reasoning") != std::string::npos) {
                enable = true;
            }
        }
    }
    if (!enable) {
        // Fall back: if the vocab can represent the markers, allow auto-config.
        // This is harmless unless the model emits <think> tokens.
        enable = !start_tokens.empty() && !end_tokens.empty();
    }
    if (!enable) return false;

    lfg_session_configure_reasoning(s, start_tokens.data(), start_tokens.size(),
                                    end_tokens.data(), end_tokens.size());
    return true;
}

static lfg_sampler * session_snapshot_sampler(const lfg_session *s) {
    if (!s->sampler) return nullptr;
    lfg_sampler *clone = lfg_sampler_clone(s->sampler);
    if (!clone) {
        spdlog::warn("Failed to clone sampler state for snapshot.");
        return nullptr;
    }
    const char *root_name = lfg_sampler_name(s->sampler);
    if (root_name && std::strcmp(root_name, "chain") == 0) {
        const int n_src = lfg_sampler_chain_n(s->sampler);
        const int n_clone = lfg_sampler_chain_n(clone);
        if (n_src != n_clone) {
            spdlog::warn("Sampler snapshot clone mismatch ({} vs {}).", n_src, n_clone);
            lfg_sampler_free(clone);
            return nullptr;
        }
    }
    return clone;
}

static bool session_set_sampler_from_snapshot(lfg_session *s, lfg_sampler *snapshot) {
    if (!snapshot) return false;
    lfg_sampler *clone = lfg_sampler_clone(snapshot);
    if (!clone) {
        spdlog::warn("Failed to clone sampler state for restore.");
        return false;
    }
    if (s->sampler) {
        lfg_sampler_free(s->sampler);
    }
    s->sampler = clone;
    session_update_prefix_sampler(s);
    return s->sampler != nullptr;
}

static bool session_should_record_sampler_token(const lfg_session *s, lfg_token token) {
    if (!s->model) return false;
    lfg_token bos = lfg_vocab_bos(lfg_model_get_vocab(s->model));
    return token != bos;
}

static void session_record_sampler_token(lfg_session *s, lfg_token token, bool pending) {
    if (!s->sampler_recording_enabled) return;
    if (!session_should_record_sampler_token(s, token)) return;
    lfg_buf_tokens_push(&s->sampler_history, token);
    if (pending) {
        s->pending_sampler_accepts++;
    }
}

static bool session_history_ends_with(const lfg_session *s, const lfg_token *pattern, size_t n) {
    if (n == 0 || s->token_history.size < n) return false;
    const lfg_token *tail = s->token_history.data + s->token_history.size - n;
    return std::memcmp(tail, pattern, n * sizeof(lfg_token)) == 0;
}

static void session_rebuild_sampler(lfg_session *s) {
    if (s->sampler) {
        lfg_sampler_free(s->sampler);
        s->sampler = nullptr;
        s->prefix_sampler = nullptr;
    }

    auto sparams = lfg_sampler_chain_default_params();
    s->sampler = lfg_sampler_chain_init(sparams);

    s->prefix_sampler = lfg_sampler_init_prefix(lfg_model_get_vocab(s->model), "");
    lfg_sampler_chain_add(s->sampler, s->prefix_sampler);

    if (s->grammar_str && s->grammar_str[0] != '\0') {
        lfg_sampler *grammar_sampler = nullptr;

        grammar_sampler = lfg_sampler_init_grammar(
            lfg_model_get_vocab(s->model),
            s->grammar_str,
            s->grammar_root ? s->grammar_root : "root"
        );

        if (s->reasoning_start_count > 0 && s->reasoning_end_count > 0) {
            grammar_sampler = lfg_sampler_init_reasoning_gate(
                grammar_sampler,
                s->reasoning_start_tokens, s->reasoning_start_count,
                s->reasoning_end_tokens, s->reasoning_end_count
            );
        }

        if (grammar_sampler) {
            lfg_sampler_chain_add(s->sampler, grammar_sampler);
        } else {
            lfg_set_last_error(LFG_ERROR_INTERNAL,
                "session_rebuild_sampler: failed to initialize grammar sampler");
        }
    }

    const lfg_sampling_config *sp = &s->config.sampling;
    
    // Optimization: If using greedy sampling (temp <= 0), we can skip prob manipulators
    // like top_k, top_p, etc. as they just filter the distribution but argmax remains the same.
    bool is_greedy = (sp->temp <= 0.0f);

    if (sp->penalty_repeat != 1.0f || sp->penalty_freq != 0.0f || sp->penalty_present != 0.0f) {
        lfg_sampler_chain_add(s->sampler, lfg_sampler_init_penalties(
            sp->penalty_last_n, sp->penalty_repeat, sp->penalty_freq, sp->penalty_present));
    }

    if (!is_greedy) {
        if (sp->top_k > 0) {
            lfg_sampler_chain_add(s->sampler, lfg_sampler_init_top_k(sp->top_k));
        }

        if (sp->typ_p < 1.0f) {
            lfg_sampler_chain_add(s->sampler, lfg_sampler_init_typical(sp->typ_p, 1));
        }

        if (sp->top_p < 1.0f) {
            lfg_sampler_chain_add(s->sampler, lfg_sampler_init_top_p(sp->top_p, 1));
        }

        if (sp->min_p > 0.0f) {
            lfg_sampler_chain_add(s->sampler, lfg_sampler_init_min_p(sp->min_p, 1));
        }
    }

    session_update_prefix_sampler(s);

    if (!is_greedy) {
        lfg_sampler_chain_add(s->sampler, lfg_sampler_init_temp(sp->temp));
        lfg_sampler_chain_add(s->sampler, lfg_sampler_init_dist(sp->seed));
    } else {
        lfg_sampler_chain_add(s->sampler, lfg_sampler_init_greedy());
    }
}

static void session_rebuild_sampler_from_history(lfg_session *s, bool full_reingest) {
    if (!s->sampler) return;

    lfg_sampler_reset(s->sampler);

    int start_idx = 0;
    if (!full_reingest) {
        const int reingest_count = 64;
        start_idx = std::max(0, (int)s->sampler_history.size - reingest_count);
    }

    const bool prev_recording = s->sampler_recording_enabled;
    s->sampler_recording_enabled = false;
    for (int i = start_idx; i < (int)s->sampler_history.size; ++i) {
        lfg_sampler_accept(s->sampler, s->sampler_history.data[i]);
    }
    s->sampler_recording_enabled = prev_recording;
}

static void session_truncate_history(lfg_session *s, size_t new_size) {
    if (new_size >= s->token_history.size) return;
    s->token_history.size = new_size;
    s->sampler_offsets.size = new_size;

    size_t new_sampler_size = 0;
    if (new_size > 0) {
        new_sampler_size = s->sampler_offsets.data[new_size - 1];
    }
    s->sampler_history.size = new_sampler_size;
    s->pending_sampler_accepts = 0;
}

// ---------------------------------------------------------------------------
// Surprise reasoning map helpers
// ---------------------------------------------------------------------------

// Check whether a token pattern ends at tokens[idx], considering prior history.
static bool seq_ends_at(const lfg_session *s, const lfg_token *tokens,
                        size_t idx, size_t hist_len,
                        const lfg_token *pattern, size_t pat_n) {
    if (pat_n == 0) return false;
    if (idx + 1 + hist_len < pat_n) return false;
    for (size_t k = 0; k < pat_n; ++k) {
        size_t rev = pat_n - 1 - k;
        lfg_token t = (k < idx + 1)
            ? tokens[idx - k]
            : s->token_history.data[hist_len - 1 - (k - idx - 1)];
        if (t != pattern[rev]) return false;
    }
    return true;
}

// Build a boolean map[n_tokens] indicating which positions are inside reasoning blocks.
// Caller must free() the returned pointer.
static bool * build_reasoning_map(const lfg_session *s,
                                  const lfg_token *tokens, size_t n_tokens) {
    bool *map = (bool *)calloc(n_tokens, sizeof(bool));
    bool in_r = s->in_reasoning;
    size_t hist_len = s->token_history.size;
    for (size_t i = 0; i < n_tokens; ++i) {
        if (!in_r && seq_ends_at(s, tokens, i, hist_len,
                s->reasoning_start_tokens, s->reasoning_start_count))
            in_r = true;
        map[i] = in_r;
        if (in_r && seq_ends_at(s, tokens, i, hist_len,
                s->reasoning_end_tokens, s->reasoning_end_count))
            in_r = false;
    }
    return map;
}

// ---------------------------------------------------------------------------
// Internal ingestion
// ---------------------------------------------------------------------------

static bool session_ingest_internal(lfg_session *s, const lfg_token *tokens, size_t n_tokens, bool update_sampler) {
    if (!s->ctx) return false;

    int32_t n_batch = s->config.n_batch;
    if (n_batch <= 0) n_batch = 512;

    // Pre-scan reasoning state for surprise gating (needs full token view before batching)
    bool *reasoning_map = nullptr;
    if (s->surprise_active && !s->surprise_include_reasoning &&
        (s->reasoning_start_count > 0 || s->reasoning_end_count > 0)) {
        reasoning_map = build_reasoning_map(s, tokens, n_tokens);
    }

    // Ensure buffers have room
    if (n_tokens > 0) {
        lfg_buf_tokens_ensure(&s->token_history, s->token_history.size + n_tokens);
        lfg_buf_size_ensure(&s->sampler_offsets, s->sampler_offsets.size + n_tokens);
    }

    // Ensure pos buffer is big enough
    if ((size_t)n_batch > s->pos_buf_cap) {
        s->pos_buf = (lfg_pos *)realloc(s->pos_buf, (size_t)n_batch * sizeof(lfg_pos));
        s->pos_buf_cap = (size_t)n_batch;
    }

    for (size_t i = 0; i < n_tokens; i += n_batch) {
        int32_t n = std::min((int32_t)(n_tokens - i), n_batch);

        for (int p = 0; p < n; ++p) {
            s->pos_buf[p] = s->n_past + p;
        }

        lfg_batch batch = {};
        batch.n_tokens = n;
        batch.token = const_cast<lfg_token *>(tokens + i);
        batch.embd = nullptr;
        batch.pos = s->pos_buf;
        batch.n_seq_id = nullptr;
        batch.seq_id = nullptr;

        // Enable all-position logits when surprise monitor is active
        if (s->surprise_active && n > 1) {
            // Ensure logits_buf is large enough
            if ((size_t)n > s->logits_buf_cap) {
                s->logits_buf = (int8_t *)realloc(s->logits_buf, (size_t)n * sizeof(int8_t));
                s->logits_buf_cap = (size_t)n;
            }
            for (int p = 0; p < n; ++p) s->logits_buf[p] = 1;
            batch.logits = s->logits_buf;
        } else {
            batch.logits = nullptr;
        }

        if (lfg_decode(s->ctx, batch) != 0) {
            spdlog::error("lfg_decode failed");
            free(reasoning_map);
            return false;
        }

        // --- Per-token surprise accumulation ---
        if (s->surprise_active && n > 1) {
            const lfg_vocab *vocab = lfg_model_get_vocab(s->model);
            int n_vocab = lfg_vocab_n_tokens(vocab);
            float log_v = logf((float)n_vocab);

            // For position p in [0, n-2]: logits[p] predict tokens[i+p+1]
            for (int p = 0; p < n - 1; ++p) {
                if (i == 0 && p == 0) continue;  // skip BOS — no context

                size_t next_idx = i + (size_t)p + 1;

                // Skip reasoning tokens when include_reasoning is false
                if (reasoning_map && reasoning_map[next_idx]) continue;

                // Skip chat context prefix tokens (chat-scoped surprise)
                if (s->surprise_skip_tokens > 0 && (int32_t)next_idx < s->surprise_skip_tokens) continue;

                float *logits = lfg_get_logits_ith(s->ctx, p);
                if (!logits) continue;
                lfg_token next_tok = tokens[next_idx];

                // 2-pass softmax: find max, sum exp, lookup P(next_tok)
                float max_l = logits[0];
                for (int v = 1; v < n_vocab; ++v)
                    if (logits[v] > max_l) max_l = logits[v];
                float sum_exp = 0.0f;
                for (int v = 0; v < n_vocab; ++v)
                    sum_exp += expf(logits[v] - max_l);
                float p_tok = expf(logits[next_tok] - max_l) / sum_exp;
                float surprise = (p_tok > 0.0f)
                    ? (-logf(p_tok) / log_v) : 1.0f;

                // Accumulate aggregate stats
                s->surprise_n_evaluated++;
                if (s->surprise_gate_mode == LFG_SURPRISE_GATE_AUTO) {
                    // Store for two-pass (realloc if needed)
                    int32_t idx = s->surprise_n_evaluated - 1;
                    if (idx >= s->surprise_per_token_cap) {
                        s->surprise_per_token_cap = (idx + 1) * 2;
                        s->surprise_per_token = (float *)realloc(s->surprise_per_token,
                            s->surprise_per_token_cap * sizeof(float));
                    }
                    s->surprise_per_token[idx] = surprise;
                } else {
                    // FIXED: accumulate on the fly
                    if (surprise >= s->surprise_threshold) {
                        s->surprise_count++;
                        s->surprise_sum += surprise;
                        if (surprise > s->surprise_max)
                            s->surprise_max = surprise;
                    }
                }
            }
        }

        s->n_past += n;
    }

    // AUTO two-pass: compute mean, then flag outliers
    if (s->surprise_active && s->surprise_gate_mode == LFG_SURPRISE_GATE_AUTO &&
        s->surprise_n_evaluated > 0) {
        float sum = 0.0f;
        for (int32_t i = 0; i < s->surprise_n_evaluated; ++i) {
            sum += s->surprise_per_token[i];
        }
        float mean = sum / s->surprise_n_evaluated;
        float gap = s->surprise_threshold > 0.0f ? s->surprise_threshold : 0.20f;
        float effective_threshold = mean + gap;

        for (int32_t i = 0; i < s->surprise_n_evaluated; ++i) {
            if (s->surprise_per_token[i] >= effective_threshold) {
                s->surprise_count++;
                s->surprise_sum += s->surprise_per_token[i];
                if (s->surprise_per_token[i] > s->surprise_max) {
                    s->surprise_max = s->surprise_per_token[i];
                }
            }
        }
    }

    // Post-ingest: emit aggregate event if any tokens exceeded threshold
    if (s->surprise_active && s->surprise_count > 0) {
        int32_t wi = s->surprise_write_idx;
        int slot = wi % s->surprise_ring_cap;

        float *slot_embd = s->surprise_embd_pool + (slot * s->surprise_n_embd);
        bool got_embd = compute_query_embedding(s, slot_embd);

        lfg_surprise_ring_slot *rs = &s->surprise_slots[slot];
        rs->event.mean_surprise = s->surprise_sum / s->surprise_count;
        rs->event.max_surprise = s->surprise_max;
        rs->event.n_above_threshold = s->surprise_count;
        rs->event.n_tokens_evaluated = s->surprise_n_evaluated;
        rs->event.n_embd = got_embd ? s->surprise_n_embd : 0;
        rs->embedding = got_embd ? slot_embd : nullptr;

        __atomic_store_n(&s->surprise_write_idx, wi + 1, __ATOMIC_RELEASE);
    }

    // Update sampler + history + offsets
    size_t pending = s->pending_sampler_accepts;

    for (size_t i = 0; i < n_tokens; ++i) {
        const lfg_token token = tokens[i];

        if (s->sampler) {
            if (pending > 0) {
                pending--;
            } else if (update_sampler) {
                if (session_should_record_sampler_token(s, token)) {
                    lfg_sampler_accept(s->sampler, token);
                }
                session_record_sampler_token(s, token, false);
            }
        }

        lfg_buf_tokens_push(&s->token_history, token);
        lfg_buf_size_push(&s->sampler_offsets, s->sampler_history.size);

        // Track reasoning state transitions
        if (s->reasoning_start_count > 0 || s->reasoning_end_count > 0) {
            if (!s->in_reasoning &&
                session_history_ends_with(s, s->reasoning_start_tokens, s->reasoning_start_count)) {
                s->in_reasoning = true;
                s->reasoning_token_count = 0;
            } else if (session_history_ends_with(s, s->reasoning_end_tokens, s->reasoning_end_count)) {
                s->in_reasoning = false;
            } else if (s->in_reasoning) {
                s->reasoning_token_count++;
            }
        }
    }
    s->pending_sampler_accepts = pending;

    free(reasoning_map);
    return true;
}

// ---------------------------------------------------------------------------
// Public API — configuration defaults
// ---------------------------------------------------------------------------

LFG_API lfg_sampling_config lfg_sampling_default_config(void) {
    lfg_sampling_config cfg{};
    cfg.seed = 0xFFFFFFFF;
    cfg.n_prev = 64;
    cfg.top_k = 50;
    cfg.top_p = 0.1f;
    cfg.min_p = 0.05f;
    cfg.typ_p = 1.0f;
    cfg.temp = 0.1f;
    cfg.penalty_last_n = 64;
    cfg.penalty_repeat = 1.05f;
    cfg.penalty_freq = 0.0f;
    cfg.penalty_present = 0.0f;
    cfg.mirostat = 0;
    cfg.mirostat_tau = 5.0f;
    cfg.mirostat_eta = 0.10f;
    return cfg;
}

LFG_API lfg_session_config lfg_session_default_config(void) {
    lfg_session_config cfg{};
    cfg.n_threads = 4;
    cfg.n_ctx = 2048;
    cfg.n_batch = 512;
    cfg.enable_healing = false;
    cfg.structured_checkpointing = true;
    cfg.max_tokens = 0;
    cfg.tool_score_mode = LFG_TOOL_SCORE_OFF;
    cfg.tool_min_score = 0.0f;
    cfg.sampling = lfg_sampling_default_config();
    return cfg;
}

LFG_API lfg_guardrail_config lfg_guardrail_default_config(void) {
    lfg_guardrail_config cfg{};
    cfg.enabled = false;
    cfg.window_size = 64;
    cfg.p50_tps_min = 0.0f;
    cfg.p95_latency_ms_max = 0.0f;
    cfg.memory_cap_bytes = 0;
    return cfg;
}

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------

LFG_API lfg_session * lfg_session_create(lfg_model * model, const lfg_session_config * config) {
    if (!model) return nullptr;
    lfg_session_config cfg = config ? *config : lfg_session_default_config();

    lfg_context_params params = lfg_context_default_params();
    params.n_ctx = cfg.n_ctx;
    params.n_threads = cfg.n_threads;
    params.n_threads_batch = cfg.n_threads;
    params.n_batch = cfg.n_batch;

    lfg_context *ctx = lfg_init_from_model(model, params);
    if (!ctx) {
        spdlog::error("Failed to create liquid context");
        return nullptr;
    }

    auto *s = (lfg_session *)calloc(1, sizeof(lfg_session));
    s->model = model;
    s->ctx = ctx;
    s->config = cfg;
    s->n_past = 0;

    // Pre-allocate buffers to n_ctx capacity
    size_t ctx_cap = (size_t)cfg.n_ctx;
    lfg_buf_tokens_init(&s->token_history, ctx_cap);
    lfg_buf_tokens_init(&s->sampler_history, ctx_cap);
    lfg_buf_size_init(&s->sampler_offsets, ctx_cap);
    s->pending_sampler_accepts = 0;
    s->sampler_recording_enabled = true;

    s->reasoning_start_tokens = nullptr;
    s->reasoning_start_count = 0;
    s->reasoning_end_tokens = nullptr;
    s->reasoning_end_count = 0;

    s->sampler = nullptr;
    s->prefix_sampler = nullptr;

    s->grammar_str = nullptr;
    s->grammar_root = nullptr;

    s->in_reasoning = false;
    s->reasoning_token_count = 0;

    lfg_buf_u8_init(&s->healing_state_buffer, 0);
    s->healing_n_past = -1;
    s->healing_sampler_snapshot = nullptr;

    // Scratch buffers
    size_t batch_cap = (size_t)cfg.n_batch;
    s->pos_buf = (lfg_pos *)malloc(batch_cap * sizeof(lfg_pos));
    s->pos_buf_cap = batch_cap;
    s->logits_buf = (int8_t *)malloc(batch_cap * sizeof(int8_t));
    s->logits_buf_cap = batch_cap;

    // Tool ranking (all nullptr/zero — allocated in register_tools)
    s->tool_ctx = nullptr;
    s->embed_none_ctx = nullptr;
    s->tool_entries = nullptr;
    s->tool_count = 0;
    s->tool_n_embd = 0;
    s->tool_cache_hashes = nullptr;
    s->tool_cache_embeds = nullptr;
    s->tool_cache_count = 0;
    s->tool_cache_cap = 0;
    s->tool_query_embd = nullptr;
    s->tool_score_indices = nullptr;
    s->tool_scores = nullptr;
    s->tool_json_buf = nullptr;
    s->tool_json_buf_cap = 0;
    s->tool_token_buf = nullptr;
    s->tool_token_buf_cap = 0;
    s->tool_top_k = 0;
    s->tool_score_mode = cfg.tool_score_mode;
    s->tool_min_score = cfg.tool_min_score;
    s->tools_injected = false;

    // Entropy monitor (all nullptr/zero — allocated in configure_entropy_monitor)
    s->entropy_threshold = 0.0f;
    s->entropy_cooldown = 1;
    s->entropy_tokens_since = 0;
    s->entropy_last = -1.0f;
    s->entropy_last_norm = -1.0f;
    s->entropy_active = false;
    s->entropy_gate_mode = LFG_ENTROPY_GATE_OFF;
    s->entropy_running_sum = 0.0f;
    s->entropy_running_count = 0;
    s->entropy_slots = nullptr;
    s->entropy_snaps = nullptr;
    s->entropy_embd_pool = nullptr;
    s->entropy_ring_cap = 0;
    s->entropy_write_idx = 0;
    s->entropy_read_idx = 0;
    s->entropy_next_id = 0;
    s->entropy_n_embd = 0;

    // Confidence monitor (all nullptr/zero — allocated in configure_confidence_monitor)
    s->confidence_threshold = 0.0f;
    s->confidence_min_span = 5;
    s->confidence_active = false;
    s->confidence_gate_mode = LFG_CONFIDENCE_GATE_OFF;
    s->confidence_running_sum = 0.0f;
    s->confidence_running_count = 0;
    s->confidence_include_reasoning = false;
    s->confidence_run_count = 0;
    s->confidence_run_entropy_sum = 0.0f;
    s->confidence_run_min_entropy = 1.0f;
    s->confidence_run_start_pos = 0;
    s->confidence_slots = nullptr;
    s->confidence_embd_pool = nullptr;
    s->confidence_ring_cap = 0;
    s->confidence_write_idx = 0;
    s->confidence_read_idx = 0;
    s->confidence_n_embd = 0;

    // Surprise monitor (all zero — allocated in configure_surprise_monitor)
    s->surprise_threshold = 0.0f;
    s->surprise_active = false;
    s->surprise_include_reasoning = false;
    s->surprise_skip_tokens = 0;
    s->surprise_gate_mode = LFG_SURPRISE_GATE_OFF;
    s->surprise_per_token = nullptr;
    s->surprise_per_token_cap = 0;
    s->surprise_count = 0;
    s->surprise_sum = 0.0f;
    s->surprise_max = 0.0f;
    s->surprise_n_evaluated = 0;
    s->surprise_slots = nullptr;
    s->surprise_embd_pool = nullptr;
    s->surprise_ring_cap = 0;
    s->surprise_write_idx = 0;
    s->surprise_read_idx = 0;
    s->surprise_n_embd = 0;

    // Tool call parsing (pre-allocate buffers)
    s->parsed_tool_calls = nullptr;
    s->parsed_tool_call_count = 0;
    s->parsed_tool_call_cap = 0;
    s->last_raw_output = (char *)malloc(2048);
    s->last_raw_output_len = 0;
    s->last_raw_output_cap = 2048;
    s->tool_call_text_buf = (char *)malloc(1024);
    s->tool_call_text_len = 0;
    s->tool_call_text_cap = 1024;
    s->in_tool_call = false;
    s->tool_call_start_token = LFG_TOKEN_NULL;
    s->tool_call_end_token = LFG_TOKEN_NULL;
    s->tool_call_tokens_cached = false;
    s->tool_call_format = LFG_TOOL_CALL_FORMAT_PYTHONIC;
    s->tool_call_id_counter = 0;

    // Guardrails (inactive by default)
    s->guardrail_active = false;
    s->guardrail_cfg = lfg_guardrail_default_config();
    s->guardrail_window_cap = 0;
    s->guardrail_sample_count = 0;
    s->guardrail_write_idx = 0;
    s->guardrail_p50_tps = 0.0f;
    s->guardrail_p95_latency_ms = 0.0f;
    s->guardrail_level = LFG_GUARDRAIL_LEVEL_NONE;
    s->guardrail_tps_samples = nullptr;
    s->guardrail_latency_samples = nullptr;
    s->guardrail_tmp_tps = nullptr;
    s->guardrail_tmp_latency = nullptr;

    session_rebuild_sampler(s);

    return s;
}

// Free strdup'd strings inside parsed_tool_calls and reset count.
static void session_free_tool_calls(lfg_session * s) {
    for (int32_t i = 0; i < s->parsed_tool_call_count; ++i) {
        free((void *)s->parsed_tool_calls[i].id);
        free((void *)s->parsed_tool_calls[i].name);
        free((void *)s->parsed_tool_calls[i].arguments);
    }
    s->parsed_tool_call_count = 0;
}

LFG_API void lfg_session_free(lfg_session * session) {
    if (!session) return;
    if (session->sampler) lfg_sampler_free(session->sampler);
    lfg_session_clear_tools(session);
    if (session->embed_none_ctx) lfg_free(session->embed_none_ctx);
    if (session->ctx) lfg_free(session->ctx);
    if (session->healing_sampler_snapshot) lfg_sampler_free(session->healing_sampler_snapshot);

    lfg_buf_tokens_free(&session->token_history);
    lfg_buf_tokens_free(&session->sampler_history);
    lfg_buf_size_free(&session->sampler_offsets);
    lfg_buf_u8_free(&session->healing_state_buffer);

    free(session->reasoning_start_tokens);
    free(session->reasoning_end_tokens);
    free(session->grammar_str);
    free(session->grammar_root);
    free(session->stop_flat);
    free(session->stop_offsets);
    free(session->stop_lengths);
    for (int32_t i = 0; i < session->stop_text_count; i++) {
        free(session->stop_texts[i]);
    }
    free(session->stop_texts);
    free(session->stop_text_lens);
    free(session->pos_buf);
    free(session->logits_buf);

    // Entropy monitor
    // Zero the entropy counter before freeing so stale pointers read 0 instead
    // of ghost data. Does not prevent UB, but makes use-after-free benign in
    // practice (polling threads see 0 pending and stop).
    session->entropy_write_idx = 0;
    session->entropy_read_idx = 0;

    free(session->entropy_slots);
    free(session->entropy_snaps);
    free(session->entropy_embd_pool);

    // Confidence monitor
    session->confidence_write_idx = 0;
    session->confidence_read_idx = 0;

    free(session->confidence_slots);
    free(session->confidence_embd_pool);
    free(session->confidence_text_buf);

    // Surprise monitor
    session->surprise_write_idx = 0;
    session->surprise_read_idx = 0;
    free(session->surprise_slots);
    free(session->surprise_embd_pool);
    free(session->surprise_per_token);

    // Last formatted prompt
    free(session->last_formatted_prompt);

    // Tool call parsing
    session_free_tool_calls(session);
    free(session->parsed_tool_calls);
    free(session->last_raw_output);
    free(session->tool_call_text_buf);

    // Guardrails
    free(session->guardrail_tps_samples);
    free(session->guardrail_latency_samples);
    free(session->guardrail_tmp_tps);
    free(session->guardrail_tmp_latency);

    free(session);
}

LFG_API void lfg_session_reset(lfg_session * session) {
    if (!session) return;
    if (session->ctx) {
        lfg_memory_clear(lfg_get_memory(session->ctx), true);
    }
    if (session->sampler) {
        lfg_sampler_reset(session->sampler);
    }
    session->n_past = 0;
    session->token_history.size = 0;
    session->sampler_history.size = 0;
    session->sampler_offsets.size = 0;
    session->pending_sampler_accepts = 0;
    session->generated_count = 0;
    session->in_reasoning = false;
    session->reasoning_token_count = 0;
    session->healing_n_past = -1;
    session->healing_state_buffer.size = 0;
    if (session->healing_sampler_snapshot) {
        lfg_sampler_free(session->healing_sampler_snapshot);
        session->healing_sampler_snapshot = nullptr;
    }
    session->tools_injected = false;

    // Entropy monitor reset
    if (session->entropy_active) {
        session->entropy_write_idx = 0;
        session->entropy_read_idx = 0;
        session->entropy_next_id = 0;
        session->entropy_tokens_since = session->entropy_cooldown; // allow first event immediately
        session->entropy_last = -1.0f;
        session->entropy_last_norm = -1.0f;
        session->entropy_running_sum = 0.0f;
        session->entropy_running_count = 0;
        // Invalidate all snaps
        for (int32_t i = 0; i < session->entropy_ring_cap; ++i) {
            session->entropy_snaps[i].valid = false;
        }
    }

    // Confidence monitor reset
    if (session->confidence_active) {
        session->confidence_write_idx = 0;
        session->confidence_read_idx = 0;
        session->confidence_run_count = 0;
        session->confidence_run_entropy_sum = 0.0f;
        session->confidence_run_min_entropy = 1.0f;
        session->confidence_running_sum = 0.0f;
        session->confidence_running_count = 0;
    }

    // Surprise monitor reset
    if (session->surprise_active) {
        session->surprise_count = 0;
        session->surprise_sum = 0.0f;
        session->surprise_max = 0.0f;
        session->surprise_n_evaluated = 0;
        session->surprise_write_idx = 0;
        session->surprise_read_idx = 0;
        session->surprise_skip_tokens = 0;
    }

    // Guardrails reset
    if (session->guardrail_active) {
        session->guardrail_sample_count = 0;
        session->guardrail_write_idx = 0;
        session->guardrail_p50_tps = 0.0f;
        session->guardrail_p95_latency_ms = 0.0f;
        session->guardrail_level = LFG_GUARDRAIL_LEVEL_NONE;
    }

    // Last formatted prompt
    free(session->last_formatted_prompt);
    session->last_formatted_prompt = nullptr;
    session->last_formatted_prompt_len = 0;

    // Tool call parsing reset
    session_free_tool_calls(session);
    session->last_raw_output_len = 0;
    session->tool_call_text_len = 0;
    session->in_tool_call = false;
    session->tool_call_id_counter = 0;
}

// ---------------------------------------------------------------------------
// Structured decoding + reasoning config
// ---------------------------------------------------------------------------

LFG_API bool lfg_session_configure_structured(lfg_session * session,
                                                    const char * grammar_or_schema,
                                                    const char * root_rule) {
    if (!session || !grammar_or_schema) return false;

    // Clear any stale error from previous API calls so the post-rebuild check is accurate.
    lfg_clear_last_error();

    // Free old grammar strings
    free(session->grammar_str);
    free(session->grammar_root);
    session->grammar_str = nullptr;
    session->grammar_root = nullptr;

    if (grammar_or_schema[0] == '{') {
        try {
            auto json = nlohmann::ordered_json::parse(grammar_or_schema);
            std::string converted = json_schema_to_grammar(json);
            session->grammar_str = strdup(converted.c_str());
        } catch (const std::exception &e) {
            spdlog::error("Failed to parse JSON schema: {}. Using raw grammar.", e.what());
            session->grammar_str = strdup(grammar_or_schema);
        }
    } else {
        session->grammar_str = strdup(grammar_or_schema);
    }
    session->grammar_root = strdup(root_rule ? root_rule : "root");

    session_rebuild_sampler(session);
    if (session->healing_sampler_snapshot) {
        lfg_sampler_free(session->healing_sampler_snapshot);
        session->healing_sampler_snapshot = nullptr;
    }

    // Verify grammar sampler was successfully created
    if (session->grammar_str && session->grammar_str[0] != '\0') {
        if (lfg_get_last_error(nullptr, 0) != LFG_ERROR_NONE) {
            return false;
        }
    }
    return true;
}

LFG_API void lfg_session_configure_reasoning(lfg_session * session,
                                                   const lfg_token * start_tokens, size_t n_start,
                                                   const lfg_token * end_tokens,   size_t n_end) {
    if (!session) return;

    free(session->reasoning_start_tokens);
    free(session->reasoning_end_tokens);

    session->reasoning_start_tokens = nullptr;
    session->reasoning_start_count = 0;
    session->reasoning_end_tokens = nullptr;
    session->reasoning_end_count = 0;

    if (start_tokens && n_start > 0) {
        session->reasoning_start_tokens = (lfg_token *)malloc(n_start * sizeof(lfg_token));
        memcpy(session->reasoning_start_tokens, start_tokens, n_start * sizeof(lfg_token));
        session->reasoning_start_count = n_start;
    }
    if (end_tokens && n_end > 0) {
        session->reasoning_end_tokens = (lfg_token *)malloc(n_end * sizeof(lfg_token));
        memcpy(session->reasoning_end_tokens, end_tokens, n_end * sizeof(lfg_token));
        session->reasoning_end_count = n_end;
    }

    session_rebuild_sampler(session);
}

LFG_API bool lfg_session_configure_stop_sequences(
        lfg_session * session,
        const lfg_token * const * sequences,
        const size_t * sequence_lengths,
        size_t n_sequences) {
    if (!session) return false;

    // Free old storage
    free(session->stop_flat);
    free(session->stop_offsets);
    free(session->stop_lengths);
    session->stop_flat = nullptr;
    session->stop_offsets = nullptr;
    session->stop_lengths = nullptr;
    session->stop_count = 0;
    session->stop_max_len = 0;

    if (n_sequences == 0 || !sequences || !sequence_lengths) return true;

    // Compute total flat size and max length
    size_t total = 0;
    size_t max_len = 0;
    for (size_t i = 0; i < n_sequences; ++i) {
        if (sequence_lengths[i] == 0 || !sequences[i]) continue;
        total += sequence_lengths[i];
        if (sequence_lengths[i] > max_len) max_len = sequence_lengths[i];
    }

    if (total == 0) return true;

    session->stop_flat    = (lfg_token *)malloc(total * sizeof(lfg_token));
    session->stop_offsets = (size_t *)malloc(n_sequences * sizeof(size_t));
    session->stop_lengths = (size_t *)malloc(n_sequences * sizeof(size_t));

    size_t offset = 0;
    size_t count = 0;
    for (size_t i = 0; i < n_sequences; ++i) {
        if (sequence_lengths[i] == 0 || !sequences[i]) continue;
        session->stop_offsets[count] = offset;
        session->stop_lengths[count] = sequence_lengths[i];
        memcpy(session->stop_flat + offset, sequences[i], sequence_lengths[i] * sizeof(lfg_token));
        offset += sequence_lengths[i];
        count++;
    }
    session->stop_count = count;
    session->stop_max_len = max_len;

    return true;
}

LFG_API bool lfg_session_configure_stop_strings(
        lfg_session * session,
        const char * const * strings,
        int32_t n_strings) {
    if (!session) return false;

    // Free old storage
    for (int32_t i = 0; i < session->stop_text_count; i++) {
        free(session->stop_texts[i]);
    }
    free(session->stop_texts);
    free(session->stop_text_lens);
    session->stop_texts = nullptr;
    session->stop_text_lens = nullptr;
    session->stop_text_count = 0;
    session->stop_text_max_len = 0;

    if (n_strings <= 0 || !strings) return true;

    session->stop_texts = (char **)malloc(n_strings * sizeof(char *));
    session->stop_text_lens = (int32_t *)malloc(n_strings * sizeof(int32_t));

    int32_t count = 0;
    int32_t max_len = 0;
    for (int32_t i = 0; i < n_strings; i++) {
        if (!strings[i] || strings[i][0] == '\0') continue;
        int32_t len = (int32_t)std::strlen(strings[i]);
        session->stop_texts[count] = strdup(strings[i]);
        session->stop_text_lens[count] = len;
        if (len > max_len) max_len = len;
        count++;
    }
    session->stop_text_count = count;
    session->stop_text_max_len = max_len;

    return true;
}

LFG_API int32_t lfg_json_schema_to_grammar(const char * json_schema,
                                                 bool force_gbnf,
                                                 char * buf,
                                                 size_t buf_size) {
    if (!json_schema) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: json_schema is NULL", __func__);
        return -1;
    }
    if (!buf && buf_size > 0) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT, "%s: buf is NULL", __func__);
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
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT,
                              "%s: failed to convert JSON schema: %s",
                              __func__,
                              err.what());
        if (buf && buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
}

// ---------------------------------------------------------------------------
// Token ingestion / decoding
// ---------------------------------------------------------------------------

LFG_API bool lfg_session_ingest_tokens(lfg_session * session,
                                             const lfg_token * tokens,
                                             size_t n_tokens,
                                             bool update_sampler) {
    if (!session || !tokens) return false;

    if (session->surprise_active) {
        session->surprise_count = 0;
        session->surprise_sum = 0.0f;
        session->surprise_max = 0.0f;
        session->surprise_n_evaluated = 0;
    }

    if (!session->config.enable_healing || n_tokens == 0) {
        return session_ingest_internal(session, tokens, n_tokens, update_sampler);
    }

    size_t split_idx = n_tokens - 1;

    // 1. Process all but last
    if (split_idx > 0) {
        if (!session_ingest_internal(session, tokens, split_idx, update_sampler)) return false;
    }

    // 2. Snapshot state before the last token
    if (session->ctx) {
        size_t size = lfg_state_get_size(session->ctx);
        lfg_buf_u8_ensure(&session->healing_state_buffer, size);
        session->healing_state_buffer.size = size;
        lfg_state_get_data(session->ctx, session->healing_state_buffer.data, size);
        session->healing_n_past = session->n_past;
    }
    if (session->config.structured_checkpointing) {
        if (session->healing_sampler_snapshot) {
            lfg_sampler_free(session->healing_sampler_snapshot);
        }
        session->healing_sampler_snapshot = session_snapshot_sampler(session);
    } else {
        if (session->healing_sampler_snapshot) {
            lfg_sampler_free(session->healing_sampler_snapshot);
            session->healing_sampler_snapshot = nullptr;
        }
    }

    // 3. Process last token
    return session_ingest_internal(session, tokens + split_idx, 1, update_sampler);
}

// Rank registered tools against a pre-computed query embedding and build the
// "List of tools: [...]" JSON block into session->tool_json_buf.
// query_embd must be L2-normalized, length = session->tool_n_embd.
// Returns the text length written, or 0 if no tools / ranking failed.
static int32_t session_rank_and_format_tools(lfg_session *session, const float *query_embd) {
    if (!session || session->tool_count <= 0 || session->tool_top_k <= 0) return 0;

    int32_t n_embd = session->tool_n_embd;

    // Cosine similarity (vectors are L2-normalized, so dot product = cosine)
    for (int32_t i = 0; i < session->tool_count; ++i) {
        float dot = 0.0f;
        const float *te = session->tool_entries[i].embedding;
        for (int j = 0; j < n_embd; ++j) dot += query_embd[j] * te[j];
        session->tool_scores[i] = dot;
        session->tool_score_indices[i] = i;
    }

    // Sort indices by descending score (insertion sort — tool_count is small)
    for (int32_t i = 1; i < session->tool_count; ++i) {
        int32_t key_idx = session->tool_score_indices[i];
        float key_score = session->tool_scores[key_idx];
        int32_t j = i - 1;
        while (j >= 0 && session->tool_scores[session->tool_score_indices[j]] < key_score) {
            session->tool_score_indices[j + 1] = session->tool_score_indices[j];
            j--;
        }
        session->tool_score_indices[j + 1] = key_idx;
    }

    // Log ranking
    for (int32_t i = 0; i < session->tool_count; ++i) {
        int32_t idx = session->tool_score_indices[i];
        LFG_LOG_DEBUG("tool_rank[%d]: %s score=%.6f",
                      i, session->tool_entries[idx].name,
                      session->tool_scores[idx]);
    }

    // Gate injection based on score mode
    if (session->tool_score_mode != LFG_TOOL_SCORE_OFF) {
        float top_score = session->tool_scores[session->tool_score_indices[0]];
        bool skip = false;

        if (session->tool_score_mode == LFG_TOOL_SCORE_FIXED) {
            skip = top_score < session->tool_min_score;
        } else { // AUTO
            float threshold = session->tool_min_score > 0.0f ? session->tool_min_score : 0.1f;
            float sum = 0.0f;
            for (int32_t i = 0; i < session->tool_count; ++i) {
                sum += session->tool_scores[i];
            }
            float mean = sum / session->tool_count;
            skip = (top_score - mean) < threshold;
        }

        if (skip) {
            LFG_LOG_DEBUG("tool_inject: skipped (mode=%d top=%.4f min=%.4f)",
                          session->tool_score_mode, top_score, session->tool_min_score);
            return 0;
        }
    }

    // Build "List of tools: [tool1, tool2, ...]" into pre-allocated buffer
    int32_t k = session->tool_top_k;
    if (k > session->tool_count) k = session->tool_count;
    int32_t len = 0;

    const char *header = "List of tools: [";
    int32_t header_len = 16;
    std::memcpy(session->tool_json_buf + len, header, header_len);
    len += header_len;

    for (int32_t i = 0; i < k; ++i) {
        if (i > 0) { session->tool_json_buf[len++] = ','; session->tool_json_buf[len++] = ' '; }
        auto &tool = session->tool_entries[session->tool_score_indices[i]];
        std::memcpy(session->tool_json_buf + len, tool.json_text, tool.json_text_len);
        len += tool.json_text_len;
    }

    session->tool_json_buf[len++] = ']';
    session->tool_json_buf[len++] = '\n';
    session->tool_json_buf[len] = '\0';

    LFG_LOG_DEBUG("tool_inject: top_k=%d len=%d", k, len);

    return len;
}

LFG_API bool lfg_session_decode(lfg_session * session) {
    // Placeholder no-op: decode is handled by ingest/generate paths.
    // Return true even on null for graceful failure semantics.
    (void)session;
    return true;
}

LFG_API lfg_token lfg_session_sample(lfg_session * session) {
    if (!session || !session->ctx || !session->sampler) return 0;

    // Max tokens enforcement
    if (session->config.max_tokens > 0 && session->generated_count >= session->config.max_tokens) {
        return lfg_vocab_eos(lfg_model_get_vocab(session->model));
    }

    lfg_token token = lfg_sampler_sample(session->sampler, session->ctx, -1);
    session_record_sampler_token(session, token, true);
    session->generated_count++;

    // Entropy + confidence monitoring — zero alloc, shared softmax computation
    if (session->entropy_active || session->confidence_active) {
        float *logits = lfg_get_logits(session->ctx);
        int n_vocab = lfg_vocab_n_tokens(lfg_model_get_vocab(session->model));

        // 3-pass softmax + entropy: O(n_vocab), trivial vs decode cost
        float max_l = logits[0];
        for (int i = 1; i < n_vocab; ++i) if (logits[i] > max_l) max_l = logits[i];
        float sum_exp = 0.0f;
        for (int i = 0; i < n_vocab; ++i) sum_exp += expf(logits[i] - max_l);
        float entropy = 0.0f, inv = 1.0f / sum_exp, token_prob = 0.0f;
        for (int i = 0; i < n_vocab; ++i) {
            float p = expf(logits[i] - max_l) * inv;
            if (p > 0.0f) entropy -= p * logf(p);
            if (i == token) token_prob = p;
        }
        float max_ent = logf((float)n_vocab);
        float norm = max_ent > 0 ? entropy / max_ent : 0;
        session->entropy_last = entropy;
        session->entropy_last_norm = norm;

        // --- Entropy monitor (high entropy → retrieve) ---
        if (session->entropy_active) {
            session->entropy_tokens_since++;

            // Update running stats for AUTO mode
            if (session->entropy_gate_mode == LFG_ENTROPY_GATE_AUTO) {
                session->entropy_running_sum += norm;
                session->entropy_running_count++;
            }

            // Compute effective threshold
            bool entropy_fires = false;
            if (session->entropy_gate_mode == LFG_ENTROPY_GATE_FIXED) {
                entropy_fires = norm >= session->entropy_threshold;
            } else if (session->entropy_gate_mode == LFG_ENTROPY_GATE_AUTO &&
                       session->entropy_running_count >= 2) {
                float running_mean = session->entropy_running_sum / session->entropy_running_count;
                float gap = session->entropy_threshold > 0.0f ? session->entropy_threshold : 0.15f;
                entropy_fires = norm >= (running_mean + gap);
            }

            if (entropy_fires &&
                session->entropy_tokens_since >= session->entropy_cooldown) {
                session->entropy_tokens_since = 0;

                int32_t wi = session->entropy_write_idx;
                int slot = wi % session->entropy_ring_cap;

                // Save snap (zero alloc — write into pre-alloc'd array)
                lfg_entropy_snap *snap = &session->entropy_snaps[slot];
                snap->n_past = session->n_past;
                snap->token_count = session->token_history.size;
                snap->sampler_count = session->sampler_history.size;
                snap->generated_count = session->generated_count;
                snap->reasoning_token_count = session->reasoning_token_count;
                snap->id = session->entropy_next_id++;
                snap->valid = true;

                // Compute embedding into slot's pre-alloc'd buffer
                float *slot_embd = session->entropy_embd_pool + (slot * session->entropy_n_embd);
                bool got_embd = compute_query_embedding(session, slot_embd);

                // Write event into ring slot (zero alloc)
                lfg_entropy_ring_slot *rs = &session->entropy_slots[slot];
                rs->event.entropy       = entropy;
                rs->event.normalized    = norm;
                rs->event.top_logprob   = token_prob > 0 ? logf(token_prob) : -INFINITY;
                rs->event.token         = token;
                rs->event.n_past        = session->n_past;
                rs->event.checkpoint_id = snap->id;
                rs->event.n_embd        = got_embd ? session->entropy_n_embd : 0;
                rs->embedding           = got_embd ? slot_embd : nullptr;

                // Publish — atomic store (release semantics)
                __atomic_store_n(&session->entropy_write_idx, wi + 1, __ATOMIC_RELEASE);
            }
        }

        // --- Confidence monitor (low entropy → store) ---
        // Hot path is O(1): only accumulate run stats + write event metadata.
        // Embedding is computed lazily in pop(), not here.
        if (session->confidence_active) {
            bool conf_skip = !session->confidence_include_reasoning && session->in_reasoning;

            // Update running stats for AUTO mode (skip reasoning if configured)
            if (session->confidence_gate_mode == LFG_CONFIDENCE_GATE_AUTO && !conf_skip) {
                session->confidence_running_sum += norm;
                session->confidence_running_count++;
            }

            // Compute effective threshold
            bool is_confident = false;
            if (!conf_skip) {
                if (session->confidence_gate_mode == LFG_CONFIDENCE_GATE_FIXED) {
                    is_confident = norm <= session->confidence_threshold;
                } else if (session->confidence_gate_mode == LFG_CONFIDENCE_GATE_AUTO &&
                           session->confidence_running_count >= 5) {
                    float running_mean = session->confidence_running_sum / session->confidence_running_count;
                    float gap = session->confidence_threshold > 0.0f ? session->confidence_threshold : 0.10f;
                    is_confident = norm <= (running_mean - gap);
                }
            }

            if (is_confident) {
                // Extend run
                if (session->confidence_run_count == 0) {
                    session->confidence_run_start_pos = session->n_past;
                    session->confidence_run_min_entropy = norm;
                }
                session->confidence_run_count++;
                session->confidence_run_entropy_sum += norm;
                if (norm < session->confidence_run_min_entropy) {
                    session->confidence_run_min_entropy = norm;
                }
            } else {
                // Run broken — emit event if long enough
                if (session->confidence_run_count >= session->confidence_min_span) {
                    int32_t wi = session->confidence_write_idx;
                    int slot = wi % session->confidence_ring_cap;

                    lfg_confidence_ring_slot *rs = &session->confidence_slots[slot];
                    rs->event.mean_entropy = session->confidence_run_entropy_sum / session->confidence_run_count;
                    rs->event.min_entropy  = session->confidence_run_min_entropy;
                    rs->event.span_length  = session->confidence_run_count;
                    rs->event.start_pos    = session->confidence_run_start_pos;
                    rs->event.end_pos      = session->n_past;
                    rs->event.n_embd       = session->confidence_n_embd;
                    rs->embedding          = nullptr;  // Lazy — computed in pop()

                    __atomic_store_n(&session->confidence_write_idx, wi + 1, __ATOMIC_RELEASE);
                }
                // Reset run
                session->confidence_run_count = 0;
                session->confidence_run_entropy_sum = 0.0f;
                session->confidence_run_min_entropy = 1.0f;
            }
        }
    }

    // Stop sequence matching: check if tail of token_history + new token matches
    if (session->stop_count > 0) {
        // The new token is already in sampler_history but not token_history yet.
        // We check against token_history (which contains prior tokens) + the new token.
        for (size_t s = 0; s < session->stop_count; ++s) {
            size_t slen = session->stop_lengths[s];
            const lfg_token *seq = session->stop_flat + session->stop_offsets[s];

            // Check if the last token matches the last element of this stop sequence
            if (seq[slen - 1] != token) continue;

            if (slen == 1) {
                session->last_stop_len = 1;
                return lfg_vocab_eos(lfg_model_get_vocab(session->model));
            }

            // Check preceding tokens in history
            size_t hist_len = session->token_history.size;
            if (hist_len < slen - 1) continue;

            const lfg_token *tail = session->token_history.data + hist_len - (slen - 1);
            if (std::memcmp(tail, seq, (slen - 1) * sizeof(lfg_token)) == 0) {
                session->last_stop_len = (int32_t)slen;
                return lfg_vocab_eos(lfg_model_get_vocab(session->model));
            }
        }
    }

    return token;
}

LFG_API bool lfg_session_heal_last_token(lfg_session * session) {
    if (!session) return false;
    if (session->token_history.size < 2) return false;

    lfg_token t_last = session->token_history.data[session->token_history.size - 1];

    // Skip healing for reasoning boundary tokens — they are structural markers
    // whose bytes conflict with the grammar constraint (e.g. "</think>" vs JSON).
    // The sampler may have already transitioned (reasoning gate → ACTIVE) before
    // the snapshot was taken, so restoring and re-sampling with a prefix constraint
    // would conflict with the now-active grammar.
    if (session->reasoning_start_count > 0) {
        for (size_t i = 0; i < session->reasoning_start_count; i++) {
            if (t_last == session->reasoning_start_tokens[i]) return false;
        }
    }
    if (session->reasoning_end_count > 0) {
        for (size_t i = 0; i < session->reasoning_end_count; i++) {
            if (t_last == session->reasoning_end_tokens[i]) return false;
        }
    }

    int target_n_past = session->n_past - 1;

    if (session->config.enable_healing &&
        session->healing_state_buffer.size > 0 &&
        session->healing_n_past == target_n_past) {

        // Fast Path: Restore from snapshot
        spdlog::debug("HealLastToken: Fast Path");
        lfg_state_set_data(session->ctx, session->healing_state_buffer.data,
                          session->healing_state_buffer.size);
        session->n_past = session->healing_n_past;

        session_truncate_history(session, session->token_history.size - 1);

        // Restore sampler state
        if (session->sampler) {
            bool restored = false;
            if (session->config.structured_checkpointing && session->healing_sampler_snapshot) {
                restored = session_set_sampler_from_snapshot(session, session->healing_sampler_snapshot);
            }
            if (!restored) {
                bool has_grammar = session->grammar_str && session->grammar_str[0] != '\0';
                session_rebuild_sampler_from_history(session, has_grammar);
            }
        }

    } else {
        // Slow Fallback Path
        const size_t history_size = session->token_history.size;
        lfg_token t_prev = session->token_history.data[history_size - 2];

        session_truncate_history(session, history_size - 1);
        session_truncate_history(session, history_size - 2);

        int pos_prev = session->n_past - 2;
        lfg_memory_t mem = lfg_get_memory(session->ctx);

        if (!lfg_memory_seq_rm(mem, 0, pos_prev, -1)) {
            // Partial rewind failed, full reset
            lfg_memory_clear(mem, true);
            if (session->sampler) lfg_sampler_reset(session->sampler);
            session->n_past = 0;

            if (session->token_history.size > 0) {
                // Replay history
                int32_t n = (int32_t)session->token_history.size;

                // Ensure pos_buf is large enough
                if ((size_t)n > session->pos_buf_cap) {
                    session->pos_buf = (lfg_pos *)realloc(session->pos_buf, (size_t)n * sizeof(lfg_pos));
                    session->pos_buf_cap = (size_t)n;
                }
                for (int i = 0; i < n; ++i) session->pos_buf[i] = i;

                lfg_batch batch = {};
                batch.n_tokens = n;
                batch.token = session->token_history.data;
                batch.pos = session->pos_buf;

                if (lfg_decode(session->ctx, batch) != 0) return false;
                session->n_past = n;
            }
        } else {
            session->n_past = pos_prev;
        }

        if (session->sampler) {
            bool has_grammar = session->grammar_str && session->grammar_str[0] != '\0';
            session_rebuild_sampler_from_history(session, has_grammar);
        }

        // Re-ingest t_prev
        if (!session_ingest_internal(session, &t_prev, 1, true)) {
            return false;
        }
    }

    // Common Path: Sample new token with prefix constraint
    char buf[256];
    const lfg_vocab *vocab = lfg_model_get_vocab(session->model);
    int n = lfg_token_to_piece(vocab, t_last, buf, sizeof(buf), 0, false);
    if (n < 0) n = 0;
    buf[n] = '\0';

    if (session->prefix_sampler) {
        lfg_sampler_prefix_set(session->prefix_sampler, buf);
    }

    lfg_token t_new = lfg_sampler_sample(session->sampler, session->ctx, -1);
    session_record_sampler_token(session, t_new, true);

    if (session->prefix_sampler) {
        lfg_sampler_prefix_set(session->prefix_sampler, "");
    }

    // Ingest new token
    session_ingest_internal(session, &t_new, 1, false);

    return (t_new != t_last);
}

// ---------------------------------------------------------------------------
// Logits access
// ---------------------------------------------------------------------------

LFG_API int32_t lfg_session_get_logits(lfg_session * session, float * out, int32_t max_out) {
    if (!session || !session->ctx) return -1;

    auto *logits = lfg_get_logits(session->ctx);
    const auto *vocab_obj = lfg_model_get_vocab(session->model);
    int n_vocab = lfg_vocab_n_tokens(vocab_obj);

    if (!out || max_out <= 0) {
        return (int32_t)n_vocab;
    }
    const int32_t n = std::min(max_out, (int32_t)n_vocab);
    if (n > 0) {
        std::memcpy(out, logits, sizeof(float) * n);
    }
    return n;
}

LFG_API int32_t lfg_session_get_vocab_size(lfg_session * session) {
    if (!session || !session->model) return -1;
    const auto *vocab = lfg_model_get_vocab(session->model);
    if (!vocab) return -1;
    return lfg_vocab_n_tokens(vocab);
}

// ---------------------------------------------------------------------------
// Checkpointing
// ---------------------------------------------------------------------------

LFG_API lfg_checkpoint * lfg_session_create_checkpoint(lfg_session * session) {
    if (!session) return nullptr;

    auto *ck = (lfg_checkpoint *)calloc(1, sizeof(lfg_checkpoint));
    ck->n_past = session->n_past;
    ck->token_count = session->token_history.size;
    ck->sampler_count = session->sampler_history.size;
    ck->generated_count = session->generated_count;
    ck->reasoning_token_count = session->reasoning_token_count;
    ck->grammar_str = session->grammar_str ? strdup(session->grammar_str) : nullptr;
    ck->grammar_root = session->grammar_root ? strdup(session->grammar_root) : nullptr;

    if (session->config.structured_checkpointing) {
        ck->sampler_state = session_snapshot_sampler(session);
    } else {
        ck->sampler_state = nullptr;
    }

    if (session->ctx) {
        size_t size = lfg_state_get_size(session->ctx);
        ck->state_data = (uint8_t *)malloc(size);
        ck->state_data_size = size;
        lfg_state_get_data(session->ctx, ck->state_data, size);
    } else {
        ck->state_data = nullptr;
        ck->state_data_size = 0;
    }

    return ck;
}

LFG_API lfg_checkpoint_restore_options lfg_checkpoint_restore_default_options(void) {
    lfg_checkpoint_restore_options opts{};
    opts.restore_sampler_state = true;
    opts.restore_grammar = true;
    return opts;
}

LFG_API bool lfg_session_restore_checkpoint_ex(lfg_session * session,
                                                     const lfg_checkpoint * checkpoint,
                                                     const lfg_checkpoint_restore_options * options) {
    if (!session || !checkpoint) return false;

    lfg_checkpoint_restore_options opts = options ? *options : lfg_checkpoint_restore_default_options();

    if (checkpoint->n_past > session->n_past ||
        checkpoint->token_count > session->token_history.size ||
        checkpoint->token_count > session->sampler_offsets.size ||
        checkpoint->sampler_count > session->sampler_history.size) {
        spdlog::error("Invalid checkpoint: moving forward in time not allowed (or history mismatch).");
        return false;
    }

    if (checkpoint->state_data && checkpoint->state_data_size > 0 && session->ctx) {
        if (lfg_state_set_data(session->ctx, checkpoint->state_data, checkpoint->state_data_size) != checkpoint->state_data_size) {
            spdlog::error("Failed to restore liquid state from checkpoint.");
            return false;
        }
    } else {
        lfg_memory_t mem = lfg_get_memory(session->ctx);
        if (!lfg_memory_seq_rm(mem, 0, checkpoint->n_past, -1)) {
            spdlog::error("Failed to truncate KV cache.");
            return false;
        }
    }

    // Truncate History
    session->token_history.size = checkpoint->token_count;
    session->sampler_offsets.size = checkpoint->token_count;
    session->sampler_history.size = checkpoint->sampler_count;
    session->pending_sampler_accepts = 0;
    session->n_past = checkpoint->n_past;
    session->generated_count = checkpoint->generated_count;
    session->reasoning_token_count = checkpoint->reasoning_token_count;

    // Reset and Re-prime Sampler
    if (session->sampler) {
        bool grammar_changed = false;
        if (opts.restore_grammar) {
            // Compare grammar strings
            const char *ck_grammar = checkpoint->grammar_str ? checkpoint->grammar_str : "";
            const char *s_grammar = session->grammar_str ? session->grammar_str : "";
            const char *ck_root = checkpoint->grammar_root ? checkpoint->grammar_root : "";
            const char *s_root = session->grammar_root ? session->grammar_root : "";
            grammar_changed = (std::strcmp(ck_grammar, s_grammar) != 0) ||
                             (std::strcmp(ck_root, s_root) != 0);

            free(session->grammar_str);
            free(session->grammar_root);
            session->grammar_str = checkpoint->grammar_str ? strdup(checkpoint->grammar_str) : nullptr;
            session->grammar_root = checkpoint->grammar_root ? strdup(checkpoint->grammar_root) : nullptr;
        }

        bool use_sampler_state = opts.restore_sampler_state && checkpoint->sampler_state;
        if (use_sampler_state && !opts.restore_grammar) {
            const char *ck_grammar = checkpoint->grammar_str ? checkpoint->grammar_str : "";
            const char *s_grammar = session->grammar_str ? session->grammar_str : "";
            const char *ck_root = checkpoint->grammar_root ? checkpoint->grammar_root : "";
            const char *s_root = session->grammar_root ? session->grammar_root : "";
            if (std::strcmp(ck_grammar, s_grammar) != 0 || std::strcmp(ck_root, s_root) != 0) {
                use_sampler_state = false;
            }
        }

        if (use_sampler_state) {
            if (!session_set_sampler_from_snapshot(session, checkpoint->sampler_state)) {
                use_sampler_state = false;
            }
        }

        if (!use_sampler_state) {
            if (grammar_changed) {
                session_rebuild_sampler(session);
            }
            bool has_grammar = session->grammar_str && session->grammar_str[0] != '\0';
            session_rebuild_sampler_from_history(session, has_grammar);
        }
    }

    return true;
}

LFG_API bool lfg_session_restore_checkpoint(lfg_session * session, const lfg_checkpoint * checkpoint) {
    return lfg_session_restore_checkpoint_ex(session, checkpoint, nullptr);
}

LFG_API void lfg_checkpoint_free(lfg_checkpoint * checkpoint) {
    if (!checkpoint) return;
    free(checkpoint->state_data);
    free(checkpoint->grammar_str);
    free(checkpoint->grammar_root);
    if (checkpoint->sampler_state) lfg_sampler_free(checkpoint->sampler_state);
    free(checkpoint);
}

// ---------------------------------------------------------------------------
// Tool Ranking API
// ---------------------------------------------------------------------------

// Free tool entry array contents (not the array itself).
static void session_free_tool_entries(lfg_session *s) {
    for (int32_t i = 0; i < s->tool_count; ++i) {
        free(s->tool_entries[i].name);
        free(s->tool_entries[i].json_text);  // malloc'd by tool_format_json()
        free(s->tool_entries[i].embedding);
    }
    free(s->tool_entries);
    s->tool_entries = nullptr;
    s->tool_count = 0;
}

// Look up hash in the linear cache. Returns pointer to embedding or nullptr.
static const float * tool_cache_lookup(const lfg_session *s, uint64_t hash) {
    for (int32_t i = 0; i < s->tool_cache_count; ++i) {
        if (s->tool_cache_hashes[i] == hash)
            return s->tool_cache_embeds + (size_t)i * s->tool_n_embd;
    }
    return nullptr;
}

// Append to linear cache, growing if needed.
static void tool_cache_insert(lfg_session *s, uint64_t hash, const float *embd) {
    if (s->tool_cache_count >= s->tool_cache_cap) {
        int32_t new_cap = s->tool_cache_cap ? s->tool_cache_cap * 2 : 16;
        s->tool_cache_hashes = (uint64_t *)realloc(s->tool_cache_hashes, new_cap * sizeof(uint64_t));
        s->tool_cache_embeds = (float *)realloc(s->tool_cache_embeds, (size_t)new_cap * s->tool_n_embd * sizeof(float));
        s->tool_cache_cap = new_cap;
    }
    s->tool_cache_hashes[s->tool_cache_count] = hash;
    std::memcpy(s->tool_cache_embeds + (size_t)s->tool_cache_count * s->tool_n_embd,
                embd, s->tool_n_embd * sizeof(float));
    s->tool_cache_count++;
}

LFG_API int32_t lfg_session_register_tools(lfg_session * session,
                                           const lfg_tool_desc * tools, int32_t n_tools,
                                           int32_t top_k) {
    if (!session || !tools || n_tools <= 0) {
        if (session) {
            lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT,
                "%s: invalid arguments", __func__);
        }
        return -1;
    }

    try {
        // Create tool context if needed (embeddings + mean pooling for semantic similarity)
        if (!session_ensure_embed_ctx(session)) {
            lfg_set_last_error(LFG_ERROR_INTERNAL,
                "%s: failed to create tool context", __func__);
            return -1;
        }

        const lfg_vocab *vocab = lfg_model_get_vocab(session->model);
        int32_t n_embd = lfg_model_n_embd_out(session->model);

        // Free previous entries (but keep cache)
        session_free_tool_entries(session);

        session->tool_n_embd = n_embd;
        session->tool_top_k = top_k;
        session->tools_injected = false;

        // Allocate tool entries
        session->tool_entries = (lfg_session::lfg_tool_entry *)calloc(n_tools, sizeof(lfg_session::lfg_tool_entry));
        session->tool_count = n_tools;

        // Scratch for tokenization during registration (freed at end)
        int32_t tok_scratch_cap = 512;
        lfg_token *tok_scratch = (lfg_token *)malloc(tok_scratch_cap * sizeof(lfg_token));

        // Total JSON size for pre-allocating decode-time buffer
        int32_t total_json_bytes = 0;

        for (int32_t i = 0; i < n_tools; ++i) {
            auto &entry = session->tool_entries[i];
            int32_t json_len = 0;
            entry.json_text = tool_format_json(&tools[i], &json_len);
            entry.json_text_len = json_len;
            entry.name = strdup(tools[i].name ? tools[i].name : "");
            entry.fn = tools[i].fn;
            entry.fn_user_data = tools[i].fn_user_data;
            total_json_bytes += json_len;

            // Semantic ranking is based on name + description only.
            std::string rank_text = tool_format_rank_text(&tools[i]);
            const char *rank_ptr = rank_text.c_str();
            int32_t rank_len = (int32_t)rank_text.size();
            uint64_t hash = fnv1a_hash(rank_ptr, (size_t)rank_len);

            // Token cost
            if (json_len + 16 > tok_scratch_cap) {
                tok_scratch_cap = json_len + 16;
                tok_scratch = (lfg_token *)realloc(tok_scratch, tok_scratch_cap * sizeof(lfg_token));
            }
            int32_t n_tok = lfg_tokenize(vocab, entry.json_text, json_len,
                                         tok_scratch, tok_scratch_cap, false, false);
            entry.token_cost = (n_tok > 0) ? n_tok : 0;

            // Check embedding cache
            const float *cached = tool_cache_lookup(session, hash);
            if (cached) {
                entry.embedding = (float *)malloc(n_embd * sizeof(float));
                std::memcpy(entry.embedding, cached, n_embd * sizeof(float));
            } else {
                // Compute embedding via tool_ctx
                lfg_memory_clear(lfg_get_memory(session->tool_ctx), true);

                if (rank_len + 16 > tok_scratch_cap) {
                    tok_scratch_cap = rank_len + 16;
                    tok_scratch = (lfg_token *)realloc(tok_scratch, tok_scratch_cap * sizeof(lfg_token));
                }
                int32_t n_emb_tok = lfg_tokenize(vocab, rank_ptr, rank_len,
                                                  tok_scratch, tok_scratch_cap, true, false);

                entry.embedding = (float *)calloc(n_embd, sizeof(float));

                if (n_emb_tok > 0) {
                    lfg_batch batch = lfg_batch_get_one(tok_scratch, n_emb_tok);
                    if (lfg_decode(session->tool_ctx, batch) == 0) {
                        // Use mean-pooled embedding via seq_id 0
                        float *embd_ptr = lfg_get_embeddings_seq(session->tool_ctx, 0);
                        if (!embd_ptr) {
                            // Fallback to last-token embedding
                            embd_ptr = lfg_get_embeddings_ith(session->tool_ctx, -1);
                        }
                        if (embd_ptr) {
                            std::memcpy(entry.embedding, embd_ptr, n_embd * sizeof(float));
                            l2_normalize(entry.embedding, n_embd);
                        }
                    }
                }

                tool_cache_insert(session, hash, entry.embedding);
            }
        }

        free(tok_scratch);

        // Pre-allocate decode-time scratch buffers
        // JSON buffer: "List of tools: [" + all tool JSON + "]\n" + NUL
        int32_t json_buf_cap = total_json_bytes + 64;  // extra for header, commas, brackets
        session->tool_json_buf = (char *)realloc(session->tool_json_buf, json_buf_cap);
        session->tool_json_buf_cap = json_buf_cap;

        // Token buffer: worst case ~ json_buf_cap tokens (1 byte = 1 token)
        int32_t token_buf_cap = json_buf_cap + 32;
        session->tool_token_buf = (lfg_token *)realloc(session->tool_token_buf, token_buf_cap * sizeof(lfg_token));
        session->tool_token_buf_cap = token_buf_cap;

        // Query embedding scratch
        session->tool_query_embd = (float *)realloc(session->tool_query_embd, n_embd * sizeof(float));

        // Score arrays
        session->tool_score_indices = (int32_t *)realloc(session->tool_score_indices, n_tools * sizeof(int32_t));
        session->tool_scores = (float *)realloc(session->tool_scores, n_tools * sizeof(float));

        return n_tools;
    } catch (const std::exception &err) {
        lfg_set_last_error(LFG_ERROR_INTERNAL,
            "%s: failed: %s", __func__, err.what());
        return -1;
    }
}

LFG_API void lfg_session_clear_tools(lfg_session * session) {
    if (!session) return;
    session_free_tool_entries(session);
    session->tool_top_k = 0;
    session->tools_injected = false;
    // Only free tool_ctx if entropy/confidence/surprise monitor isn't using it
    if (session->tool_ctx && !session->entropy_active && !session->confidence_active && !session->surprise_active) {
        lfg_free(session->tool_ctx);
        session->tool_ctx = nullptr;
    }
    free(session->tool_cache_hashes);
    free(session->tool_cache_embeds);
    session->tool_cache_hashes = nullptr;
    session->tool_cache_embeds = nullptr;
    session->tool_cache_count = 0;
    session->tool_cache_cap = 0;
    free(session->tool_query_embd);    session->tool_query_embd = nullptr;
    free(session->tool_score_indices); session->tool_score_indices = nullptr;
    free(session->tool_scores);        session->tool_scores = nullptr;
    free(session->tool_json_buf);       session->tool_json_buf = nullptr;
    session->tool_json_buf_cap = 0;
    free(session->tool_token_buf);     session->tool_token_buf = nullptr;
    session->tool_token_buf_cap = 0;
    session->tool_n_embd = 0;
}

LFG_API int32_t lfg_session_rank_tools(lfg_session * session,
                                       const char * query, int32_t query_len,
                                       char * buf, int32_t buf_size) {
    if (!session || !query || query_len <= 0) return -1;
    if (session->tool_count <= 0 || session->tool_top_k <= 0) return -1;

    // Compute query embedding via the public embed API
    int32_t n_embd = session->tool_n_embd;
    int32_t got = lfg_session_embed(session, query, query_len,
                                     session->tool_query_embd, n_embd);
    if (got <= 0) return -1;

    // Rank and format into session->tool_json_buf
    int32_t tools_len = session_rank_and_format_tools(session, session->tool_query_embd);
    if (tools_len <= 0) return -1;

    // Return required size if buf is NULL or buf_size is 0
    if (!buf || buf_size <= 0) return tools_len;

    // Copy to user buffer (truncate if needed)
    int32_t copy_len = tools_len < buf_size ? tools_len : buf_size - 1;
    std::memcpy(buf, session->tool_json_buf, copy_len);
    buf[copy_len] = '\0';
    return copy_len;
}

// ---------------------------------------------------------------------------
// Entropy Monitor API
// ---------------------------------------------------------------------------

LFG_API lfg_entropy_monitor_config lfg_entropy_monitor_default_config(void) {
    lfg_entropy_monitor_config cfg{};
    cfg.threshold = 0.7f;
    cfg.cooldown_tokens = 16;
    cfg.ring_size = 4;
    cfg.gate_mode = LFG_ENTROPY_GATE_FIXED;
    return cfg;
}

LFG_API int32_t lfg_session_configure_entropy_monitor(
    lfg_session * session, const lfg_entropy_monitor_config * config) {
    if (!session) return 0;

    // Pass NULL to disable
    if (!config) {
        session->entropy_active = false;
        return 0;
    }

    if (!session_ensure_embed_ctx(session)) {
        lfg_set_last_error(LFG_ERROR_INTERNAL,
            "%s: failed to create embedding context", __func__);
        return 0;
    }

    int32_t cap = config->ring_size > 0 ? config->ring_size : 4;
    int32_t n_embd = lfg_model_n_embd_out(session->model);

    // Free previous allocations
    free(session->entropy_slots);
    free(session->entropy_snaps);
    free(session->entropy_embd_pool);

    // One-time allocation of ring buffer + embedding pool
    session->entropy_slots     = (lfg_entropy_ring_slot *)calloc(cap, sizeof(lfg_entropy_ring_slot));
    session->entropy_snaps     = (lfg_entropy_snap *)calloc(cap, sizeof(lfg_entropy_snap));
    session->entropy_embd_pool = (float *)calloc(cap * n_embd, sizeof(float));

    // Wire each slot's embedding pointer to pool
    for (int32_t i = 0; i < cap; ++i) {
        session->entropy_slots[i].embedding = session->entropy_embd_pool + (i * n_embd);
    }

    session->entropy_ring_cap     = cap;
    session->entropy_n_embd       = n_embd;
    session->entropy_write_idx    = 0;
    session->entropy_read_idx     = 0;
    session->entropy_next_id      = 0;
    session->entropy_threshold    = config->threshold;
    session->entropy_cooldown     = config->cooldown_tokens > 0 ? config->cooldown_tokens : 1;
    session->entropy_tokens_since = config->cooldown_tokens > 0 ? config->cooldown_tokens : 1; // allow first event immediately
    session->entropy_last         = -1.0f;
    session->entropy_last_norm    = -1.0f;
    session->entropy_gate_mode    = config->gate_mode;
    session->entropy_running_sum  = 0.0f;
    session->entropy_running_count = 0;
    session->entropy_active       = true;

    return n_embd;
}

LFG_API bool lfg_session_entropy_pop(lfg_session * session,
                                      lfg_entropy_event * event_out,
                                      float * embd_out, int32_t embd_cap) {
    if (!session || !session->entropy_slots) return false;

    int32_t wi = __atomic_load_n(&session->entropy_write_idx, __ATOMIC_ACQUIRE);
    if (session->entropy_read_idx >= wi) return false;

    int slot = session->entropy_read_idx % session->entropy_ring_cap;
    lfg_entropy_ring_slot *rs = &session->entropy_slots[slot];

    if (event_out) *event_out = rs->event;

    if (embd_out && rs->embedding && rs->event.n_embd > 0) {
        int32_t n = rs->event.n_embd < embd_cap ? rs->event.n_embd : embd_cap;
        std::memcpy(embd_out, rs->embedding, n * sizeof(float));
    }

    session->entropy_read_idx++;
    return true;
}

LFG_API int32_t lfg_session_entropy_pending(lfg_session * session) {
    if (!session || !session->entropy_slots) return 0;
    int32_t wi = __atomic_load_n(&session->entropy_write_idx, __ATOMIC_ACQUIRE);
    int32_t pending = wi - session->entropy_read_idx;
    return pending > 0 ? pending : 0;
}

LFG_API void lfg_session_entropy_flush(lfg_session * session) {
    if (!session) return;
    session->entropy_read_idx = __atomic_load_n(&session->entropy_write_idx, __ATOMIC_ACQUIRE);
}

LFG_API volatile int32_t * lfg_session_entropy_counter(lfg_session * session) {
    if (!session) return nullptr;
    return &session->entropy_write_idx;
}

LFG_API bool lfg_session_rewind(lfg_session * session, int32_t checkpoint_id) {
    if (!session || !session->entropy_snaps) return false;

    // Find snap by checkpoint_id
    lfg_entropy_snap *snap = nullptr;
    for (int32_t i = 0; i < session->entropy_ring_cap; ++i) {
        if (session->entropy_snaps[i].valid && session->entropy_snaps[i].id == checkpoint_id) {
            snap = &session->entropy_snaps[i];
            break;
        }
    }
    if (!snap) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT,
            "%s: checkpoint_id %d not found or expired", __func__, checkpoint_id);
        return false;
    }

    // Cannot rewind forward
    if (snap->n_past > session->n_past) {
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT,
            "%s: checkpoint n_past (%d) > current n_past (%d)", __func__, snap->n_past, session->n_past);
        return false;
    }

    // Restore session state (truncate histories first, before KV replay fallback)
    session->token_history.size = snap->token_count;
    session->sampler_offsets.size = snap->token_count;
    if (snap->sampler_count <= session->sampler_history.size) {
        session->sampler_history.size = snap->sampler_count;
    }
    session->pending_sampler_accepts = 0;
    session->generated_count = snap->generated_count;
    session->reasoning_token_count = snap->reasoning_token_count;
    session->n_past = snap->n_past;

    // Truncate KV cache
    lfg_memory_t mem = lfg_get_memory(session->ctx);
    bool kv_ok = lfg_memory_seq_rm(mem, 0, snap->n_past, -1);
    if (!kv_ok) {
        // Fallback: full clear + replay token history up to snap point
        lfg_memory_clear(mem, true);
        session->n_past = 0;

        if (snap->token_count > 0) {
            int32_t n = (int32_t)snap->token_count;
            if ((size_t)n > session->pos_buf_cap) {
                session->pos_buf = (lfg_pos *)realloc(session->pos_buf, (size_t)n * sizeof(lfg_pos));
                session->pos_buf_cap = (size_t)n;
            }
            for (int i = 0; i < n; ++i) session->pos_buf[i] = i;

            lfg_batch batch = {};
            batch.n_tokens = n;
            batch.token = session->token_history.data;
            batch.pos = session->pos_buf;
            if (lfg_decode(session->ctx, batch) != 0) {
                lfg_set_last_error(LFG_ERROR_INTERNAL, "%s: failed to replay KV cache", __func__);
                return false;
            }
            session->n_past = n;
        }
    }

    // Reset sampler from history
    if (session->sampler) {
        bool has_grammar = session->grammar_str && session->grammar_str[0] != '\0';
        session_rebuild_sampler_from_history(session, has_grammar);
    }

    // Invalidate the used snap
    snap->valid = false;

    // Reset entropy ring (events are stale after rewind)
    session->entropy_write_idx = 0;
    session->entropy_read_idx = 0;
    session->entropy_tokens_since = session->entropy_cooldown; // allow event on next uncertain token

    return true;
}

LFG_API float lfg_session_get_last_entropy(lfg_session * session) {
    if (!session) return -1.0f;
    return session->entropy_last_norm;
}

// ---------------------------------------------------------------------------
// Confidence Monitor API
// ---------------------------------------------------------------------------

LFG_API lfg_confidence_monitor_config lfg_confidence_monitor_default_config(void) {
    lfg_confidence_monitor_config cfg{};
    cfg.threshold = 0.3f;
    cfg.min_span = 5;
    cfg.ring_size = 4;
    cfg.gate_mode = LFG_CONFIDENCE_GATE_FIXED;
    return cfg;
}

LFG_API int32_t lfg_session_configure_confidence_monitor(
    lfg_session * session, const lfg_confidence_monitor_config * config) {
    if (!session) return 0;

    // Pass NULL to disable
    if (!config) {
        session->confidence_active = false;
        return 0;
    }

    if (!session_ensure_embed_ctx(session)) {
        lfg_set_last_error(LFG_ERROR_INTERNAL,
            "%s: failed to create embedding context", __func__);
        return 0;
    }

    int32_t cap = config->ring_size > 0 ? config->ring_size : 4;
    int32_t min_span = config->min_span > 0 ? config->min_span : 5;
    int32_t n_embd = lfg_model_n_embd_out(session->model);

    // Free previous allocations
    free(session->confidence_slots);
    free(session->confidence_embd_pool);
    free(session->confidence_text_buf);
    session->confidence_slots = nullptr;
    session->confidence_embd_pool = nullptr;
    session->confidence_text_buf = nullptr;
    session->confidence_text_cap = 0;

    // One-time allocation of ring buffer + embedding pool
    session->confidence_slots     = (lfg_confidence_ring_slot *)calloc(cap, sizeof(lfg_confidence_ring_slot));
    session->confidence_embd_pool = (float *)calloc(cap * n_embd, sizeof(float));

    // Wire each slot's embedding pointer to pool
    for (int32_t i = 0; i < cap; ++i) {
        session->confidence_slots[i].embedding = session->confidence_embd_pool + (i * n_embd);
    }

    session->confidence_ring_cap     = cap;
    session->confidence_n_embd       = n_embd;
    session->confidence_write_idx    = 0;
    session->confidence_read_idx     = 0;
    session->confidence_threshold    = config->threshold;
    session->confidence_min_span     = min_span;
    session->confidence_run_count    = 0;
    session->confidence_run_entropy_sum = 0.0f;
    session->confidence_run_min_entropy = 1.0f;
    session->confidence_run_start_pos   = 0;
    session->confidence_include_reasoning = config->include_reasoning;
    session->confidence_gate_mode    = config->gate_mode;
    session->confidence_running_sum  = 0.0f;
    session->confidence_running_count = 0;
    session->confidence_active       = true;

    // Span text buffer (realloc'd on demand at pop time)
    session->confidence_text_cap = 4096;
    session->confidence_text_buf = (char *)malloc(session->confidence_text_cap);

    return n_embd;
}

LFG_API bool lfg_session_confidence_pop(lfg_session * session,
                                         lfg_confidence_event * event_out,
                                         float * embd_out, int32_t embd_cap) {
    if (!session || !session->confidence_slots) return false;

    int32_t wi = __atomic_load_n(&session->confidence_write_idx, __ATOMIC_ACQUIRE);
    if (session->confidence_read_idx >= wi) return false;

    int slot = session->confidence_read_idx % session->confidence_ring_cap;
    lfg_confidence_ring_slot *rs = &session->confidence_slots[slot];

    // Lazy embedding: compute on pop if caller wants it and we haven't yet
    if (embd_out && embd_cap >= session->confidence_n_embd && !rs->embedding) {
        float *slot_embd = session->confidence_embd_pool + (slot * session->confidence_n_embd);
        bool got = compute_query_embedding(session, slot_embd);
        if (got) {
            rs->embedding = slot_embd;
        } else {
            rs->event.n_embd = 0;
        }
    }

    // Detokenize span text from token_history
    int32_t span_n = rs->event.end_pos - rs->event.start_pos;
    if (span_n > 0 && session->confidence_text_buf &&
        rs->event.start_pos >= 0 &&
        rs->event.end_pos <= (int32_t)session->token_history.size) {
        const lfg_vocab *vocab = lfg_model_get_vocab(session->model);
        lfg_token *span_tokens = session->token_history.data + rs->event.start_pos;

        int32_t text_len = lfg_detokenize(vocab, span_tokens, span_n,
                                           session->confidence_text_buf,
                                           session->confidence_text_cap,
                                           false, false);
        if (text_len < 0) {
            // Buffer too small — grow and retry
            session->confidence_text_cap = -text_len + 1;
            session->confidence_text_buf = (char *)realloc(
                session->confidence_text_buf, session->confidence_text_cap);
            text_len = lfg_detokenize(vocab, span_tokens, span_n,
                                       session->confidence_text_buf,
                                       session->confidence_text_cap,
                                       false, false);
        }
        if (text_len >= 0) {
            session->confidence_text_buf[text_len] = '\0';
            rs->event.span_text = session->confidence_text_buf;
            rs->event.span_text_len = text_len;
        }
    }

    if (event_out) *event_out = rs->event;

    if (embd_out && rs->embedding && rs->event.n_embd > 0) {
        int32_t n = rs->event.n_embd < embd_cap ? rs->event.n_embd : embd_cap;
        std::memcpy(embd_out, rs->embedding, n * sizeof(float));
    }

    session->confidence_read_idx++;
    return true;
}

LFG_API int32_t lfg_session_confidence_pending(lfg_session * session) {
    if (!session || !session->confidence_slots) return 0;
    int32_t wi = __atomic_load_n(&session->confidence_write_idx, __ATOMIC_ACQUIRE);
    int32_t pending = wi - session->confidence_read_idx;
    return pending > 0 ? pending : 0;
}

LFG_API void lfg_session_confidence_flush(lfg_session * session) {
    if (!session) return;
    session->confidence_read_idx = __atomic_load_n(&session->confidence_write_idx, __ATOMIC_ACQUIRE);
}

LFG_API volatile int32_t * lfg_session_confidence_counter(lfg_session * session) {
    if (!session) return nullptr;
    return &session->confidence_write_idx;
}

// ---------------------------------------------------------------------------
// Surprise Monitor API
// ---------------------------------------------------------------------------

LFG_API lfg_surprise_monitor_config lfg_surprise_monitor_default_config(void) {
    lfg_surprise_monitor_config cfg{};
    cfg.threshold = 0.5f;
    cfg.ring_size = 4;
    cfg.gate_mode = LFG_SURPRISE_GATE_FIXED;
    return cfg;
}

LFG_API int32_t lfg_session_configure_surprise_monitor(
    lfg_session * session, const lfg_surprise_monitor_config * config) {
    if (!session) return 0;

    // Pass NULL to disable
    if (!config) {
        session->surprise_active = false;
        return 0;
    }

    if (!session_ensure_embed_ctx(session)) {
        lfg_set_last_error(LFG_ERROR_INTERNAL,
            "%s: failed to create embedding context", __func__);
        return 0;
    }

    int32_t n_embd = lfg_model_n_embd_out(session->model);
    int32_t cap = config->ring_size > 0 ? config->ring_size : 4;

    // Free previous allocations
    free(session->surprise_slots);
    free(session->surprise_embd_pool);

    // Ring buffer + embedding pool
    session->surprise_slots = (lfg_surprise_ring_slot *)calloc(cap, sizeof(lfg_surprise_ring_slot));
    session->surprise_embd_pool = (float *)calloc(cap * n_embd, sizeof(float));

    for (int32_t i = 0; i < cap; ++i) {
        session->surprise_slots[i].embedding = session->surprise_embd_pool + (i * n_embd);
    }

    session->surprise_ring_cap     = cap;
    session->surprise_write_idx    = 0;
    session->surprise_read_idx     = 0;
    session->surprise_n_embd       = n_embd;
    session->surprise_threshold    = config->threshold;
    session->surprise_count        = 0;
    session->surprise_sum          = 0.0f;
    session->surprise_max          = 0.0f;
    session->surprise_n_evaluated  = 0;
    session->surprise_include_reasoning = config->include_reasoning;
    session->surprise_skip_tokens  = 0;
    session->surprise_gate_mode    = config->gate_mode;

    // Pre-allocate scratch for AUTO two-pass
    if (config->gate_mode == LFG_SURPRISE_GATE_AUTO && !session->surprise_per_token) {
        session->surprise_per_token_cap = 512;
        session->surprise_per_token = (float *)malloc(512 * sizeof(float));
    }

    session->surprise_active       = true;

    return n_embd;
}

LFG_API bool lfg_session_surprise_pop(lfg_session * session,
                                       lfg_surprise_event * event_out,
                                       float * embd_out, int32_t embd_cap) {
    if (!session || !session->surprise_active || !session->surprise_slots) return false;

    int32_t wi = __atomic_load_n(&session->surprise_write_idx, __ATOMIC_ACQUIRE);
    if (session->surprise_read_idx >= wi) return false;

    int slot = session->surprise_read_idx % session->surprise_ring_cap;
    lfg_surprise_ring_slot *rs = &session->surprise_slots[slot];

    if (event_out) *event_out = rs->event;

    if (embd_out && rs->embedding && rs->event.n_embd > 0) {
        int32_t n = rs->event.n_embd < embd_cap ? rs->event.n_embd : embd_cap;
        std::memcpy(embd_out, rs->embedding, n * sizeof(float));
    }

    session->surprise_read_idx++;
    return true;
}

LFG_API int32_t lfg_session_surprise_pending(lfg_session * session) {
    if (!session || !session->surprise_slots) return 0;
    int32_t wi = __atomic_load_n(&session->surprise_write_idx, __ATOMIC_ACQUIRE);
    int32_t pending = wi - session->surprise_read_idx;
    return pending > 0 ? pending : 0;
}

LFG_API void lfg_session_surprise_flush(lfg_session * session) {
    if (!session) return;
    session->surprise_read_idx = __atomic_load_n(&session->surprise_write_idx, __ATOMIC_ACQUIRE);
}

LFG_API volatile int32_t * lfg_session_surprise_counter(lfg_session * session) {
    if (!session) return nullptr;
    return &session->surprise_write_idx;
}

// ---------------------------------------------------------------------------
// Embedding API
// ---------------------------------------------------------------------------

LFG_API int32_t lfg_session_embed(lfg_session * session,
                                   const char * text, int32_t text_len,
                                   float * out, int32_t out_cap) {
    if (!session || !text || text_len <= 0 || !out || out_cap <= 0) return 0;

    if (!session_ensure_embed_ctx(session)) {
        lfg_set_last_error(LFG_ERROR_INTERNAL,
            "%s: failed to create embedding context", __func__);
        return 0;
    }

    int32_t n_embd = lfg_model_n_embd_out(session->model);
    if (out_cap < n_embd) return 0;

    const lfg_vocab *vocab = lfg_model_get_vocab(session->model);

    // Tokenize
    int32_t tok_cap = text_len + 16;
    lfg_token *toks = (lfg_token *)malloc(tok_cap * sizeof(lfg_token));
    int32_t n_tok = lfg_tokenize(vocab, text, text_len, toks, tok_cap, true, false);
    if (n_tok < 0) {
        tok_cap = -n_tok;
        toks = (lfg_token *)realloc(toks, tok_cap * sizeof(lfg_token));
        n_tok = lfg_tokenize(vocab, text, text_len, toks, tok_cap, true, false);
    }
    if (n_tok <= 0) { free(toks); return 0; }

    // Truncate to context capacity
    int32_t ctx_cap = (int32_t)lfg_n_ctx(session->tool_ctx);
    lfg_token *tok_ptr = toks;
    if (n_tok > ctx_cap) {
        tok_ptr = toks + (n_tok - ctx_cap);
        n_tok = ctx_cap;
    }

    // Forward pass
    lfg_memory_clear(lfg_get_memory(session->tool_ctx), true);
    lfg_batch batch = lfg_batch_get_one(tok_ptr, n_tok);
    if (lfg_decode(session->tool_ctx, batch) != 0) { free(toks); return 0; }

    float *embd_ptr = lfg_get_embeddings_seq(session->tool_ctx, 0);
    if (!embd_ptr) embd_ptr = lfg_get_embeddings_ith(session->tool_ctx, -1);
    if (!embd_ptr) { free(toks); return 0; }

    // Copy + L2 normalize
    std::memcpy(out, embd_ptr, n_embd * sizeof(float));
    l2_normalize(out, n_embd);

    free(toks);
    return n_embd;
}

LFG_API int32_t lfg_session_embed_tokens(lfg_session * session,
                                          const char * text, int32_t text_len,
                                          float * out, int32_t out_cap) {
    if (!session || !text || text_len <= 0 || !out || out_cap <= 0) return 0;

    if (!session_ensure_embed_none_ctx(session)) {
        lfg_set_last_error(LFG_ERROR_INTERNAL,
            "%s: failed to create per-token embedding context", __func__);
        return 0;
    }

    int32_t n_embd = lfg_model_n_embd_out(session->model);
    const lfg_vocab *vocab = lfg_model_get_vocab(session->model);

    // Tokenize
    int32_t tok_cap = text_len + 16;
    lfg_token *toks = (lfg_token *)malloc(tok_cap * sizeof(lfg_token));
    int32_t n_tok = lfg_tokenize(vocab, text, text_len, toks, tok_cap, true, false);
    if (n_tok < 0) {
        tok_cap = -n_tok;
        toks = (lfg_token *)realloc(toks, tok_cap * sizeof(lfg_token));
        n_tok = lfg_tokenize(vocab, text, text_len, toks, tok_cap, true, false);
    }
    if (n_tok <= 0) { free(toks); return 0; }

    // Truncate to context capacity
    int32_t ctx_cap = (int32_t)lfg_n_ctx(session->embed_none_ctx);
    lfg_token *tok_ptr = toks;
    if (n_tok > ctx_cap) {
        tok_ptr = toks + (n_tok - ctx_cap);
        n_tok = ctx_cap;
    }

    // Check output capacity
    if (out_cap < n_tok * n_embd) { free(toks); return 0; }

    // Forward pass
    lfg_memory_clear(lfg_get_memory(session->embed_none_ctx), true);
    lfg_batch batch = lfg_batch_get_one(tok_ptr, n_tok);
    if (lfg_decode(session->embed_none_ctx, batch) != 0) { free(toks); return 0; }

    // Extract per-token embeddings, L2-normalized
    for (int32_t i = 0; i < n_tok; i++) {
        float *src = lfg_get_embeddings_ith(session->embed_none_ctx, i);
        float *dst = out + i * n_embd;
        if (src) {
            std::memcpy(dst, src, n_embd * sizeof(float));
            l2_normalize(dst, n_embd);
        } else {
            std::memset(dst, 0, n_embd * sizeof(float));
        }
    }

    free(toks);
    return n_tok;
}

// ---------------------------------------------------------------------------
// Guardrails API
// ---------------------------------------------------------------------------

static float guardrail_percentile(const float *src, int32_t count, float pct, float *tmp) {
    if (!src || !tmp || count <= 0) return 0.0f;
    if (pct < 0.0f) pct = 0.0f;
    if (pct > 1.0f) pct = 1.0f;
    std::memcpy(tmp, src, (size_t)count * sizeof(float));
    int32_t idx = (int32_t)std::floor(pct * (count - 1));
    std::nth_element(tmp, tmp + idx, tmp + count);
    return tmp[idx];
}

static void session_guardrail_update(lfg_session *s, int32_t tokens, double elapsed_ms) {
    if (!s || !s->guardrail_active || s->guardrail_window_cap <= 0) return;
    if (tokens <= 0 || elapsed_ms <= 0.0) return;

    float tps = (float)(tokens / (elapsed_ms / 1000.0));
    float latency_ms = (float)elapsed_ms;

    int32_t slot = s->guardrail_write_idx % s->guardrail_window_cap;
    s->guardrail_tps_samples[slot] = tps;
    s->guardrail_latency_samples[slot] = latency_ms;
    s->guardrail_write_idx++;
    if (s->guardrail_sample_count < s->guardrail_window_cap) {
        s->guardrail_sample_count++;
    }

    int32_t count = s->guardrail_sample_count;
    s->guardrail_p50_tps = guardrail_percentile(s->guardrail_tps_samples, count, 0.50f, s->guardrail_tmp_tps);
    s->guardrail_p95_latency_ms = guardrail_percentile(s->guardrail_latency_samples, count, 0.95f, s->guardrail_tmp_latency);

    float score = 0.0f;
    if (s->guardrail_cfg.p50_tps_min > 0.0f && s->guardrail_p50_tps > 0.0f &&
        s->guardrail_p50_tps < s->guardrail_cfg.p50_tps_min) {
        score += (s->guardrail_cfg.p50_tps_min / s->guardrail_p50_tps) - 1.0f;
    }
    if (s->guardrail_cfg.p95_latency_ms_max > 0.0f &&
        s->guardrail_p95_latency_ms > s->guardrail_cfg.p95_latency_ms_max) {
        score += (s->guardrail_p95_latency_ms / s->guardrail_cfg.p95_latency_ms_max) - 1.0f;
    }

    lfg_guardrail_level level = LFG_GUARDRAIL_LEVEL_NONE;
    if (score > 0.0f) {
        if (score < 0.25f) {
            level = LFG_GUARDRAIL_LEVEL_L3_OFF;
        } else if (score < 0.50f) {
            level = LFG_GUARDRAIL_LEVEL_L2_OFF;
        } else if (score < 1.00f) {
            level = LFG_GUARDRAIL_LEVEL_RETRIEVAL_MIN;
        } else if (score < 1.50f) {
            level = LFG_GUARDRAIL_LEVEL_RETRIEVAL_OFF;
        } else if (score < 2.00f) {
            level = LFG_GUARDRAIL_LEVEL_WRITES_OFF;
        } else {
            level = LFG_GUARDRAIL_LEVEL_TRAINING_OFF;
        }
    }

    s->guardrail_level = level;
}

LFG_API bool lfg_session_configure_guardrails(lfg_session * session, const lfg_guardrail_config * config) {
    if (!session) return false;

    if (!config || !config->enabled) {
        session->guardrail_active = false;
        session->guardrail_sample_count = 0;
        session->guardrail_write_idx = 0;
        session->guardrail_p50_tps = 0.0f;
        session->guardrail_p95_latency_ms = 0.0f;
        session->guardrail_level = LFG_GUARDRAIL_LEVEL_NONE;
        free(session->guardrail_tps_samples); session->guardrail_tps_samples = nullptr;
        free(session->guardrail_latency_samples); session->guardrail_latency_samples = nullptr;
        free(session->guardrail_tmp_tps); session->guardrail_tmp_tps = nullptr;
        free(session->guardrail_tmp_latency); session->guardrail_tmp_latency = nullptr;
        session->guardrail_window_cap = 0;
        return true;
    }

    int32_t cap = config->window_size > 0 ? config->window_size : 64;
    float *tps = (float *)malloc((size_t)cap * sizeof(float));
    float *lat = (float *)malloc((size_t)cap * sizeof(float));
    float *tmp_tps = (float *)malloc((size_t)cap * sizeof(float));
    float *tmp_lat = (float *)malloc((size_t)cap * sizeof(float));
    if (!tps || !lat || !tmp_tps || !tmp_lat) {
        free(tps);
        free(lat);
        free(tmp_tps);
        free(tmp_lat);
        lfg_set_last_error(LFG_ERROR_OUT_OF_MEMORY, "%s: failed to allocate guardrail buffers", __func__);
        return false;
    }

    free(session->guardrail_tps_samples);
    free(session->guardrail_latency_samples);
    free(session->guardrail_tmp_tps);
    free(session->guardrail_tmp_latency);

    session->guardrail_tps_samples = tps;
    session->guardrail_latency_samples = lat;
    session->guardrail_tmp_tps = tmp_tps;
    session->guardrail_tmp_latency = tmp_lat;
    session->guardrail_window_cap = cap;
    session->guardrail_sample_count = 0;
    session->guardrail_write_idx = 0;
    session->guardrail_p50_tps = 0.0f;
    session->guardrail_p95_latency_ms = 0.0f;
    session->guardrail_level = LFG_GUARDRAIL_LEVEL_NONE;
    session->guardrail_cfg = *config;
    session->guardrail_active = true;

    std::memset(session->guardrail_tps_samples, 0, (size_t)cap * sizeof(float));
    std::memset(session->guardrail_latency_samples, 0, (size_t)cap * sizeof(float));

    return true;
}

LFG_API bool lfg_session_get_guardrail_stats(lfg_session * session, lfg_guardrail_stats * out) {
    if (!session || !out || !session->guardrail_active) return false;
    out->window_size = session->guardrail_window_cap;
    out->sample_count = session->guardrail_sample_count;
    out->p50_tps = session->guardrail_p50_tps;
    out->p95_latency_ms = session->guardrail_p95_latency_ms;
    out->level = session->guardrail_level;
    out->memory_bytes = 0;
    return true;
}

// ---------------------------------------------------------------------------
// Pythonic Tool Call Parser (PEG combinator-based)
// ---------------------------------------------------------------------------
// Parses: [func_name(key1=val1, key2=val2, ...)]
// Output: JSON arguments string {"key1": val1, "key2": val2, ...}
// Values: "str", 'str', 123, -3.14, True, False, None, {...}, [...]

// AST tag names for semantic extraction
static const char * TAG_TOOL_CALL = "tool_call";
static const char * TAG_TOOL_NAME = "tool_name";
static const char * TAG_ARG       = "arg";
static const char * TAG_ARG_NAME  = "arg_name";
static const char * TAG_ARG_VALUE = "arg_value";

// Build the PEG grammar for Pythonic tool calls (cached as static).
static const common_peg_arena & get_pythonic_grammar() {
    static common_peg_arena arena = build_peg_parser([](common_peg_parser_builder & b) {
        auto ws = b.space();

        // Identifier: [a-zA-Z_][a-zA-Z0-9_]*
        auto ident = b.sequence({
            b.chars("[a-zA-Z_]", 1, 1),
            b.chars("[a-zA-Z0-9_]", 0, -1)
        });

        // Single-quoted string: everything between ' delimiters (handles \' escapes)
        // We match the content character-by-character: either \<any> or non-quote
        auto sq_escape = b.sequence({b.literal("\\"), b.any()});
        auto sq_char = b.choice({sq_escape, b.chars("[^']", 1, 1)});
        auto sq_string = b.sequence({
            b.literal("'"),
            b.zero_or_more(sq_char),
            b.literal("'")
        });

        // Double-quoted string: same pattern with "
        auto dq_escape = b.sequence({b.literal("\\"), b.any()});
        auto dq_char = b.choice({dq_escape, b.chars("[^\"]", 1, 1)});
        auto dq_string = b.sequence({
            b.literal("\""),
            b.zero_or_more(dq_char),
            b.literal("\"")
        });

        auto py_string = b.choice({sq_string, dq_string});

        // Number: optional minus, digits, optional .digits, optional exponent
        auto digits = b.chars("[0-9]", 1, -1);
        auto opt_sign = b.optional(b.chars("[+-]", 1, 1));
        auto frac = b.sequence({b.literal("."), digits});
        auto exp = b.sequence({
            b.chars("[eE]", 1, 1), opt_sign, digits
        });
        auto number = b.sequence({
            b.optional(b.literal("-")),
            digits,
            b.optional(frac),
            b.optional(exp)
        });

        // Python keywords
        auto py_true  = b.literal("True");
        auto py_false = b.literal("False");
        auto py_none  = b.literal("None");

        // Nested JSON value (object/array) — use the built-in JSON parser
        auto json_obj = b.json_object();
        auto json_arr = b.json_array();

        // Value: string | number | keyword | nested JSON
        auto py_value = b.tag(TAG_ARG_VALUE, b.choice({
            py_string, number, py_true, py_false, py_none, json_obj, json_arr
        }));

        // Argument: name = value
        auto arg = b.tag(TAG_ARG, b.sequence({
            b.tag(TAG_ARG_NAME, ident), ws, b.literal("="), ws, py_value
        }));

        // Argument list: arg (, arg)*
        auto arg_list = b.sequence({
            arg,
            b.zero_or_more(b.sequence({ws, b.literal(","), ws, arg})),
            b.optional(b.sequence({ws, b.literal(",")}))  // trailing comma
        });

        // Single tool call: [name(args)]
        auto tool_call = b.tag(TAG_TOOL_CALL, b.sequence({
            b.literal("["),
            ws,
            b.tag(TAG_TOOL_NAME, ident),
            ws,
            b.literal("("),
            ws,
            b.optional(arg_list),
            ws,
            b.literal(")"),
            ws,
            b.literal("]")
        }));

        // Top-level: whitespace-separated tool calls
        return b.zero_or_more(b.sequence({ws, tool_call, ws}));
    });
    return arena;
}

// Convert a Pythonic argument value (extracted text) to a JSON value string.
// Handles: 'str' → "str", "str" → "str", True → true, False → false, None → null,
// numbers pass through, nested JSON passes through.
static std::string pythonic_value_to_json(std::string_view text) {
    if (text.empty()) return "null";

    // Python keywords
    if (text == "True")  return "true";
    if (text == "False") return "false";
    if (text == "None")  return "null";

    // Quoted strings — convert to JSON double-quoted string
    if ((text.front() == '\'' || text.front() == '"') && text.size() >= 2) {
        char quote = text.front();
        std::string_view content = text.substr(1, text.size() - 2);

        std::string result = "\"";
        for (size_t i = 0; i < content.size(); i++) {
            char c = content[i];
            if (c == '\\' && i + 1 < content.size()) {
                char esc = content[i + 1];
                if (esc == quote) {
                    // Escaped delimiter: \' or \"
                    if (quote == '"') {
                        result += "\\\"";
                    } else {
                        result += '\'';
                    }
                    i++;
                    continue;
                }
                // Pass through other escape sequences
                result += c;
                result += esc;
                i++;
                continue;
            }
            // Unescaped double-quote inside single-quoted string needs JSON escaping
            if (quote == '\'' && c == '"') {
                result += "\\\"";
                continue;
            }
            result += c;
        }
        result += '"';
        return result;
    }

    // Numbers, nested JSON objects/arrays — pass through as-is
    return std::string(text);
}

// Public API: parse Pythonic tool calls from text using PEG grammar.
// Input: text like "[func(key='val')]" — may contain multiple calls separated by whitespace.
// Fills `out` array (up to `out_cap` entries). Each entry's id, name, arguments are strdup'd;
// caller must free them with free().
// Returns the number of tool calls parsed.
int32_t lfg_parse_pythonic_tool_calls(const char *text, int32_t text_len,
                                       lfg_tool_call *out, int32_t out_cap) {
    if (!text || text_len <= 0 || !out || out_cap <= 0) return 0;

    const auto & grammar = get_pythonic_grammar();
    common_peg_parse_context ctx(std::string(text, text_len));

    auto result = grammar.parse(ctx, 0);
    if (result.fail() || result.nodes.empty()) return 0;

    int32_t count = 0;

    // Walk AST: find TOOL_CALL tags, extract TOOL_NAME and ARGs within each
    for (auto node_id : result.nodes) {
        if (count >= out_cap) break;

        const auto & node = ctx.ast.get(node_id);
        if (node.tag != TAG_TOOL_CALL) continue;

        std::string name;
        std::string args = "{";
        bool first_arg = true;

        // Walk children of this tool_call node
        for (auto child_id : node.children) {
            const auto & child = ctx.ast.get(child_id);

            if (child.tag == TAG_TOOL_NAME) {
                name = std::string(child.text);
            } else if (child.tag == TAG_ARG) {
                // Each arg has ARG_NAME and ARG_VALUE children
                std::string arg_name;
                std::string arg_value;

                for (auto arg_child_id : child.children) {
                    const auto & arg_child = ctx.ast.get(arg_child_id);
                    if (arg_child.tag == TAG_ARG_NAME) {
                        arg_name = std::string(arg_child.text);
                    } else if (arg_child.tag == TAG_ARG_VALUE) {
                        arg_value = pythonic_value_to_json(arg_child.text);
                    }
                }

                if (!arg_name.empty()) {
                    if (!first_arg) args += ", ";
                    first_arg = false;
                    args += "\"";
                    args += arg_name;
                    args += "\": ";
                    args += arg_value;
                }
            }
        }

        args += '}';

        if (!name.empty()) {
            out[count].id = nullptr;
            out[count].name = strdup(name.c_str());
            out[count].arguments = strdup(args.c_str());
            count++;
        }
    }

    return count;
}

// Parse tool calls from raw output text. Finds <|tool_call_start|>...<|tool_call_end|>
// regions and parses the Pythonic content inside each one. Returns the number of
// tool calls parsed and stored on the session.
static int32_t parse_tool_calls_from_raw_output(lfg_session * session,
                                                 const char * raw, int32_t raw_len) {
    static const char TC_START[] = "<|tool_call_start|>";
    static const char TC_END[]   = "<|tool_call_end|>";
    static const int TC_START_LEN = sizeof(TC_START) - 1;
    static const int TC_END_LEN   = sizeof(TC_END) - 1;

    if (!raw || raw_len <= 0) return 0;

    std::string raw_str(raw, raw_len);
    int32_t total_calls = 0;

    size_t search_pos = 0;
    while (search_pos < raw_str.size()) {
        size_t start_pos = raw_str.find(TC_START, search_pos);
        if (start_pos == std::string::npos) break;

        size_t content_start = start_pos + TC_START_LEN;
        size_t end_pos = raw_str.find(TC_END, content_start);
        if (end_pos == std::string::npos) {
            end_pos = raw_str.size();
        }

        int32_t content_len = (int32_t)(end_pos - content_start);

        // Temporary buffer for parsing — max 16 tool calls per region
        lfg_tool_call tmp[16];
        int32_t n = lfg_parse_pythonic_tool_calls(
            raw_str.c_str() + content_start, content_len, tmp, 16);

        for (int32_t i = 0; i < n; i++) {
            // Ensure capacity on session
            if (session->parsed_tool_call_count >= session->parsed_tool_call_cap) {
                int32_t new_cap = session->parsed_tool_call_cap > 0
                    ? session->parsed_tool_call_cap * 2 : 4;
                session->parsed_tool_calls = (lfg_tool_call *)realloc(
                    session->parsed_tool_calls, new_cap * sizeof(lfg_tool_call));
                session->parsed_tool_call_cap = new_cap;
            }

            lfg_tool_call *slot = &session->parsed_tool_calls[session->parsed_tool_call_count];

            // Generate unique ID
            char id_buf[32];
            std::snprintf(id_buf, sizeof(id_buf), "call_%d", session->tool_call_id_counter++);
            slot->id = strdup(id_buf);
            slot->name = tmp[i].name;        // transfer ownership
            slot->arguments = tmp[i].arguments;  // transfer ownership

            session->parsed_tool_call_count++;
            total_calls++;
        }

        search_pos = (end_pos < raw_str.size()) ? end_pos + TC_END_LEN : raw_str.size();
    }

    return total_calls;
}

// Convert JSON arguments to Pythonic format for history reconstruction.
// {"location":"SF","units":"celsius"} → location="SF", units="celsius"
// Returns malloc'd string, or NULL on error. Caller must free.
static char * json_args_to_pythonic(const char * json_args) {
    if (!json_args || json_args[0] != '{') return nullptr;

    std::string result;
    std::string input(json_args);

    // Simple state-machine JSON object parser for flat key-value pairs
    size_t pos = 1;  // skip '{'
    bool first = true;

    while (pos < input.size()) {
        // Skip whitespace
        while (pos < input.size() && (input[pos] == ' ' || input[pos] == '\t' || input[pos] == '\n')) pos++;
        if (pos >= input.size() || input[pos] == '}') break;

        // Expect key string
        if (input[pos] != '"') break;
        size_t key_start = pos + 1;
        size_t key_end = input.find('"', key_start);
        if (key_end == std::string::npos) break;
        std::string key = input.substr(key_start, key_end - key_start);
        pos = key_end + 1;

        // Skip ':'
        while (pos < input.size() && (input[pos] == ' ' || input[pos] == ':')) pos++;

        if (!first) result += ", ";
        first = false;
        result += key + "=";

        // Parse value
        if (pos >= input.size()) break;

        if (input[pos] == '"') {
            // String value
            size_t val_start = pos + 1;
            size_t val_end = val_start;
            while (val_end < input.size()) {
                if (input[val_end] == '\\' && val_end + 1 < input.size()) {
                    val_end += 2;
                    continue;
                }
                if (input[val_end] == '"') break;
                val_end++;
            }
            result += "\"" + input.substr(val_start, val_end - val_start) + "\"";
            pos = val_end + 1;
        } else if (input.compare(pos, 4, "true") == 0) {
            result += "True";
            pos += 4;
        } else if (input.compare(pos, 5, "false") == 0) {
            result += "False";
            pos += 5;
        } else if (input.compare(pos, 4, "null") == 0) {
            result += "None";
            pos += 4;
        } else if (input[pos] == '{' || input[pos] == '[') {
            // Nested object/array — find matching close bracket
            char open = input[pos], close = (open == '{') ? '}' : ']';
            int depth = 1;
            size_t nest_start = pos;
            pos++;
            while (pos < input.size() && depth > 0) {
                if (input[pos] == '"') {
                    pos++;
                    while (pos < input.size() && input[pos] != '"') {
                        if (input[pos] == '\\') pos++;
                        pos++;
                    }
                }
                else if (input[pos] == open) depth++;
                else if (input[pos] == close) depth--;
                pos++;
            }
            result += input.substr(nest_start, pos - nest_start);
        } else {
            // Number — read until comma, '}', or whitespace
            size_t val_start = pos;
            while (pos < input.size() && input[pos] != ',' && input[pos] != '}'
                   && input[pos] != ' ' && input[pos] != '\n') pos++;
            result += input.substr(val_start, pos - val_start);
        }

        // Skip comma
        while (pos < input.size() && (input[pos] == ' ' || input[pos] == ',')) pos++;
    }

    return strdup(result.c_str());
}

// Ensure raw output buffer has enough capacity.
static void session_ensure_raw_output_cap(lfg_session * s, int32_t needed) {
    if (needed <= s->last_raw_output_cap) return;
    int32_t cap = s->last_raw_output_cap;
    while (cap < needed) cap *= 2;
    s->last_raw_output = (char *)realloc(s->last_raw_output, cap);
    s->last_raw_output_cap = cap;
}

// ---------------------------------------------------------------------------
// Auto tool execution helpers
// ---------------------------------------------------------------------------

// Check that ALL parsed tool calls have a matching registered entry with non-NULL fn.
// Returns false if any call targets a tool without fn (no partial execution).
static bool session_can_auto_execute_tools(lfg_session *session) {
    if (session->parsed_tool_call_count == 0) return false;
    for (int32_t i = 0; i < session->parsed_tool_call_count; ++i) {
        const char *name = session->parsed_tool_calls[i].name;
        if (!name) return false;
        bool found = false;
        for (int32_t j = 0; j < session->tool_count; ++j) {
            if (session->tool_entries[j].fn &&
                std::strcmp(session->tool_entries[j].name, name) == 0) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

// Execute all parsed tool calls and ingest a continuation into the KV cache.
// Returns true on success (caller should set auto_continue = true).
static bool session_execute_and_continue(
    lfg_session *session, const lfg_generate_config *config,
    lfg_generate_result *result, int32_t round)
{
    const lfg_vocab *vocab = lfg_model_get_vocab(session->model);

    // Build combined tool results for all calls
    std::string continuation;

    // <|tool_call_end|> was sampled but NOT ingested (break happens before ingest at line ~3300).
    // It needs to be part of the continuation we tokenize.
    continuation += "<|tool_call_end|><|im_end|>\n";

    for (int32_t i = 0; i < session->parsed_tool_call_count; ++i) {
        const lfg_tool_call *call = &session->parsed_tool_calls[i];

        // Look up fn + fn_user_data from matching tool_entries[]
        lfg_tool_fn fn = nullptr;
        void *fn_ud = nullptr;
        for (int32_t j = 0; j < session->tool_count; ++j) {
            if (std::strcmp(session->tool_entries[j].name, call->name) == 0) {
                fn = session->tool_entries[j].fn;
                fn_ud = session->tool_entries[j].fn_user_data;
                break;
            }
        }
        if (!fn) return false;  // should not happen — checked by can_auto_execute

        const char *result_str = fn(call->arguments, fn_ud);
        if (!result_str) result_str = strdup("{\"error\": \"tool returned null\"}");

        // Fire observation callback
        if (config->tool_call_cb) {
            config->tool_call_cb(call, result_str, (int32_t)std::strlen(result_str),
                                 round, config->tool_call_cb_data);
        }

        continuation += "<|im_start|>tool\n";
        continuation += result_str;
        continuation += "<|im_end|>\n";

        free((void *)result_str);
    }

    continuation += "<|im_start|>assistant\n";

    // Tokenize with parse_special=true for <|...|> tokens
    int32_t cont_len = (int32_t)continuation.size();
    int32_t tok_cap = cont_len + 32;
    lfg_token *toks = (lfg_token *)malloc(tok_cap * sizeof(lfg_token));
    int32_t n_toks = lfg_tokenize(vocab, continuation.c_str(), cont_len,
                                   toks, tok_cap, false, true);
    if (n_toks < 0) {
        tok_cap = -n_toks;
        toks = (lfg_token *)realloc(toks, tok_cap * sizeof(lfg_token));
        n_toks = lfg_tokenize(vocab, continuation.c_str(), cont_len,
                               toks, tok_cap, false, true);
    }

    if (n_toks <= 0) {
        free(toks);
        return false;
    }

    // Ingest via session_ingest_internal — appends to KV at current n_past
    bool ok = session_ingest_internal(session, toks, n_toks, false);
    free(toks);

    if (!ok) return false;

    result->n_tool_rounds = round + 1;
    return true;
}

static bool session_is_reasoning_marker(const lfg_session *s, lfg_token token) {
    if (!s) return false;
    for (size_t i = 0; i < s->reasoning_start_count; ++i) {
        if (s->reasoning_start_tokens[i] == token) return true;
    }
    for (size_t i = 0; i < s->reasoning_end_count; ++i) {
        if (s->reasoning_end_tokens[i] == token) return true;
    }
    return false;
}

static int32_t session_token_to_piece_for_stream(
    const lfg_session *s, const lfg_vocab *vocab, lfg_token token,
    char *buf, int32_t length) {
    int32_t n = lfg_token_to_piece(vocab, token, buf, length, 0, false);
    if (n == 0 && session_is_reasoning_marker(s, token)) {
        n = lfg_token_to_piece(vocab, token, buf, length, 0, true);
    }
    return n;
}

// ---------------------------------------------------------------------------
// Generate Loop API
// ---------------------------------------------------------------------------

LFG_API lfg_generate_config lfg_generate_default_config(void) {
    lfg_generate_config cfg{};
    cfg.max_tokens = 0;
    cfg.include_history_reasoning = false;
    cfg.token_cb = nullptr;
    cfg.token_cb_data = nullptr;
    cfg.entropy_cb = nullptr;
    cfg.entropy_cb_data = nullptr;
    cfg.tool_call_cb = nullptr;
    cfg.tool_call_cb_data = nullptr;
    cfg.max_tool_rounds = 0;
    return cfg;
}

LFG_API lfg_generate_result lfg_session_generate(
    lfg_session * session, lfg_generate_config config)
{
    lfg_generate_result result{};
    if (!session) return result;

    int64_t t_start_us = ggml_time_us();

    const lfg_vocab *vocab = lfg_model_get_vocab(session->model);
    int32_t max_tokens = config.max_tokens > 0
        ? config.max_tokens
        : (session->config.max_tokens > 0 ? session->config.max_tokens : 4096);

    char piece_buf[256];
    bool stopped = false;

    // Clear previous tool call state
    session_free_tool_calls(session);
    session->last_raw_output_len = 0;
    session->tool_call_text_len = 0;
    session->in_tool_call = false;

    // Lazy-cache tool call start/end token IDs (always — model can emit tool
    // calls even without registered tools, e.g. via chat template)
    if (!session->tool_call_tokens_cached) {
        lfg_token toks[4];
        int32_t n = lfg_tokenize(vocab, "<|tool_call_start|>", 19, toks, 4, false, true);
        session->tool_call_start_token = (n == 1) ? toks[0] : LFG_TOKEN_NULL;
        n = lfg_tokenize(vocab, "<|tool_call_end|>", 17, toks, 4, false, true);
        session->tool_call_end_token = (n == 1) ? toks[0] : LFG_TOKEN_NULL;
        session->tool_call_tokens_cached = true;
    }

    // -----------------------------------------------------------------------
    // Stop look-ahead buffer (token-level + text-level).
    // Holds back the last N tokens before emitting them through the callback.
    // Token-level: suppresses prefix tokens when sample() fires a stop seq.
    // Text-level: matches accumulated piece text against stop strings.
    // -----------------------------------------------------------------------
    struct stop_slot {
        lfg_token token;
        int32_t   piece_len;
        char      piece[256];
    };

    // Buffer capacity: max of token-level need and text-level need
    int32_t tok_buf_need = (session->stop_max_len > 1)
        ? (int32_t)(session->stop_max_len - 1) : 0;
    int32_t txt_buf_need = session->stop_text_max_len;  // worst case: 1 char/token
    int32_t stop_buf_cap = (tok_buf_need > txt_buf_need) ? tok_buf_need : txt_buf_need;
    if (!config.token_cb) stop_buf_cap = 0;

    stop_slot *stop_buf = nullptr;
    int32_t stop_buf_count = 0;

    if (stop_buf_cap > 0) {
        // Allocate cap+1 slots for push-before-emit ordering
        stop_buf = (stop_slot *)alloca((stop_buf_cap + 1) * sizeof(stop_slot));
    }

    session->last_stop_len = 0;

    // Auto tool execution state
    int32_t max_tool_rounds = config.max_tool_rounds > 0 ? config.max_tool_rounds : 5;
    int32_t tool_round = 0;
    int32_t tokens_remaining = max_tokens;
    int32_t total_tokens = 0;  // accumulated across all rounds
    bool auto_continue = true;

    while (auto_continue) {
    auto_continue = false;

    for (int32_t i = 0; i < tokens_remaining; ++i) {
        lfg_session_decode(session);
        lfg_token tok = lfg_session_sample(session);

        // Optional live entropy hook: rewind + inject before the sampled token
        // is committed to output or ingested into the KV cache.
        if (config.entropy_cb && session->entropy_active) {
            lfg_entropy_event ev{};
            std::vector<float> embd_buf;
            float *embd_ptr = nullptr;

            if (session->entropy_n_embd > 0) {
                embd_buf.resize(session->entropy_n_embd);
                embd_ptr = embd_buf.data();
            }

            bool restarted_from_entropy = false;
            while (lfg_session_entropy_pop(session, &ev, embd_ptr, session->entropy_n_embd)) {
                const char *inject = config.entropy_cb(
                    &ev, ev.n_embd > 0 ? embd_ptr : nullptr, config.entropy_cb_data);
                if (!inject || inject[0] == '\0') {
                    continue;
                }

                if (!lfg_session_rewind(session, ev.checkpoint_id)) {
                    continue;
                }

                restarted_from_entropy = true;

                int32_t inject_len = (int32_t)std::strlen(inject);
                int32_t tok_cap = inject_len + 16;
                lfg_token *inject_toks = (lfg_token *)malloc(tok_cap * sizeof(lfg_token));
                int32_t n_inject = lfg_tokenize(
                    vocab, inject, inject_len, inject_toks, tok_cap, false, false);

                if (n_inject < 0) {
                    tok_cap = -n_inject;
                    inject_toks = (lfg_token *)realloc(inject_toks, tok_cap * sizeof(lfg_token));
                    n_inject = lfg_tokenize(
                        vocab, inject, inject_len, inject_toks, tok_cap, false, false);
                }

                if (n_inject > 0) {
                    bool ok = lfg_session_ingest_tokens(session, inject_toks, n_inject, false);
                    if (ok) {
                        result.n_retrievals++;
                    }
                }

                free(inject_toks);
                lfg_session_entropy_flush(session);
                if (restarted_from_entropy) {
                    i--;
                    break;
                }
            }

            if (restarted_from_entropy) {
                continue;
            }
        }

        // Accumulate raw output (with special tokens) for tool call parsing
        {
            char raw_piece[256];
            int32_t rn = lfg_token_to_piece(vocab, tok, raw_piece, sizeof(raw_piece), 0, true);
            if (rn > 0) {
                session_ensure_raw_output_cap(session, session->last_raw_output_len + rn + 1);
                std::memcpy(session->last_raw_output + session->last_raw_output_len, raw_piece, rn);
                session->last_raw_output_len += rn;
                session->last_raw_output[session->last_raw_output_len] = '\0';
            }
        }

        // Detect <|tool_call_start|> token — enter tool call accumulation mode
        if (session->tool_call_start_token != LFG_TOKEN_NULL &&
            tok == session->tool_call_start_token) {
            session->in_tool_call = true;
            session->tool_call_text_len = 0;
        }

        // Detect <|tool_call_end|> token — stop generation with TOOL_CALL reason.
        // This is needed because special tokens don't appear in the text-based
        // stop string matching (lfg_token_to_piece with special=false).
        if (session->tool_call_end_token != LFG_TOKEN_NULL &&
            tok == session->tool_call_end_token) {
            result.stop_reason = LFG_STOP_TOOL_CALL;
            total_tokens += i + 1;
            stopped = true;
            break;
        }

        if (lfg_vocab_is_eog(vocab, tok)) {
            // Flush buffered tokens, discarding stop sequence prefix tokens
            if (stop_buf_count > 0) {
                int32_t discard = (session->last_stop_len > 1)
                    ? (session->last_stop_len - 1) : 0;

                // Text-level: suppress trailing tokens whose text forms a
                // prefix of any stop string (e.g. "<|" before EOS when the
                // stop string is "<|im_end|>").
                if (session->stop_text_count > 0 && discard == 0) {
                    // Build suffix text from buffer tail inward
                    char tail_buf[1024];
                    int32_t tail_len = 0;
                    for (int32_t j = 0; j < stop_buf_count && tail_len < (int32_t)sizeof(tail_buf) - 256; j++) {
                        std::memcpy(tail_buf + tail_len, stop_buf[j].piece, stop_buf[j].piece_len);
                        tail_len += stop_buf[j].piece_len;
                    }
                    tail_buf[tail_len] = '\0';

                    // Check if any stop string starts with the buffered text
                    for (int32_t s = 0; s < session->stop_text_count; s++) {
                        if (tail_len > 0 && tail_len <= session->stop_text_lens[s] &&
                            std::strncmp(tail_buf, session->stop_texts[s], tail_len) == 0) {
                            discard = stop_buf_count;
                            break;
                        }
                    }
                }

                int32_t flush_n = stop_buf_count - discard;
                if (flush_n < 0) flush_n = 0;
                for (int32_t j = 0; j < flush_n && !session->in_tool_call; j++) {
                    lfg_generate_action action = config.token_cb(
                        stop_buf[j].token, stop_buf[j].piece,
                        stop_buf[j].piece_len, config.token_cb_data);
                    if (action == LFG_GENERATE_STOP) {
                        result.stop_reason = LFG_STOP_CALLBACK;
                        stopped = true;
                        break;
                    }
                }
                stop_buf_count = 0;
            }
            if (!stopped) {
                result.stop_reason = LFG_STOP_EOS;
            }
            total_tokens += i + 1;
            stopped = true;
            break;
        }

        // Token callback with stop look-ahead buffering (token + text level)
        if (stop_buf_cap > 0) {
            // Convert token to piece text
            int32_t n = session_token_to_piece_for_stream(session, vocab, tok, piece_buf, sizeof(piece_buf));
            if (n < 0) n = 0;

            // Push new token into buffer (may temporarily exceed cap)
            stop_buf[stop_buf_count].token = tok;
            stop_buf[stop_buf_count].piece_len = n;
            std::memcpy(stop_buf[stop_buf_count].piece, piece_buf, n);
            stop_buf[stop_buf_count].piece[n] = '\0';
            stop_buf_count++;

            // Text-level stop matching: check if buffered text contains any stop string.
            // Uses strstr (not suffix match) because BPE tokens can include trailing
            // characters after the stop text (e.g. ">\n" instead of just ">").
            if (session->stop_text_count > 0) {
                // Concatenate piece texts from buffer
                char match_buf[1024];
                int32_t match_len = 0;
                for (int32_t j = 0; j < stop_buf_count && match_len < (int32_t)sizeof(match_buf) - 256; j++) {
                    std::memcpy(match_buf + match_len, stop_buf[j].piece, stop_buf[j].piece_len);
                    match_len += stop_buf[j].piece_len;
                }
                match_buf[match_len] = '\0';

                for (int32_t s = 0; s < session->stop_text_count; s++) {
                    const char *found = std::strstr(match_buf, session->stop_texts[s]);
                    if (!found) continue;

                    // Stop string found — emit tokens before it, discard from it onward
                    int32_t emit_chars = (int32_t)(found - match_buf);

                    // Find token boundary: count whole tokens that fit before emit_chars
                    int32_t chars_so_far = 0;
                    int32_t flush_n = 0;
                    for (int32_t j = 0; j < stop_buf_count; j++) {
                        if (chars_so_far + stop_buf[j].piece_len <= emit_chars) {
                            chars_so_far += stop_buf[j].piece_len;
                            flush_n = j + 1;
                        } else {
                            break;
                        }
                    }

                    // Flush safe whole tokens (suppress during tool call)
                    for (int32_t j = 0; j < flush_n && !stopped; j++) {
                        if (!session->in_tool_call) {
                            lfg_generate_action action = config.token_cb(
                                stop_buf[j].token, stop_buf[j].piece,
                                stop_buf[j].piece_len, config.token_cb_data);
                            if (action == LFG_GENERATE_STOP) {
                                result.stop_reason = LFG_STOP_CALLBACK;
                                stopped = true;
                            }
                        }
                    }

                    // Emit safe prefix of partially-matched boundary token
                    if (!stopped && flush_n < stop_buf_count && chars_so_far < emit_chars) {
                        int32_t safe_chars = emit_chars - chars_so_far;
                        if (!session->in_tool_call) {
                            lfg_generate_action action = config.token_cb(
                                stop_buf[flush_n].token,
                                stop_buf[flush_n].piece,
                                safe_chars, config.token_cb_data);
                            if (action == LFG_GENERATE_STOP) {
                                result.stop_reason = LFG_STOP_CALLBACK;
                                stopped = true;
                            }
                        }
                    }

                    if (!stopped) {
                        if (std::strcmp(session->stop_texts[s], "<|tool_call_end|>") == 0)
                            result.stop_reason = LFG_STOP_TOOL_CALL;
                        else
                            result.stop_reason = LFG_STOP_EOS;
                    }
                    total_tokens += i + 1;
                    stopped = true;
                    stop_buf_count = 0;
                    break;
                }
                if (stopped) break;
            }

            // If buffer exceeds capacity, emit oldest token (confirmed safe)
            while (stop_buf_count > stop_buf_cap) {
                if (!session->in_tool_call) {
                    lfg_generate_action action = config.token_cb(
                        stop_buf[0].token, stop_buf[0].piece,
                        stop_buf[0].piece_len, config.token_cb_data);
                    if (action == LFG_GENERATE_STOP) {
                        result.stop_reason = LFG_STOP_CALLBACK;
                        stopped = true;
                        break;
                    }
                }
                if (stop_buf_count > 1) {
                    std::memmove(stop_buf, stop_buf + 1,
                                 (stop_buf_count - 1) * sizeof(stop_slot));
                }
                stop_buf_count--;
            }
            if (stopped) break;
        } else if (config.token_cb && !session->in_tool_call) {
            // No buffering needed — emit directly (suppress during tool call)
            int32_t n = session_token_to_piece_for_stream(session, vocab, tok, piece_buf, sizeof(piece_buf));
            if (n < 0) n = 0;
            lfg_generate_action action = config.token_cb(tok, piece_buf, n, config.token_cb_data);
            if (action == LFG_GENERATE_STOP) {
                result.stop_reason = LFG_STOP_CALLBACK;
                stopped = true;
                break;
            }
        }

        lfg_session_ingest_tokens(session, &tok, 1, false);
    }

    // Flush remaining buffered tokens on non-EOS stop (max_tokens)
    if (stop_buf_count > 0 && !stopped && !session->in_tool_call) {
        for (int32_t j = 0; j < stop_buf_count; j++) {
            lfg_generate_action action = config.token_cb(
                stop_buf[j].token, stop_buf[j].piece,
                stop_buf[j].piece_len, config.token_cb_data);
            if (action == LFG_GENERATE_STOP) {
                result.stop_reason = LFG_STOP_CALLBACK;
                stopped = true;
                break;
            }
        }
    }

    if (!stopped) {
        total_tokens += tokens_remaining;
        result.stop_reason = LFG_STOP_MAX_TOKENS;
    }

    // Parse tool calls from raw output when generation stopped on tool_call_end
    if (result.stop_reason == LFG_STOP_TOOL_CALL) {
        result.n_tool_calls = parse_tool_calls_from_raw_output(
            session, session->last_raw_output, session->last_raw_output_len);
    }

    // --- Auto tool execution: if all calls have fn callbacks, execute and continue ---
    if (result.stop_reason == LFG_STOP_TOOL_CALL &&
        tool_round < max_tool_rounds &&
        session_can_auto_execute_tools(session))
    {
        if (session_execute_and_continue(session, &config, &result, tool_round)) {
            // Decrease remaining tokens by what this round consumed
            tokens_remaining = max_tokens - total_tokens;
            if (tokens_remaining <= 0) {
                // No budget left — keep TOOL_CALL stop reason
                break;
            }

            // Reset per-round state for next generation round
            tool_round++;
            stopped = false;
            stop_buf_count = 0;
            session->in_tool_call = false;
            session->last_raw_output_len = 0;
            session->tool_call_text_len = 0;
            session_free_tool_calls(session);
            result.n_tool_calls = 0;
            result.stop_reason = LFG_STOP_EOS;  // reset — will be set by next round
            session->last_stop_len = 0;
            auto_continue = true;
        }
    }

    } // while (auto_continue)

    result.n_tokens = total_tokens;

    // Post-loop: flush any in-progress confidence run as a final event
    if (session->confidence_active &&
        session->confidence_run_count >= session->confidence_min_span) {
        int32_t wi = session->confidence_write_idx;
        int slot = wi % session->confidence_ring_cap;

        lfg_confidence_ring_slot *rs = &session->confidence_slots[slot];
        rs->event.mean_entropy = session->confidence_run_entropy_sum / session->confidence_run_count;
        rs->event.min_entropy  = session->confidence_run_min_entropy;
        rs->event.span_length  = session->confidence_run_count;
        rs->event.start_pos    = session->confidence_run_start_pos;
        rs->event.end_pos      = session->n_past;
        rs->event.n_embd       = session->confidence_n_embd;
        rs->embedding          = nullptr;  // Lazy — computed in pop()

        __atomic_store_n(&session->confidence_write_idx, wi + 1, __ATOMIC_RELEASE);

        session->confidence_run_count = 0;
        session->confidence_run_entropy_sum = 0.0f;
        session->confidence_run_min_entropy = 1.0f;
    }

    {
        int64_t t_end_us = ggml_time_us();
        double elapsed_ms = (t_end_us - t_start_us) / 1000.0;
        session_guardrail_update(session, result.n_tokens, elapsed_ms);
    }

    return result;
}

LFG_API lfg_generate_result lfg_session_prompt_generate(
    lfg_session * session,
    const char * prompt, int32_t prompt_len,
    bool add_bos,
    lfg_generate_config config)
{
    lfg_generate_result result{};
    if (!session || !prompt || prompt_len <= 0) return result;

    // Tool injection: rank tools and prepend JSON to the prompt text so tools
    // appear inside the prompt context, not between prompt and generated output.
    const char *effective_prompt = prompt;
    int32_t effective_len = prompt_len;
    char *combined_prompt = nullptr;

    if (session->tool_count > 0 && !session->tools_injected && session->tool_top_k > 0) {
        int32_t n_embd = session->tool_n_embd;
        int32_t got = lfg_session_embed(session, prompt, prompt_len,
                                         session->tool_query_embd, n_embd);
        if (got > 0) {
            int32_t tools_len = session_rank_and_format_tools(session, session->tool_query_embd);
            if (tools_len > 0) {
                // Prepend tool list + newline to prompt
                int32_t combined_len = tools_len + 1 + prompt_len;
                combined_prompt = (char *)malloc(combined_len + 1);
                std::memcpy(combined_prompt, session->tool_json_buf, tools_len);
                combined_prompt[tools_len] = '\n';
                std::memcpy(combined_prompt + tools_len + 1, prompt, prompt_len);
                combined_prompt[combined_len] = '\0';

                effective_prompt = combined_prompt;
                effective_len = combined_len;
            }
        }

        session->tools_injected = true;
    }

    const lfg_vocab *vocab = lfg_model_get_vocab(session->model);
    int32_t tok_cap = effective_len + 16;
    lfg_token *tokens = (lfg_token *)malloc(tok_cap * sizeof(lfg_token));
    int32_t n = lfg_tokenize(vocab, effective_prompt, effective_len, tokens, tok_cap, add_bos, false);
    if (n < 0) {
        tok_cap = -n;
        tokens = (lfg_token *)realloc(tokens, tok_cap * sizeof(lfg_token));
        n = lfg_tokenize(vocab, effective_prompt, effective_len, tokens, tok_cap, add_bos, false);
    }

    free(combined_prompt);

    if (n <= 0) {
        free(tokens);
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT,
            "%s: tokenization failed", __func__);
        return result;
    }

    // Capture: detokenize the actual token array so we see exactly what the model sees
    {
        free(session->last_formatted_prompt);
        int32_t text_cap = n * 8;
        session->last_formatted_prompt = (char *)malloc(text_cap);
        int32_t text_len = lfg_detokenize(vocab, tokens, n,
            session->last_formatted_prompt, text_cap, false, true);
        if (text_len < 0) {
            text_cap = -text_len;
            session->last_formatted_prompt = (char *)realloc(session->last_formatted_prompt, text_cap);
            text_len = lfg_detokenize(vocab, tokens, n,
                session->last_formatted_prompt, text_cap, false, true);
        }
        session->last_formatted_prompt_len = text_len > 0 ? text_len : 0;
    }

    // Prompt tokens must NOT be fed through the grammar sampler — grammar
    // constrains generated output only.  Pass update_sampler=false so the
    // grammar (and other accept-sensitive samplers) never see prompt tokens.
    lfg_session_ingest_tokens(session, tokens, n, false);
    free(tokens);

    return lfg_session_generate(session, config);
}

LFG_API lfg_generate_result lfg_session_chat_generate(
    lfg_session * session,
    const lfg_chat_message * messages, size_t n_messages,
    lfg_generate_config config)
{
    lfg_generate_result result{};
    if (!session || !messages || n_messages == 0) return result;

    // 1. Tool injection: rank tools and build JSON to inject into the system message.
    //    This must happen BEFORE template application so the tools appear inside
    //    the prompt (system message), not between the prompt and generated output.
    int32_t tool_json_len = 0;
    char *tool_json_text = nullptr;

    if (session->tool_count > 0 && !session->tools_injected && session->tool_top_k > 0) {
        // Find last user message for query embedding
        const char *query_text = nullptr;
        int32_t query_len = 0;
        for (int i = (int)n_messages - 1; i >= 0; --i) {
            if (messages[i].role && std::strcmp(messages[i].role, "user") == 0) {
                query_text = messages[i].content;
                query_len = query_text ? (int32_t)std::strlen(query_text) : 0;
                break;
            }
        }

        if (query_text && query_len > 0) {
            int32_t n_embd = session->tool_n_embd;
            int32_t got = lfg_session_embed(session, query_text, query_len,
                                             session->tool_query_embd, n_embd);
            if (got > 0) {
                tool_json_len = session_rank_and_format_tools(session, session->tool_query_embd);
                if (tool_json_len > 0) {
                    tool_json_text = session->tool_json_buf;
                }
            }
        }

        session->tools_injected = true;
    }

    // 2. Build message array with tools injected into the system message
    const lfg_chat_message *tmpl_msgs = messages;
    size_t tmpl_n = n_messages;

    // Temporary storage for modified messages (only allocated if tools present)
    lfg_chat_message *mod_msgs = nullptr;
    char *sys_content_buf = nullptr;

    if (tool_json_text && tool_json_len > 0) {
        // Find existing system message
        int sys_idx = -1;
        for (size_t i = 0; i < n_messages; ++i) {
            if (messages[i].role && std::strcmp(messages[i].role, "system") == 0) {
                sys_idx = (int)i;
                break;
            }
        }

        if (sys_idx >= 0) {
            // Append tool list to existing system message
            const char *orig = messages[sys_idx].content ? messages[sys_idx].content : "";
            size_t orig_len = std::strlen(orig);
            size_t buf_len = orig_len + 2 + tool_json_len + 1;
            sys_content_buf = (char *)malloc(buf_len);
            int written = std::snprintf(sys_content_buf, buf_len, "%s\n\n%.*s",
                                        orig, (int)tool_json_len, tool_json_text);
            sys_content_buf[written] = '\0';

            mod_msgs = (lfg_chat_message *)malloc(n_messages * sizeof(lfg_chat_message));
            std::memcpy(mod_msgs, messages, n_messages * sizeof(lfg_chat_message));
            mod_msgs[sys_idx].content = sys_content_buf;

            tmpl_msgs = mod_msgs;
            tmpl_n = n_messages;
        } else {
            // Insert a new system message at the beginning with the tool list
            size_t new_n = n_messages + 1;
            sys_content_buf = (char *)malloc(tool_json_len + 1);
            std::memcpy(sys_content_buf, tool_json_text, tool_json_len);
            sys_content_buf[tool_json_len] = '\0';

            mod_msgs = (lfg_chat_message *)malloc(new_n * sizeof(lfg_chat_message));
            mod_msgs[0].role = "system";
            mod_msgs[0].content = sys_content_buf;
            std::memcpy(mod_msgs + 1, messages, n_messages * sizeof(lfg_chat_message));

            tmpl_msgs = mod_msgs;
            tmpl_n = new_n;
        }
    }

    // 2b. Strip <think>...</think> from assistant messages in history.
    //     Saves context for multi-turn chat — reasoning is internal to the model
    //     and doesn't need to be replayed.  Caller can set
    //     config.include_history_reasoning=true to keep it.
    char **stripped_bufs = nullptr;
    int    stripped_count = 0;

    if (!config.include_history_reasoning) {
        // Count assistant messages that contain <think>
        int n_strip = 0;
        for (size_t i = 0; i < tmpl_n; ++i) {
            if (tmpl_msgs[i].role && std::strcmp(tmpl_msgs[i].role, "assistant") == 0 &&
                tmpl_msgs[i].content && std::strstr(tmpl_msgs[i].content, "<think>")) {
                n_strip++;
            }
        }

        if (n_strip > 0) {
            // Ensure we have a mutable message array
            if (!mod_msgs) {
                mod_msgs = (lfg_chat_message *)malloc(tmpl_n * sizeof(lfg_chat_message));
                std::memcpy(mod_msgs, tmpl_msgs, tmpl_n * sizeof(lfg_chat_message));
                tmpl_msgs = mod_msgs;
            }

            stripped_bufs = (char **)calloc(n_strip, sizeof(char *));

            for (size_t i = 0; i < tmpl_n; ++i) {
                if (mod_msgs[i].role && std::strcmp(mod_msgs[i].role, "assistant") == 0 &&
                    mod_msgs[i].content) {
                    const char *think_open = std::strstr(mod_msgs[i].content, "<think>");
                    if (!think_open) continue;

                    const char *think_close = std::strstr(think_open, "</think>");
                    if (think_close) {
                        // Strip <think>...</think> block, keep text before and after
                        const char *after = think_close + 8;  // len("</think>")
                        // Skip leading whitespace after </think>
                        while (*after == ' ' || *after == '\t' || *after == '\n' || *after == '\r') after++;

                        size_t before_len = (size_t)(think_open - mod_msgs[i].content);
                        size_t after_len = std::strlen(after);
                        char *buf = (char *)malloc(before_len + after_len + 1);
                        if (before_len > 0) std::memcpy(buf, mod_msgs[i].content, before_len);
                        std::memcpy(buf + before_len, after, after_len);
                        buf[before_len + after_len] = '\0';

                        stripped_bufs[stripped_count++] = buf;
                        mod_msgs[i].content = buf;
                    } else {
                        // Unclosed <think> (truncated) — strip from <think> onward
                        size_t before_len = (size_t)(think_open - mod_msgs[i].content);
                        char *buf = (char *)malloc(before_len + 1);
                        if (before_len > 0) std::memcpy(buf, mod_msgs[i].content, before_len);
                        buf[before_len] = '\0';

                        stripped_bufs[stripped_count++] = buf;
                        mod_msgs[i].content = buf;
                    }
                }
            }
        }
    }

    // 2c. Reconstruct tool call content for assistant messages with structured tool_calls.
    //     For multi-turn tool use, the consumer sets tool_calls[] on assistant messages
    //     (from a previous generation). We reconstruct the Pythonic format the model expects:
    //     <|tool_call_start|>[name(key="val")]<|tool_call_end|>\n<original content>
    char **tc_recon_bufs = nullptr;
    int    tc_recon_count = 0;
    {
        // Count assistant messages with tool_calls
        int n_recon = 0;
        for (size_t i = 0; i < tmpl_n; ++i) {
            if (tmpl_msgs[i].role && std::strcmp(tmpl_msgs[i].role, "assistant") == 0 &&
                tmpl_msgs[i].n_tool_calls > 0 && tmpl_msgs[i].tool_calls) {
                n_recon++;
            }
        }

        if (n_recon > 0) {
            // Ensure we have a mutable message array
            if (!mod_msgs) {
                mod_msgs = (lfg_chat_message *)malloc(tmpl_n * sizeof(lfg_chat_message));
                std::memcpy(mod_msgs, tmpl_msgs, tmpl_n * sizeof(lfg_chat_message));
                tmpl_msgs = mod_msgs;
            }

            tc_recon_bufs = (char **)calloc(n_recon, sizeof(char *));

            for (size_t i = 0; i < tmpl_n; ++i) {
                if (!(mod_msgs[i].role && std::strcmp(mod_msgs[i].role, "assistant") == 0 &&
                      mod_msgs[i].n_tool_calls > 0 && mod_msgs[i].tool_calls)) continue;

                // Build reconstructed content
                std::string recon;
                for (int32_t tc = 0; tc < mod_msgs[i].n_tool_calls; ++tc) {
                    const lfg_tool_call *call = &mod_msgs[i].tool_calls[tc];
                    char *pythonic_args = json_args_to_pythonic(call->arguments);
                    recon += "<|tool_call_start|>[";
                    recon += call->name ? call->name : "";
                    recon += "(";
                    if (pythonic_args) {
                        recon += pythonic_args;
                        free(pythonic_args);
                    }
                    recon += ")]<|tool_call_end|>\n";
                }

                // Append original content (if any)
                if (mod_msgs[i].content && mod_msgs[i].content[0] != '\0') {
                    recon += mod_msgs[i].content;
                }

                tc_recon_bufs[tc_recon_count] = strdup(recon.c_str());
                mod_msgs[i].content = tc_recon_bufs[tc_recon_count];
                tc_recon_count++;
            }
        }
    }

    // 3. Detect chat template from model metadata
    const char *tmpl_str = lfg_model_chat_template(session->model, nullptr);
    const bool auto_reasoning = session_auto_configure_reasoning_from_markers(session, tmpl_str);


    // 3b. Chat-scoped surprise: compute prefix token count for context messages.
    //     Surprise should only evaluate the last user turn, not the full history.
    if (session->surprise_active && n_messages > 1) {
        // Format all-but-last message without assistant prompt
        int32_t prefix_needed = lfg_chat_apply_template(
            tmpl_str, tmpl_msgs, tmpl_n - 1, false, nullptr, 0);
        if (prefix_needed > 0) {
            char *prefix_buf = (char *)malloc(prefix_needed + 1);
            lfg_chat_apply_template(tmpl_str, tmpl_msgs, tmpl_n - 1, false,
                                    prefix_buf, prefix_needed + 1);
            prefix_buf[prefix_needed] = '\0';

            const lfg_vocab *vocab = lfg_model_get_vocab(session->model);
            int32_t prefix_tok_cap = prefix_needed + 16;
            lfg_token *prefix_toks = (lfg_token *)malloc(prefix_tok_cap * sizeof(lfg_token));
            int32_t pn = lfg_tokenize(vocab, prefix_buf, prefix_needed,
                                       prefix_toks, prefix_tok_cap, true, true);
            if (pn < 0) {
                prefix_tok_cap = -pn;
                prefix_toks = (lfg_token *)realloc(prefix_toks, prefix_tok_cap * sizeof(lfg_token));
                pn = lfg_tokenize(vocab, prefix_buf, prefix_needed,
                                   prefix_toks, prefix_tok_cap, true, true);
            }
            session->surprise_skip_tokens = pn > 0 ? pn : 0;
            free(prefix_toks);
            free(prefix_buf);
        }
    }

    // 4. Apply template: first call with NULL buf to get required size
    int32_t needed = lfg_chat_apply_template(tmpl_str, tmpl_msgs, tmpl_n, true, nullptr, 0);
    if (needed <= 0) {
        for (int i = 0; i < stripped_count; ++i) free(stripped_bufs[i]);
        free(stripped_bufs);
        for (int i = 0; i < tc_recon_count; ++i) free(tc_recon_bufs[i]);
        free(tc_recon_bufs);
        free(mod_msgs);
        free(sys_content_buf);
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT,
            "%s: chat template failed (unknown template?)", __func__);
        return result;
    }

    // Allocate and format
    char *formatted = (char *)malloc(needed + 1);
    lfg_chat_apply_template(tmpl_str, tmpl_msgs, tmpl_n, true, formatted, needed + 1);
    formatted[needed] = '\0';

    for (int i = 0; i < stripped_count; ++i) free(stripped_bufs[i]);
    free(stripped_bufs);
    for (int i = 0; i < tc_recon_count; ++i) free(tc_recon_bufs[i]);
    free(tc_recon_bufs);
    free(mod_msgs);
    free(sys_content_buf);

    // 5. Tokenize
    const lfg_vocab *vocab = lfg_model_get_vocab(session->model);
    int32_t tok_cap = needed + 16;
    lfg_token *tokens = (lfg_token *)malloc(tok_cap * sizeof(lfg_token));
    int32_t n = lfg_tokenize(vocab, formatted, needed, tokens, tok_cap, true, true);
    if (n < 0) {
        tok_cap = -n;
        tokens = (lfg_token *)realloc(tokens, tok_cap * sizeof(lfg_token));
        n = lfg_tokenize(vocab, formatted, needed, tokens, tok_cap, true, true);
    }
    free(formatted);

    if (n <= 0) {
        free(tokens);
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT,
            "%s: tokenization failed", __func__);
        return result;
    }

    // Capture: detokenize the actual token array so we see exactly what the model sees
    {
        free(session->last_formatted_prompt);
        int32_t text_cap = n * 8; // generous estimate
        session->last_formatted_prompt = (char *)malloc(text_cap);
        int32_t text_len = lfg_detokenize(vocab, tokens, n,
            session->last_formatted_prompt, text_cap, false, true);
        if (text_len < 0) {
            text_cap = -text_len;
            session->last_formatted_prompt = (char *)realloc(session->last_formatted_prompt, text_cap);
            text_len = lfg_detokenize(vocab, tokens, n,
                session->last_formatted_prompt, text_cap, false, true);
        }
        session->last_formatted_prompt_len = text_len > 0 ? text_len : 0;
    }

    // 6. Ingest prompt — update_sampler=false so grammar isn't fed prompt tokens
    bool ok = lfg_session_ingest_tokens(session, tokens, n, false);
    session->surprise_skip_tokens = 0;  // consumed
    free(tokens);

    if (!ok) {
        lfg_set_last_error(LFG_ERROR_INTERNAL,
            "%s: prompt ingestion failed (prompt %d tokens may exceed n_ctx %d)",
            __func__, n, session->config.n_ctx);
        return result;
    }

    // 7. Auto-configure text stop strings for the EOS token's text representation,
    //    and for <|tool_call_end|> when tools are registered.
    //    The generate loop already stops on the special EOS token, but small
    //    models sometimes generate the TEXT version (e.g. "<|im_end|>" spelled
    //    out as regular tokens) which doesn't trigger EOG detection.  Using a
    //    text-level stop string catches the text form regardless of how the
    //    tokenizer splits it.
    if (session->stop_text_count == 0) {
        const char *stop_strs[4];
        int32_t n_stop = 0;

        lfg_token eos = lfg_vocab_eos(vocab);
        char eos_text[64];
        char close_text[66];
        int32_t eos_len = lfg_token_to_piece(vocab, eos, eos_text, sizeof(eos_text), 0, true);
        if (eos_len > 0) {
            eos_text[eos_len] = '\0';
            stop_strs[n_stop++] = eos_text;
            // Models sometimes generate a "closing tag" variant of the EOS
            // token text, e.g. "</|im_end|>" instead of "<|im_end|>".
            if (eos_text[0] == '<' && eos_len + 1 < (int32_t)sizeof(eos_text)) {
                close_text[0] = '<';
                close_text[1] = '/';
                std::memcpy(close_text + 2, eos_text + 1, eos_len - 1);
                close_text[eos_len + 1] = '\0';
                stop_strs[n_stop++] = close_text;
            }
        }

        // Always stop on <|tool_call_end|> — the model can emit tool calls
        // even without registered tools (e.g. via chat template with tools)
        stop_strs[n_stop++] = "<|tool_call_end|>";

        if (n_stop > 0) {
            lfg_session_configure_stop_strings(session, stop_strs, n_stop);
        }
    }

    // 8. Generate
    lfg_generate_result out = lfg_session_generate(session, config);


    (void)auto_reasoning;
    return out;
}

// ---------------------------------------------------------------------------
// Last Formatted Prompt getter
// ---------------------------------------------------------------------------

LFG_API const char * lfg_session_get_last_prompt(lfg_session * session, int32_t * len_out) {
    if (!session) return nullptr;
    if (len_out) *len_out = session->last_formatted_prompt_len;
    return session->last_formatted_prompt;
}

// ---------------------------------------------------------------------------
// Structured Tool Call Accessors
// ---------------------------------------------------------------------------

LFG_API const lfg_tool_call * lfg_session_get_tool_calls(lfg_session * session, int32_t * n_out) {
    if (!session) {
        if (n_out) *n_out = 0;
        return nullptr;
    }
    if (n_out) *n_out = session->parsed_tool_call_count;
    return session->parsed_tool_calls;
}

LFG_API const char * lfg_session_get_last_output(lfg_session * session, int32_t * len_out) {
    if (!session) {
        if (len_out) *len_out = 0;
        return nullptr;
    }
    if (len_out) *len_out = session->last_raw_output_len;
    return session->last_raw_output;
}

LFG_API void lfg_session_set_tool_call_format(lfg_session * session, lfg_tool_call_format format) {
    if (session) session->tool_call_format = format;
}

// ---------------------------------------------------------------------------
// Model Loader C API
// ---------------------------------------------------------------------------

LFG_API lfg_model_load_config lfg_model_load_default_config(void) {
    lfg_model_load_config cfg{};
    cfg.model_path = nullptr;
    cfg.use_mmap = true;
    cfg.use_mlock = false;
    cfg.n_gpu_layers = 0;
    return cfg;
}

LFG_API lfg_model * lfg_load_model(const lfg_model_load_config * config) {
    if (!config || !config->model_path) return nullptr;

    lfg_model_params params = lfg_model_default_params();
    params.use_mmap = config->use_mmap;
    params.use_mlock = config->use_mlock;
    params.n_gpu_layers = config->n_gpu_layers;

    lfg_model *model = lfg_model_load_from_file(config->model_path, params);
    if (!model) {
        spdlog::error("Failed to load model from {}", config->model_path);
    }
    return model;
}

LFG_API lfg_model_stats lfg_model_get_stats(const lfg_model * model) {
    lfg_model_stats stats{};
    if (!model) return stats;
    stats.n_params = lfg_model_n_params(model);
    stats.size_bytes = lfg_model_size(model);
    stats.n_vocab = lfg_vocab_n_tokens(lfg_model_get_vocab(model));
    stats.n_ctx_train = lfg_model_n_ctx_train(model);
    return stats;
}

LFG_API int32_t lfg_model_get_metadata_str(const lfg_model * model,
                                                 const char * key,
                                                 char * buf, size_t buf_size) {
    if (!model || !key) return -1;
    return lfg_model_meta_val_str(model, key, buf, (int32_t)buf_size);
}
