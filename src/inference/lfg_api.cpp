#include "lfg_api.h"

#include "json_schema_to_grammar.h"
#include "lfg_impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// (Surprise monitor uses simple aggregate — no ring buffer needed)

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
    int    forcing_reasoning_end_index;  // -1 = not forcing

    // Scratch (pre-allocated to n_batch)
    lfg_pos *pos_buf;      size_t pos_buf_cap;
    int8_t  *logits_buf;   size_t logits_buf_cap;

    // Tool ranking state (all buffers pre-allocated at register time)
    struct lfg_tool_entry {
        char    *name;                 // strdup'd
        char    *xml_text;             // strdup'd pre-formatted XML
        int32_t  xml_text_len;         // strlen of xml_text
        int32_t  token_cost;           // token count of xml_text
        float   *embedding;           // malloc'd, L2-normalized, size = n_embd
    };

    lfg_context     *tool_ctx;         // Separate context for computing tool embeddings
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
    char            *tool_xml_buf;        // malloc'd, pre-computed full XML block
    int32_t          tool_xml_buf_cap;
    lfg_token       *tool_token_buf;      // malloc'd, for tokenized XML block
    int32_t          tool_token_buf_cap;

    int32_t          tool_top_k;            // 0 = disabled
    bool             tools_injected;       // Reset on session_reset()

    // Entropy monitor config
    float                entropy_threshold;
    int32_t              entropy_cooldown;
    int32_t              entropy_tokens_since;
    float                entropy_last;
    float                entropy_last_norm;
    bool                 entropy_active;

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
    bool                 confidence_ignore_reasoning;

    // Surprise monitor (input novelty — single aggregate event per ingestion)
    float                surprise_threshold;
    bool                 surprise_active;
    bool                 surprise_ignore_reasoning;
    int32_t              surprise_skip_tokens;  // tokens to skip at start of next ingestion (chat context)

    // Accumulator (filled during ingestion, read via pop)
    int32_t              surprise_count;        // tokens above threshold
    float                surprise_sum;          // sum of surprises (for mean)
    float                surprise_max;          // max surprise
    int32_t              surprise_n_evaluated;  // total tokens evaluated
    bool                 surprise_ready;        // event available for pop
    bool                 surprise_popped;       // already consumed

    // Single embedding buffer (allocated at configure time)
    float               *surprise_embd;         // malloc'd, size = n_embd
    int32_t              surprise_n_embd;
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

// Returns malloc'd string. Caller must free.
static char * tool_format_xml(const lfg_tool_desc *tool, int32_t *out_len) {
    const char *name = tool->name ? tool->name : "";
    const char *desc = tool->description ? tool->description : "";
    const char *schema = tool->json_schema;
    bool has_schema = schema && schema[0] != '\0';

    // Calculate required size
    // <tool name="..." description="..."[ schema='...']/>
    size_t len = 13 + std::strlen(name) + 15 + std::strlen(desc) + 1; // <tool name="..." description="..."
    if (has_schema) len += 9 + std::strlen(schema) + 1;               //  schema='...'
    len += 3; // />\n

    char *buf = (char *)malloc(len + 1);
    int n;
    if (has_schema) {
        n = std::snprintf(buf, len + 1, "<tool name=\"%s\" description=\"%s\" schema='%s'/>\n",
                          name, desc, schema);
    } else {
        n = std::snprintf(buf, len + 1, "<tool name=\"%s\" description=\"%s\"/>\n", name, desc);
    }
    if (out_len) *out_len = n;
    return buf;
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
    tparams.n_threads = s->config.n_threads;
    tparams.embeddings = true;
    tparams.pooling_type = LFG_POOLING_TYPE_MEAN;
    s->tool_ctx = lfg_init_from_model(s->model, tparams);
    return s->tool_ctx != nullptr;
}

// Compute mean-pooled embedding of the current prompt into out_embd.
// Returns true on success. Zero-alloc (uses existing tool_ctx).
static bool compute_query_embedding(lfg_session *s, float *out_embd) {
    if (!s->tool_ctx || s->token_history.size == 0) return false;

    int32_t n_embd = lfg_model_n_embd(s->model);
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

    if (s->config.reasoning_budget > 0 && s->reasoning_start_count > 0 && s->reasoning_end_count > 0) {
        lfg_sampler_chain_add(s->sampler, lfg_sampler_init_reasoning_budget(
            s->config.reasoning_budget,
            s->reasoning_start_tokens, s->reasoning_start_count,
            s->reasoning_end_tokens, s->reasoning_end_count
        ));
    }

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
    if (s->surprise_active && s->surprise_ignore_reasoning &&
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

                // Skip reasoning tokens when ignore_reasoning is set
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
                if (surprise >= s->surprise_threshold) {
                    s->surprise_count++;
                    s->surprise_sum += surprise;
                    if (surprise > s->surprise_max)
                        s->surprise_max = surprise;
                }
            }
        }

        s->n_past += n;
    }

    // Post-ingest: mark event ready if any tokens exceeded threshold
    if (s->surprise_active && s->surprise_count > 0) {
        s->surprise_ready = true;
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

LFG_API lfg_session_config lfg_session_default_config(void) {
    lfg_session_config cfg{};
    cfg.n_threads = 4;
    cfg.n_ctx = 2048;
    cfg.n_batch = 512;
    cfg.enable_healing = false;
    cfg.structured_checkpointing = true;
    cfg.reasoning_budget = 0;
    cfg.max_tokens = 0;
    cfg.sampling = lfg_sampling_default_config();
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
    s->forcing_reasoning_end_index = -1;

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
    s->tool_xml_buf = nullptr;
    s->tool_xml_buf_cap = 0;
    s->tool_token_buf = nullptr;
    s->tool_token_buf_cap = 0;
    s->tool_top_k = 0;
    s->tools_injected = false;

    // Entropy monitor (all nullptr/zero — allocated in configure_entropy_monitor)
    s->entropy_threshold = 0.0f;
    s->entropy_cooldown = 1;
    s->entropy_tokens_since = 0;
    s->entropy_last = -1.0f;
    s->entropy_last_norm = -1.0f;
    s->entropy_active = false;
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
    s->confidence_ignore_reasoning = false;
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
    s->surprise_ignore_reasoning = false;
    s->surprise_skip_tokens = 0;
    s->surprise_count = 0;
    s->surprise_sum = 0.0f;
    s->surprise_max = 0.0f;
    s->surprise_n_evaluated = 0;
    s->surprise_ready = false;
    s->surprise_popped = false;
    s->surprise_embd = nullptr;
    s->surprise_n_embd = 0;

    session_rebuild_sampler(s);

    return s;
}

LFG_API void lfg_session_free(lfg_session * session) {
    if (!session) return;
    if (session->sampler) lfg_sampler_free(session->sampler);
    lfg_session_clear_tools(session);
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

    // Surprise monitor
    free(session->surprise_embd);

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
    session->forcing_reasoning_end_index = -1;
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
    }

    // Surprise monitor reset
    if (session->surprise_active) {
        session->surprise_count = 0;
        session->surprise_sum = 0.0f;
        session->surprise_max = 0.0f;
        session->surprise_n_evaluated = 0;
        session->surprise_ready = false;
        session->surprise_popped = false;
        session->surprise_skip_tokens = 0;
    }
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
// <tools>...</tools> XML block into session->tool_xml_buf.
// query_embd must be L2-normalized, length = session->tool_n_embd.
// Returns the XML text length written, or 0 if no tools / ranking failed.
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

    // Build XML block with top_k tools into pre-allocated buffer
    int32_t k = session->tool_top_k;
    if (k > session->tool_count) k = session->tool_count;
    int32_t xml_len = 0;

    const char *header = "<tools>\n";
    const char *footer = "</tools>\n";
    int32_t header_len = 8;
    int32_t footer_len = 9;

    std::memcpy(session->tool_xml_buf + xml_len, header, header_len);
    xml_len += header_len;

    for (int32_t i = 0; i < k; ++i) {
        auto &tool = session->tool_entries[session->tool_score_indices[i]];
        std::memcpy(session->tool_xml_buf + xml_len, tool.xml_text, tool.xml_text_len);
        xml_len += tool.xml_text_len;
    }

    std::memcpy(session->tool_xml_buf + xml_len, footer, footer_len);
    xml_len += footer_len;
    session->tool_xml_buf[xml_len] = '\0';

    LFG_LOG_DEBUG("tool_inject: top_k=%d xml_len=%d", k, xml_len);

    return xml_len;
}

LFG_API bool lfg_session_decode(lfg_session * session) {
    if (!session) return false;

    // Tool ranking and injection for low-level API users (ingest + decode + sample).
    // For chat_generate / prompt_generate, tools are already injected into the
    // prompt text before tokenization, so tools_injected is already true.
    if (session->tool_count > 0 && !session->tools_injected && session->tool_top_k > 0) {
        bool got_query_embd = compute_query_embedding(session, session->tool_query_embd);

        if (got_query_embd) {
            int32_t xml_len = session_rank_and_format_tools(session, session->tool_query_embd);

            if (xml_len > 0) {
                const lfg_vocab *vocab = lfg_model_get_vocab(session->model);
                int32_t n_tok = lfg_tokenize(vocab, session->tool_xml_buf, xml_len,
                                             session->tool_token_buf, session->tool_token_buf_cap,
                                             false, false);
                if (n_tok > 0) {
                    lfg_session_ingest_tokens(session, session->tool_token_buf, n_tok, false);
                }
            }
        }

        session->tools_injected = true;
    }

    return true;
}

LFG_API lfg_token lfg_session_sample(lfg_session * session) {
    if (!session || !session->ctx || !session->sampler) return 0;

    // Reasoning budget enforcement (matches old InferenceCore::Sample)
    if (session->in_reasoning && session->config.reasoning_budget > 0) {
        if (session->reasoning_token_count >= (size_t)session->config.reasoning_budget) {
            if (session->forcing_reasoning_end_index == -1) {
                session->forcing_reasoning_end_index = 0;
            }
        }
    }

    // Forced reasoning end tokens always complete before max_tokens takes effect.
    // This prevents corrupted grammar state from a mid-sequence truncation.
    if (session->forcing_reasoning_end_index >= 0) {
        if (session->forcing_reasoning_end_index < (int)session->reasoning_end_count) {
            lfg_token forced = session->reasoning_end_tokens[session->forcing_reasoning_end_index];
            session->forcing_reasoning_end_index++;
            return forced;
        } else {
            session->forcing_reasoning_end_index = -1;
        }
    }

    // Max tokens enforcement (after forced reasoning end to avoid mid-sequence truncation)
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

            // Threshold + cooldown gate
            if (norm >= session->entropy_threshold &&
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
            bool conf_skip = session->confidence_ignore_reasoning && session->in_reasoning;
            if (!conf_skip && norm <= session->confidence_threshold) {
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
        free(s->tool_entries[i].xml_text);
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
        int32_t n_embd = lfg_model_n_embd(session->model);

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

        // Total XML size for pre-allocating decode-time buffer
        int32_t total_xml_bytes = 0;

        for (int32_t i = 0; i < n_tools; ++i) {
            auto &entry = session->tool_entries[i];
            int32_t xml_len = 0;
            entry.xml_text = tool_format_xml(&tools[i], &xml_len);
            entry.xml_text_len = xml_len;
            entry.name = strdup(tools[i].name ? tools[i].name : "");
            total_xml_bytes += xml_len;

            uint64_t hash = fnv1a_hash(entry.xml_text, xml_len);

            // Token cost
            if (xml_len + 16 > tok_scratch_cap) {
                tok_scratch_cap = xml_len + 16;
                tok_scratch = (lfg_token *)realloc(tok_scratch, tok_scratch_cap * sizeof(lfg_token));
            }
            int32_t n_tok = lfg_tokenize(vocab, entry.xml_text, xml_len,
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

                if (xml_len + 16 > tok_scratch_cap) {
                    tok_scratch_cap = xml_len + 16;
                    tok_scratch = (lfg_token *)realloc(tok_scratch, tok_scratch_cap * sizeof(lfg_token));
                }
                int32_t n_emb_tok = lfg_tokenize(vocab, entry.xml_text, xml_len,
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
        // XML buffer: <tools>\n + all tool XML + </tools>\n + NUL
        int32_t xml_buf_cap = total_xml_bytes + 32;
        session->tool_xml_buf = (char *)realloc(session->tool_xml_buf, xml_buf_cap);
        session->tool_xml_buf_cap = xml_buf_cap;

        // Token buffer: worst case ~ xml_buf_cap tokens (1 byte = 1 token)
        int32_t token_buf_cap = xml_buf_cap + 32;
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
    free(session->tool_xml_buf);       session->tool_xml_buf = nullptr;
    session->tool_xml_buf_cap = 0;
    free(session->tool_token_buf);     session->tool_token_buf = nullptr;
    session->tool_token_buf_cap = 0;
    session->tool_n_embd = 0;
}

// ---------------------------------------------------------------------------
// Entropy Monitor API
// ---------------------------------------------------------------------------

LFG_API lfg_entropy_monitor_config lfg_entropy_monitor_default_config(void) {
    lfg_entropy_monitor_config cfg{};
    cfg.threshold = 0.7f;
    cfg.cooldown_tokens = 16;
    cfg.ring_size = 4;
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
    int32_t n_embd = lfg_model_n_embd(session->model);

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
    int32_t n_embd = lfg_model_n_embd(session->model);

    // Free previous allocations
    free(session->confidence_slots);
    free(session->confidence_embd_pool);

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
    session->confidence_ignore_reasoning = config->ignore_reasoning;
    session->confidence_active       = true;

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

    int32_t n_embd = lfg_model_n_embd(session->model);

    // Free previous allocation
    free(session->surprise_embd);

    // Single embedding buffer
    session->surprise_embd = (float *)calloc(n_embd, sizeof(float));

    session->surprise_n_embd       = n_embd;
    session->surprise_threshold    = config->threshold;
    session->surprise_count        = 0;
    session->surprise_sum          = 0.0f;
    session->surprise_max          = 0.0f;
    session->surprise_n_evaluated  = 0;
    session->surprise_ready        = false;
    session->surprise_popped       = false;
    session->surprise_ignore_reasoning = config->ignore_reasoning;
    session->surprise_skip_tokens  = 0;
    session->surprise_active       = true;

    return n_embd;
}

LFG_API bool lfg_session_surprise_pop(lfg_session * session,
                                       lfg_surprise_event * event_out,
                                       float * embd_out, int32_t embd_cap) {
    if (!session || !session->surprise_active) return false;
    if (!session->surprise_ready || session->surprise_popped) return false;

    // Build the aggregate event
    lfg_surprise_event ev{};
    ev.mean_surprise      = (session->surprise_count > 0)
        ? (session->surprise_sum / session->surprise_count) : 0.0f;
    ev.max_surprise       = session->surprise_max;
    ev.n_above_threshold  = session->surprise_count;
    ev.n_tokens_evaluated = session->surprise_n_evaluated;
    ev.n_embd             = session->surprise_n_embd;

    // Lazy embedding: compute on pop if caller wants it
    if (embd_out && embd_cap >= session->surprise_n_embd && session->surprise_embd) {
        bool got = compute_query_embedding(session, session->surprise_embd);
        if (got) {
            int32_t n = session->surprise_n_embd < embd_cap ? session->surprise_n_embd : embd_cap;
            std::memcpy(embd_out, session->surprise_embd, n * sizeof(float));
        } else {
            ev.n_embd = 0;
        }
    }

    if (event_out) *event_out = ev;

    session->surprise_popped = true;
    return true;
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

    int32_t n_embd = lfg_model_n_embd(session->model);
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

// ---------------------------------------------------------------------------
// Generate Loop API
// ---------------------------------------------------------------------------

LFG_API lfg_generate_config lfg_generate_default_config(void) {
    lfg_generate_config cfg{};
    cfg.max_tokens = 0;
    cfg.token_cb = nullptr;
    cfg.token_cb_data = nullptr;
    cfg.entropy_cb = nullptr;
    cfg.entropy_cb_data = nullptr;
    cfg.confidence_cb = nullptr;
    cfg.confidence_cb_data = nullptr;
    cfg.surprise_cb = nullptr;
    cfg.surprise_cb_data = nullptr;
    return cfg;
}

LFG_API lfg_generate_result lfg_session_generate(
    lfg_session * session, lfg_generate_config config)
{
    lfg_generate_result result{};
    if (!session) return result;

    const lfg_vocab *vocab = lfg_model_get_vocab(session->model);
    int32_t max_tokens = config.max_tokens > 0
        ? config.max_tokens
        : (session->config.max_tokens > 0 ? session->config.max_tokens : 4096);

    char piece_buf[256];
    bool stopped = false;

    // Pre-allocate embedding buffer for entropy callbacks (zero-alloc in hot path)
    float *embd_buf = nullptr;
    if (config.entropy_cb && session->entropy_active && session->entropy_n_embd > 0) {
        embd_buf = (float *)alloca(session->entropy_n_embd * sizeof(float));
    }

    // Pre-allocate embedding buffer for confidence callbacks
    float *conf_embd_buf = nullptr;
    if (config.confidence_cb && session->confidence_active && session->confidence_n_embd > 0) {
        conf_embd_buf = (float *)alloca(session->confidence_n_embd * sizeof(float));
    }

    // Drain surprise event from prompt ingestion (before generation loop)
    if (config.surprise_cb && session->surprise_active && session->surprise_ready && !session->surprise_popped) {
        float *surp_embd_buf = nullptr;
        if (session->surprise_n_embd > 0) {
            surp_embd_buf = (float *)alloca(session->surprise_n_embd * sizeof(float));
        }
        lfg_surprise_event sev;
        if (lfg_session_surprise_pop(session, &sev, surp_embd_buf, session->surprise_n_embd)) {
            config.surprise_cb(&sev, sev.n_embd > 0 ? surp_embd_buf : nullptr,
                               config.surprise_cb_data);
            result.n_surprise_events = 1;
        }
    } else if (session->surprise_active) {
        // No callback — mark as consumed
        session->surprise_popped = true;
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

    for (int32_t i = 0; i < max_tokens; ++i) {
        lfg_session_decode(session);
        lfg_token tok = lfg_session_sample(session);

        if (lfg_vocab_is_eog(vocab, tok)) {
            // Flush buffered tokens, discarding stop sequence prefix tokens
            if (stop_buf_count > 0) {
                int32_t discard = (session->last_stop_len > 1)
                    ? (session->last_stop_len - 1) : 0;
                int32_t flush_n = stop_buf_count - discard;
                if (flush_n < 0) flush_n = 0;
                for (int32_t j = 0; j < flush_n; j++) {
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
            stopped = true;
            break;
        }

        // Entropy callback — pop with embedding, let callback pick text to inject.
        if (config.entropy_cb && session->entropy_active) {
            lfg_entropy_event ev;
            if (lfg_session_entropy_pop(session, &ev, embd_buf, session->entropy_n_embd)) {
                const float *embd_ptr = ev.n_embd > 0 ? embd_buf : nullptr;
                const char *inject = config.entropy_cb(&ev, embd_ptr, config.entropy_cb_data);
                if (inject) {
                    // Rewind to checkpoint, tokenize injected text, ingest, continue
                    if (lfg_session_rewind(session, ev.checkpoint_id)) {
                        int32_t len = (int32_t)std::strlen(inject);
                        int32_t tok_cap = len + 16;
                        lfg_token *inject_toks = (lfg_token *)alloca(tok_cap * sizeof(lfg_token));
                        int32_t n_inj = lfg_tokenize(vocab, inject, len,
                                                      inject_toks, tok_cap, false, false);
                        if (n_inj < 0) {
                            // Need more space — fall back to malloc
                            tok_cap = -n_inj;
                            lfg_token *heap_toks = (lfg_token *)malloc(tok_cap * sizeof(lfg_token));
                            n_inj = lfg_tokenize(vocab, inject, len,
                                                  heap_toks, tok_cap, false, false);
                            if (n_inj > 0) {
                                lfg_session_ingest_tokens(session, heap_toks, n_inj, false);
                            }
                            free(heap_toks);
                        } else if (n_inj > 0) {
                            lfg_session_ingest_tokens(session, inject_toks, n_inj, false);
                        }
                        result.n_retrievals++;
                        lfg_session_entropy_flush(session);
                        stop_buf_count = 0;  // Rewound — buffered tokens are invalid
                        continue;  // re-decode from injected position
                    }
                }
                lfg_session_entropy_flush(session);
            }
        } else if (session->entropy_active) {
            lfg_session_entropy_flush(session);
        }

        // Confidence callback — informational, no rewind
        if (config.confidence_cb && session->confidence_active) {
            lfg_confidence_event cev;
            while (lfg_session_confidence_pop(session, &cev, conf_embd_buf, session->confidence_n_embd)) {
                const float *cembd_ptr = cev.n_embd > 0 ? conf_embd_buf : nullptr;
                config.confidence_cb(&cev, cembd_ptr, config.confidence_cb_data);
                result.n_confidence_spans++;
            }
        }

        result.n_tokens = i + 1;

        // Token callback with stop look-ahead buffering (token + text level)
        if (stop_buf_cap > 0) {
            // Convert token to piece text
            int32_t n = lfg_token_to_piece(vocab, tok, piece_buf, sizeof(piece_buf), 0, false);
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

                    // Flush safe whole tokens
                    for (int32_t j = 0; j < flush_n && !stopped; j++) {
                        lfg_generate_action action = config.token_cb(
                            stop_buf[j].token, stop_buf[j].piece,
                            stop_buf[j].piece_len, config.token_cb_data);
                        if (action == LFG_GENERATE_STOP) {
                            result.stop_reason = LFG_STOP_CALLBACK;
                            stopped = true;
                        }
                    }

                    // Emit safe prefix of partially-matched boundary token
                    if (!stopped && flush_n < stop_buf_count && chars_so_far < emit_chars) {
                        int32_t safe_chars = emit_chars - chars_so_far;
                        lfg_generate_action action = config.token_cb(
                            stop_buf[flush_n].token,
                            stop_buf[flush_n].piece,
                            safe_chars, config.token_cb_data);
                        if (action == LFG_GENERATE_STOP) {
                            result.stop_reason = LFG_STOP_CALLBACK;
                            stopped = true;
                        }
                    }

                    if (!stopped) result.stop_reason = LFG_STOP_EOS;
                    stopped = true;
                    stop_buf_count = 0;
                    break;
                }
                if (stopped) break;
            }

            // If buffer exceeds capacity, emit oldest token (confirmed safe)
            while (stop_buf_count > stop_buf_cap) {
                lfg_generate_action action = config.token_cb(
                    stop_buf[0].token, stop_buf[0].piece,
                    stop_buf[0].piece_len, config.token_cb_data);
                if (action == LFG_GENERATE_STOP) {
                    result.stop_reason = LFG_STOP_CALLBACK;
                    stopped = true;
                    break;
                }
                if (stop_buf_count > 1) {
                    std::memmove(stop_buf, stop_buf + 1,
                                 (stop_buf_count - 1) * sizeof(stop_slot));
                }
                stop_buf_count--;
            }
            if (stopped) break;
        } else if (config.token_cb) {
            // No buffering needed — emit directly
            int32_t n = lfg_token_to_piece(vocab, tok, piece_buf, sizeof(piece_buf), 0, false);
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
    if (stop_buf_count > 0 && !stopped) {
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
        result.stop_reason = LFG_STOP_MAX_TOKENS;
    }

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

    // Drain any remaining confidence events through the callback
    if (config.confidence_cb && session->confidence_active) {
        lfg_confidence_event cev;
        while (lfg_session_confidence_pop(session, &cev, conf_embd_buf, session->confidence_n_embd)) {
            const float *cembd_ptr = cev.n_embd > 0 ? conf_embd_buf : nullptr;
            config.confidence_cb(&cev, cembd_ptr, config.confidence_cb_data);
            result.n_confidence_spans++;
        }
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

    // Tool injection: rank tools and prepend XML to the prompt text so tools
    // appear inside the prompt context, not between prompt and generated output.
    const char *effective_prompt = prompt;
    int32_t effective_len = prompt_len;
    char *combined_prompt = nullptr;

    if (session->tool_count > 0 && !session->tools_injected && session->tool_top_k > 0) {
        int32_t n_embd = session->tool_n_embd;
        int32_t got = lfg_session_embed(session, prompt, prompt_len,
                                         session->tool_query_embd, n_embd);
        if (got > 0) {
            int32_t xml_len = session_rank_and_format_tools(session, session->tool_query_embd);
            if (xml_len > 0) {
                // Prepend tool XML + newline to prompt
                int32_t combined_len = xml_len + 1 + prompt_len;
                combined_prompt = (char *)malloc(combined_len + 1);
                std::memcpy(combined_prompt, session->tool_xml_buf, xml_len);
                combined_prompt[xml_len] = '\n';
                std::memcpy(combined_prompt + xml_len + 1, prompt, prompt_len);
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

    // 1. Tool injection: rank tools and build XML to inject into the system message.
    //    This must happen BEFORE template application so the tools appear inside
    //    the prompt (system message), not between the prompt and generated output.
    int32_t tool_xml_len = 0;
    char *tool_xml_text = nullptr;

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
                tool_xml_len = session_rank_and_format_tools(session, session->tool_query_embd);
                if (tool_xml_len > 0) {
                    tool_xml_text = session->tool_xml_buf;
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

    if (tool_xml_text && tool_xml_len > 0) {
        // Find existing system message
        int sys_idx = -1;
        for (size_t i = 0; i < n_messages; ++i) {
            if (messages[i].role && std::strcmp(messages[i].role, "system") == 0) {
                sys_idx = (int)i;
                break;
            }
        }

        if (sys_idx >= 0) {
            // Append tool XML to existing system message
            const char *orig = messages[sys_idx].content ? messages[sys_idx].content : "";
            size_t orig_len = std::strlen(orig);
            size_t buf_len = orig_len + 2 + tool_xml_len + 1;
            sys_content_buf = (char *)malloc(buf_len);
            int written = std::snprintf(sys_content_buf, buf_len, "%s\n\n%.*s",
                                        orig, (int)tool_xml_len, tool_xml_text);
            sys_content_buf[written] = '\0';

            mod_msgs = (lfg_chat_message *)malloc(n_messages * sizeof(lfg_chat_message));
            std::memcpy(mod_msgs, messages, n_messages * sizeof(lfg_chat_message));
            mod_msgs[sys_idx].content = sys_content_buf;

            tmpl_msgs = mod_msgs;
            tmpl_n = n_messages;
        } else {
            // Insert a new system message at the beginning with the tool XML
            size_t new_n = n_messages + 1;
            sys_content_buf = (char *)malloc(tool_xml_len + 1);
            std::memcpy(sys_content_buf, tool_xml_text, tool_xml_len);
            sys_content_buf[tool_xml_len] = '\0';

            mod_msgs = (lfg_chat_message *)malloc(new_n * sizeof(lfg_chat_message));
            mod_msgs[0].role = "system";
            mod_msgs[0].content = sys_content_buf;
            std::memcpy(mod_msgs + 1, messages, n_messages * sizeof(lfg_chat_message));

            tmpl_msgs = mod_msgs;
            tmpl_n = new_n;
        }
    }

    // 3. Detect chat template from model metadata
    const char *tmpl_str = lfg_model_chat_template(session->model, nullptr);

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
                                       prefix_toks, prefix_tok_cap, true, false);
            if (pn < 0) {
                prefix_tok_cap = -pn;
                prefix_toks = (lfg_token *)realloc(prefix_toks, prefix_tok_cap * sizeof(lfg_token));
                pn = lfg_tokenize(vocab, prefix_buf, prefix_needed,
                                   prefix_toks, prefix_tok_cap, true, false);
            }
            session->surprise_skip_tokens = pn > 0 ? pn : 0;
            free(prefix_toks);
            free(prefix_buf);
        }
    }

    // 4. Apply template: first call with NULL buf to get required size
    int32_t needed = lfg_chat_apply_template(tmpl_str, tmpl_msgs, tmpl_n, true, nullptr, 0);
    if (needed <= 0) {
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

    free(mod_msgs);
    free(sys_content_buf);

    // 5. Tokenize
    const lfg_vocab *vocab = lfg_model_get_vocab(session->model);
    int32_t tok_cap = needed + 16;
    lfg_token *tokens = (lfg_token *)malloc(tok_cap * sizeof(lfg_token));
    int32_t n = lfg_tokenize(vocab, formatted, needed, tokens, tok_cap, true, false);
    if (n < 0) {
        tok_cap = -n;
        tokens = (lfg_token *)realloc(tokens, tok_cap * sizeof(lfg_token));
        n = lfg_tokenize(vocab, formatted, needed, tokens, tok_cap, true, false);
    }
    free(formatted);

    if (n <= 0) {
        free(tokens);
        lfg_set_last_error(LFG_ERROR_INVALID_ARGUMENT,
            "%s: tokenization failed", __func__);
        return result;
    }

    // 6. Ingest prompt — update_sampler=false so grammar isn't fed prompt tokens
    lfg_session_ingest_tokens(session, tokens, n, false);
    session->surprise_skip_tokens = 0;  // consumed
    free(tokens);

    // 7. Auto-configure text stop string for the EOS token's text representation.
    //    The generate loop already stops on the special EOS token, but small
    //    models sometimes generate the TEXT version (e.g. "<|im_end|>" spelled
    //    out as regular tokens) which doesn't trigger EOG detection.  Using a
    //    text-level stop string catches the text form regardless of how the
    //    tokenizer splits it.
    if (session->stop_text_count == 0) {
        lfg_token eos = lfg_vocab_eos(vocab);
        char eos_text[64];
        int32_t eos_len = lfg_token_to_piece(vocab, eos, eos_text, sizeof(eos_text), 0, true);
        if (eos_len > 0) {
            eos_text[eos_len] = '\0';
            // Models sometimes generate a "closing tag" variant of the EOS
            // token text, e.g. "</|im_end|>" instead of "<|im_end|>".  Add
            // both variants so the text stop catches either form.
            if (eos_text[0] == '<' && eos_len + 1 < (int32_t)sizeof(eos_text)) {
                char close_text[66];
                close_text[0] = '<';
                close_text[1] = '/';
                std::memcpy(close_text + 2, eos_text + 1, eos_len - 1);
                close_text[eos_len + 1] = '\0';
                const char *strs[] = { eos_text, close_text };
                lfg_session_configure_stop_strings(session, strs, 2);
            } else {
                const char *strs[] = { eos_text };
                lfg_session_configure_stop_strings(session, strs, 1);
            }
        }
    }

    // 8. Generate
    return lfg_session_generate(session, config);
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
