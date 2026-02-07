#include "lfg_api.h"

#include "json_schema_to_grammar.h"
#include "lfg_impl.h"

#include <algorithm>
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

    // Stop sequences (flat storage)
    lfg_token *stop_flat;          // all sequences concatenated
    size_t    *stop_offsets;       // start offset of each sequence
    size_t    *stop_lengths;       // length of each sequence
    size_t     stop_count;         // number of sequences
    size_t     stop_max_len;       // max sequence length (for lookback)

    // Reasoning state tracking (mirrors old InferenceCore fields)
    bool   in_reasoning;
    size_t reasoning_token_count;
    int    forcing_reasoning_end_index;  // -1 = not forcing

    // Scratch (pre-allocated to n_batch)
    lfg_pos *pos_buf;      size_t pos_buf_cap;
    int8_t  *logits_buf;   size_t logits_buf_cap;
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

static bool session_ingest_internal(lfg_session *s, const lfg_token *tokens, size_t n_tokens, bool update_sampler) {
    if (!s->ctx) return false;

    int32_t n_batch = s->config.n_batch;
    if (n_batch <= 0) n_batch = 512;

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
        batch.logits = nullptr;

        if (lfg_decode(s->ctx, batch) != 0) {
            spdlog::error("lfg_decode failed");
            return false;
        }
        s->n_past += n;
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

    session_rebuild_sampler(s);

    return s;
}

LFG_API void lfg_session_free(lfg_session * session) {
    if (!session) return;
    if (session->sampler) lfg_sampler_free(session->sampler);
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
    free(session->pos_buf);
    free(session->logits_buf);

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

LFG_API bool lfg_session_decode(lfg_session * session) {
    (void)session;
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
                return lfg_vocab_eos(lfg_model_get_vocab(session->model));
            }

            // Check preceding tokens in history
            size_t hist_len = session->token_history.size;
            if (hist_len < slen - 1) continue;

            const lfg_token *tail = session->token_history.data + hist_len - (slen - 1);
            if (std::memcmp(tail, seq, (slen - 1) * sizeof(lfg_token)) == 0) {
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
