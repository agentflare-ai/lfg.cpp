# Auto Surprise Monitor — Adaptive Threshold Gating

## Context

The surprise monitor computes per-token surprise during prompt ingestion and fires a single aggregate event if any tokens exceed a fixed threshold. Surprise is already somewhat self-calibrating (it's `-log P(token) / log(n_vocab)`, normalized by vocab size), but the absolute scale still varies by model architecture and training data. An AUTO mode compares each token's surprise against the running mean across the prompt, catching genuinely novel tokens regardless of the model's baseline surprise level.

## API

```c
typedef enum {
    LFG_SURPRISE_GATE_OFF   = 0,  // Disabled (no surprise events)
    LFG_SURPRISE_GATE_FIXED = 1,  // Token surprising when surprise >= threshold (current, default)
    LFG_SURPRISE_GATE_AUTO  = 2,  // Token surprising when surprise >= running_mean + threshold
} lfg_surprise_gate_mode;
```

On `lfg_surprise_monitor_config`:
```c
lfg_surprise_gate_mode gate_mode;  // 0 = OFF, 1 = FIXED (default), 2 = AUTO
// existing `threshold` field is reused:
// FIXED: absolute normalized surprise floor (current behavior)
// AUTO: gap above running mean. 0 = use default (0.20)
```

| Mode | `threshold` | Behavior |
|---|---|---|
| `OFF` | ignored | No surprise events |
| `FIXED` | `0.5` (example) | Surprising when `surprise >= threshold` |
| `AUTO` | `0.20` (default) | Surprising when `surprise >= prompt_mean + threshold` |

## Difference from Entropy/Confidence

Surprise operates on **prompt ingestion** (a batch of tokens processed at once), not on the **generate loop** (one token at a time). This means:

1. We see all tokens in one pass — we could do a **two-pass approach**: first compute all surprises, then compute the mean, then flag outliers. This is cleaner than a running mean since the full prompt is available.
2. The running mean naturally covers the full prompt context, not just a warmup window.

The two-pass approach is preferred because it avoids the warmup problem entirely and produces more stable results (the mean isn't biased by token order).

## Files to modify

1. `src/inference/lfg_api.h` — enum + field on `lfg_surprise_monitor_config`
2. `src/inference/lfg_api.cpp` — session state, configure, ingestion gate

## Implementation

### 1. Header (lfg_api.h)

Add enum before `lfg_surprise_monitor_config`:
```c
typedef enum {
    LFG_SURPRISE_GATE_OFF   = 0,
    LFG_SURPRISE_GATE_FIXED = 1,
    LFG_SURPRISE_GATE_AUTO  = 2,
} lfg_surprise_gate_mode;
```

Add to `lfg_surprise_monitor_config`:
```c
lfg_surprise_gate_mode gate_mode;  // Gating mode. 0 = OFF, 1 = FIXED (default), 2 = AUTO.
```

### 2. Engine (lfg_api.cpp)

**Session struct** — add near surprise fields:
```c
lfg_surprise_gate_mode surprise_gate_mode;
float               *surprise_per_token;     // malloc'd scratch for AUTO two-pass
int32_t              surprise_per_token_cap;  // capacity of above
```

**Default config** (`lfg_surprise_monitor_default_config()`):
```c
cfg.gate_mode = LFG_SURPRISE_GATE_FIXED;
```

**Configure** (`lfg_session_configure_surprise_monitor()`):
```c
session->surprise_gate_mode = config->gate_mode;
// Pre-allocate scratch for AUTO (realloc'd if prompt exceeds capacity)
if (config->gate_mode == LFG_SURPRISE_GATE_AUTO) {
    session->surprise_per_token_cap = 512;  // reasonable default
    session->surprise_per_token = (float *)malloc(512 * sizeof(float));
}
```

**Ingestion** (`session_ingest_internal()`, surprise computation at line ~758):

For FIXED mode, the current logic is unchanged — accumulate on the fly:
```c
if (surprise >= s->surprise_threshold) {
    s->surprise_count++;
    s->surprise_sum += surprise;
    if (surprise > s->surprise_max) s->surprise_max = surprise;
}
```

For AUTO mode, use two-pass:

**Pass 1** — store all surprise values (replace the accumulate block):
```c
if (s->surprise_gate_mode == LFG_SURPRISE_GATE_AUTO) {
    // Store for second pass
    int32_t idx = s->surprise_n_evaluated;  // already incremented above
    if (idx >= s->surprise_per_token_cap) {
        s->surprise_per_token_cap = idx * 2;
        s->surprise_per_token = (float *)realloc(s->surprise_per_token,
            s->surprise_per_token_cap * sizeof(float));
    }
    s->surprise_per_token[idx - 1] = surprise;  // -1 because n_evaluated was pre-incremented
} else {
    // FIXED: accumulate on the fly (existing logic)
    if (surprise >= s->surprise_threshold) {
        s->surprise_count++;
        s->surprise_sum += surprise;
        if (surprise > s->surprise_max) s->surprise_max = surprise;
    }
}
```

**Pass 2** — after the ingestion loop, before marking `surprise_ready`:
```c
if (s->surprise_active && s->surprise_gate_mode == LFG_SURPRISE_GATE_AUTO &&
    s->surprise_n_evaluated > 0) {
    // Compute mean
    float sum = 0.0f;
    for (int32_t i = 0; i < s->surprise_n_evaluated; ++i) {
        sum += s->surprise_per_token[i];
    }
    float mean = sum / s->surprise_n_evaluated;
    float gap = s->surprise_threshold > 0.0f ? s->surprise_threshold : 0.20f;
    float effective_threshold = mean + gap;

    // Flag outliers
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

if (s->surprise_active && s->surprise_count > 0) {
    s->surprise_ready = true;
}
```

### 3. Cleanup

In `lfg_session_free()` and configure (disable path):
```c
free(session->surprise_per_token);
session->surprise_per_token = nullptr;
session->surprise_per_token_cap = 0;
```

### 4. Session reset

```c
session->surprise_n_evaluated = 0;
// surprise_per_token buffer kept allocated (reused across turns)
```

## Backward Compatibility

- Default `gate_mode = FIXED` preserves existing behavior exactly
- Two-pass only runs when `gate_mode = AUTO`
- Zero-init (calloc) gives `gate_mode = 0 = OFF` — matches disabled

## Performance

- FIXED mode: unchanged (single-pass accumulate)
- AUTO mode: stores N floats (prompt length), then two O(N) passes
- N is typically 100-2000 tokens — negligible vs the decode cost of computing logits for all positions
- The `surprise_per_token` scratch buffer is allocated once and reused across turns

## Design Decision: Two-Pass vs Running Mean

Unlike entropy/confidence (which see tokens one at a time during generation), surprise processes the full prompt in one batch. Two-pass is strictly better:
- No warmup bias (early tokens don't skew the threshold)
- Mean is computed over the exact token set being evaluated
- Outliers are identified against the true distribution, not a partial view
- Same O(N) complexity (one extra linear scan)
