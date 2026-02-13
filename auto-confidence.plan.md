# Auto Confidence Monitor — Adaptive Threshold Gating

## Context

The confidence monitor detects sustained low-entropy spans (model is "confident") for store/cache candidates. The threshold is a fixed ceiling — tokens with normalized entropy below it count as confident. But "low entropy" is model-relative: a 350M that naturally runs at 0.15 baseline vs a 1.2B at 0.05. An AUTO mode fires spans where entropy drops significantly *below* the running mean, adapting to each model's natural floor.

## API

```c
typedef enum {
    LFG_CONFIDENCE_GATE_OFF   = 0,  // Disabled (no confidence events)
    LFG_CONFIDENCE_GATE_FIXED = 1,  // Token confident when norm <= threshold (current, default)
    LFG_CONFIDENCE_GATE_AUTO  = 2,  // Token confident when norm <= running_mean - threshold
} lfg_confidence_gate_mode;
```

On `lfg_confidence_monitor_config`:
```c
lfg_confidence_gate_mode gate_mode;  // 0 = OFF, 1 = FIXED (default), 2 = AUTO
// existing `threshold` field is reused:
// FIXED: absolute normalized entropy ceiling (current behavior)
// AUTO: gap below running mean. 0 = use default (0.10)
```

| Mode | `threshold` | Behavior |
|---|---|---|
| `OFF` | ignored | No confidence events |
| `FIXED` | `0.3` (example) | Confident when `norm <= threshold` |
| `AUTO` | `0.10` (default) | Confident when `norm <= running_mean - threshold` |

## Files to modify

1. `src/inference/lfg_api.h` — enum + field on `lfg_confidence_monitor_config`
2. `src/inference/lfg_api.cpp` — session state, configure, hot path gate

## Implementation

### 1. Header (lfg_api.h)

Add enum before `lfg_confidence_monitor_config`:
```c
typedef enum {
    LFG_CONFIDENCE_GATE_OFF   = 0,
    LFG_CONFIDENCE_GATE_FIXED = 1,
    LFG_CONFIDENCE_GATE_AUTO  = 2,
} lfg_confidence_gate_mode;
```

Add to `lfg_confidence_monitor_config`:
```c
lfg_confidence_gate_mode gate_mode;  // Gating mode. 0 = OFF, 1 = FIXED (default), 2 = AUTO.
```

### 2. Engine (lfg_api.cpp)

**Session struct** — add running stats next to `confidence_threshold`:
```c
lfg_confidence_gate_mode confidence_gate_mode;
float                    confidence_running_sum;
int32_t                  confidence_running_count;
```

Note: confidence shares the softmax computation with entropy. The running stats here track the *same* normalized entropy values. If both entropy and confidence AUTO are active, they maintain independent running stats (they may diverge due to reasoning-skip gating on confidence).

**Default config** (`lfg_confidence_monitor_default_config()`):
```c
cfg.gate_mode = LFG_CONFIDENCE_GATE_FIXED;
```

**Configure** (`lfg_session_configure_confidence_monitor()`):
```c
session->confidence_gate_mode = config->gate_mode;
session->confidence_running_sum = 0.0f;
session->confidence_running_count = 0;
```

**Hot path** (`lfg_session_sample()`, confidence gate at line ~1576):

Replace:
```c
if (!conf_skip && norm <= session->confidence_threshold) {
```

With:
```c
// Update running stats (skip reasoning tokens if configured)
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
```

And change the `} else {` (run broken) accordingly to just:
```c
} else {
```

Warmup is `>= 5` tokens (same as default `min_span`) — need a reasonable sample before the running mean is meaningful.

### 3. Session reset

In `lfg_session_reset()`:
```c
session->confidence_running_sum = 0.0f;
session->confidence_running_count = 0;
```

## Backward Compatibility

- Default `gate_mode = FIXED` preserves existing behavior exactly
- Zero-init (calloc) gives `gate_mode = 0 = OFF` which matches disabled state
- The `conf_skip` reasoning gate still applies — AUTO only tracks non-reasoning tokens

## Performance

- Same as entropy AUTO: 1 add + 1 increment per sample, 1 divide at gate check
- Running stats independent from entropy's (confidence may skip reasoning tokens)

## Edge Case: AUTO + short generations

If the model generates fewer than 5 tokens, AUTO never fires (warmup not met). This is acceptable — short outputs don't produce meaningful spans anyway. The `min_span` default is 5, so FIXED mode also wouldn't fire a span in under 5 tokens.
