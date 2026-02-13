# Auto Entropy Monitor — Adaptive Threshold Gating

## Context

The entropy monitor fires when normalized entropy exceeds a fixed threshold. Different models have very different baseline entropy distributions — a 350M model may cruise at 0.5 normalized entropy while a 1.2B sits at 0.3. A fixed threshold of 0.7 fires constantly on one and never on the other. An AUTO mode that adapts to the model's running baseline eliminates per-model tuning.

## API

```c
typedef enum {
    LFG_ENTROPY_GATE_OFF   = 0,  // Disabled (no entropy events)
    LFG_ENTROPY_GATE_FIXED = 1,  // Fire when norm >= threshold (current behavior, default)
    LFG_ENTROPY_GATE_AUTO  = 2,  // Fire when norm >= running_mean + threshold
} lfg_entropy_gate_mode;
```

On `lfg_entropy_monitor_config`:
```c
lfg_entropy_gate_mode gate_mode;  // 0 = OFF, 1 = FIXED (default), 2 = AUTO
// existing `threshold` field is reused:
// FIXED: absolute normalized entropy threshold (current behavior)
// AUTO: gap above running mean. 0 = use default (0.15)
```

| Mode | `threshold` | Behavior |
|---|---|---|
| `OFF` | ignored | No entropy events fired |
| `FIXED` | `0.7` (example) | Fire when `norm >= threshold` (backward compat) |
| `AUTO` | `0.15` (default) | Fire when `norm >= running_mean + threshold` |

## Files to modify

1. `src/inference/lfg_api.h` — enum + field on `lfg_entropy_monitor_config`
2. `src/inference/lfg_api.cpp` — session state, configure, hot path gate

## Implementation

### 1. Header (lfg_api.h)

Add enum before `lfg_entropy_monitor_config`:
```c
typedef enum {
    LFG_ENTROPY_GATE_OFF   = 0,
    LFG_ENTROPY_GATE_FIXED = 1,
    LFG_ENTROPY_GATE_AUTO  = 2,
} lfg_entropy_gate_mode;
```

Add to `lfg_entropy_monitor_config`:
```c
lfg_entropy_gate_mode gate_mode;  // Gating mode. 0 = OFF, 1 = FIXED (default), 2 = AUTO.
```

### 2. Engine (lfg_api.cpp)

**Session struct** — add running stats next to `entropy_threshold`:
```c
lfg_entropy_gate_mode entropy_gate_mode;
float                entropy_running_sum;     // sum of all normalized entropies
int32_t              entropy_running_count;   // number of samples seen
```

**Default config** (`lfg_entropy_monitor_default_config()`):
```c
cfg.gate_mode = LFG_ENTROPY_GATE_FIXED;  // backward compat
```

**Configure** (`lfg_session_configure_entropy_monitor()`):
```c
session->entropy_gate_mode = config->gate_mode;
session->entropy_running_sum = 0.0f;
session->entropy_running_count = 0;
```

**Hot path** (`lfg_session_sample()`, entropy gate at line ~1534):

Replace:
```c
if (norm >= session->entropy_threshold &&
    session->entropy_tokens_since >= session->entropy_cooldown) {
```

With:
```c
// Always update running stats for AUTO mode
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

if (entropy_fires && session->entropy_tokens_since >= session->entropy_cooldown) {
```

Note: `entropy_running_count >= 2` ensures at least 2 samples before AUTO can fire (warmup). The running mean includes ALL tokens (not just above-threshold ones) so it tracks the model's true baseline.

### 3. Session reset

In `lfg_session_reset()` — reset running stats:
```c
session->entropy_running_sum = 0.0f;
session->entropy_running_count = 0;
```

### 4. Demo UI (optional, separate from this plan)

In Settings > Monitors > Entropy, add a Combo for gate mode similar to the tool score gate.

## Backward Compatibility

- Default `gate_mode = FIXED` preserves existing behavior exactly
- `threshold` field semantics change only when `gate_mode = AUTO`
- Zero-init structs (calloc) get `gate_mode = 0 = OFF`, but that matches `threshold = 0 = disabled` — both mean no events

## Performance

- AUTO adds 1 float add + 1 int increment per sample (trivial)
- 1 float divide only when checking the gate (already gated by cooldown)
- No allocations, no branches in the common case
