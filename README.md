# lfg.cpp

A session-centric C inference engine for GGUF language models. Built on a vendored `ggml` backend with a pure C11 API, lfg.cpp moves the generate loop, tool execution, and monitor signal production into the engine — so consumers avoid wiring decode/sample/repeat loops by hand.

## What makes it different

**The generate loop lives in the engine, not in your code.** Most inference libraries expose low-level context + batch primitives and leave the generate loop, chat templating, stop sequence handling, and token stitching to the caller. lfg.cpp wraps all of that behind opaque session handles:

```c
lfg_chat_message messages[] = {
    { "system", "You are a helpful assistant." },
    { "user",   "What is the capital of France?" },
};

lfg_generate_config gen = lfg_generate_default_config();
gen.token_cb = my_streaming_callback;

lfg_generate_result result = lfg_session_chat_generate(session, messages, 2, gen);
```

Three levels of abstraction — `lfg_session_generate` (raw state), `lfg_session_prompt_generate` (text), and `lfg_session_chat_generate` (chat template) — so you pick the level you need without reimplementing the rest.

**RAG signals are computed during generation, not after.** Most RAG systems are bolted on top: generate the full response, then decide if you should have retrieved something. lfg.cpp computes entropy, surprise, and confidence *as tokens are sampled*, and can act on them mid-generation:

| Signal | Monitor | Meaning | RAG action |
|---|---|---|---|
| High entropy output | Entropy | Model is uncertain | **Retrieve** from memory |
| High surprise input | Surprise | Input contains novel info | **Store** input |
| Low entropy output | Confidence | Model is confident about this span | **Store** output |

All three monitors produce mean-pooled, L2-normalized embeddings alongside their events — ready for vector similarity search with no additional embedding calls. Entropy events include a rewind checkpoint: when the model is uncertain at token N, callers can pop the event, run retrieval, call `lfg_session_rewind()`, inject context, and continue from that exact point. No re-encoding of everything before N. The signal detection itself happens in the hot loop; policy/orchestration remains caller-controlled.

**Tool execution doesn't break the KV cache.** Most tool-calling implementations generate, stop, parse the tool call, execute, rebuild the entire prompt with the result, and re-encode everything from scratch. lfg.cpp appends the tool result as a continuation directly into the existing KV cache and resumes generation. For a 2K-token conversation, that's 2K tokens you don't re-encode per tool round. The engine handles the full cycle — embedding-based ranking, prompt injection of top-K tools, structured parsing, callback execution, result injection, and continuation — across multiple rounds without session resets.

**Structured output is reasoning-aware.** Grammar constraints (GBNF or JSON Schema) are automatically suspended inside `<think>...</think>` blocks so the model reasons freely, then constrained output resumes. A reasoning budget sampler enforces a hard token cap on thinking.

## Why it's fast

**Zero heap allocations in the hot path.** Every buffer — entropy ring slots, confidence accumulators, tool embedding caches, sampler state — is pre-allocated at session creation or tool registration. The decode/sample/generate loop does not call `malloc`. This matters when generating thousands of tokens where every allocation is a potential cache miss.

**Shared computation across monitors.** Entropy and confidence both need softmax probabilities and Shannon entropy. The engine computes softmax once and shares it — enabling both monitors costs the same as enabling one. The surprise monitor reuses logits already computed during ingestion, just enabling all-position logits instead of last-token-only.

**Lazy embedding computation.** The confidence monitor tracks span statistics during sampling (~5% overhead) but only computes the mean-pooled embedding when you actually pop the event. If you don't pop, no embedding work happens. Early designs computed embeddings eagerly during sampling — that was 84% overhead.

**Lock-free event delivery.** Entropy, confidence, and surprise write events into pre-allocated ring buffers with atomic counters. Consumers poll `*_pending()` or `*_counter()` with no mutex on the generation path.

**FNV-1a embedding cache.** Tool description embeddings are computed once at registration and cached by hash. Query embeddings are cached the same way — if the same query appears twice, it's a hash lookup, not a full model forward pass. The cache uses flat parallel arrays, not `std::unordered_map` — no pointer chasing, no allocator overhead.

**KV cache rewind instead of replay.** When an entropy event is popped and the caller decides to inject context, the engine truncates the KV cache to the trigger position and re-encodes only the injected text plus the few tokens after the trigger. It doesn't replay the entire sequence from the beginning. If truncation fails (some backends don't support partial removal), it falls back to full replay — but on Metal and CPU backends, truncation works and saves substantial compute.

**ISA-specific builds with no runtime dispatch.** The dist libraries ship per-ISA: avx2, avx512, amx on x86; dotprod, i8mm on ARM. The consumer picks the right one for their hardware. No runtime dispatch overhead, no lowest-common-denominator codegen. The Zig build system cross-compiles all five platform targets from a single machine.

## Features

- **Pure C11 API** with opaque handles (`lfg_session*`, `lfg_checkpoint*`) — C++ RAII wrapper included
- **Engine-side generate loop** with token streaming callbacks, queue-driven entropy/confidence/surprise events, and synchronous tool callbacks
- **Three-signal RAG** — entropy (retrieve), surprise (store input), confidence (store output) with embeddings
- **Auto tool execution** — parse tool calls, execute callbacks, inject results, continue generation
- **Embedding-based tool ranking** — cosine similarity with score gating (OFF/AUTO/FIXED)
- **Structured decoding** — GBNF grammars or JSON schemas (auto-detected and converted)
- **Reasoning blocks** — grammar suspension inside thinking spans, token budget enforcement
- **Checkpointing** — snapshot and restore full session state (KV cache + sampler + grammar)
- **Token healing** — prefix sampler corrects tokenization boundary artifacts
- **Stop sequences** — token-level sequences and encoding-independent text-level stop strings
- **Vision** — integrated CLIP and SigLIP encoders
- **Zero-allocation hot path** — all buffers pre-allocated; no heap allocations during generation
- **Cross-platform** — prebuilt static/shared libraries for macOS, Linux, Windows with ISA variants

## Quick start

```c
#include "lfg_api.h"

int main(void) {
    lfg_backend_init();

    // Load model
    lfg_model_load_config lcfg = lfg_model_load_default_config();
    lcfg.model_path = "models/your-model.gguf";
    struct lfg_model *model = lfg_load_model(&lcfg);

    // Create session
    lfg_session_config scfg = lfg_session_default_config();
    scfg.n_ctx = 2048;
    scfg.sampling.temp = 0.0f;
    lfg_session *session = lfg_session_create(model, &scfg);

    // Generate
    lfg_generate_config gc = lfg_generate_default_config();
    gc.max_tokens = 128;
    lfg_generate_result result = lfg_session_prompt_generate(
        session, "Hello, world!", 13, true, gc);

    lfg_session_free(session);
    lfg_model_free(model);
    return 0;
}
```

## Building

The project uses the **Zig build system** for cross-compilation. The implementation is C++17 internally; the public API is C11.

```bash
# Build everything
zig build

# Build optimized (recommended — debug builds hit ggml assertions)
zig build -Doptimize=ReleaseFast

# Run all tests (test/bench steps are available when all_targets=false)
zig build test -Dall_targets=false

# Build benchmark binaries
zig build bench -Dall_targets=false

# Run a specific test
zig build install -Dall_targets=false && ./zig-out/bin/test_entropy_monitor

# Filter test cases (doctest)
./zig-out/bin/test_entropy_monitor -tc="Entropy event fires"

# Build the ImGui demo (macOS)
zig build -Ddemo=true -Doptimize=ReleaseFast
```

### Build options

| Option | Default | Description |
|---|---|---|
| `-Doptimize` | Debug | `ReleaseFast`, `ReleaseSafe`, `ReleaseSmall` |
| `-Dmetal` | Auto (macOS) | Metal GPU backend |
| `-Daccelerate` | Auto (macOS) | Apple Accelerate BLAS |
| `-Dopenmp` | false | OpenMP threading |
| `-Dall_targets` | true | Cross-compile for all supported platforms |
| `-Ddemo` | false | Build the ImGui demo application |

### Prebuilt libraries

The `dist/` directory contains prebuilt static and shared libraries:

| Platform | Architecture | ISA variants |
|---|---|---|
| macOS | aarch64 | baseline, dotprod, i8mm |
| Linux | aarch64 | baseline, dotprod, i8mm |
| Linux | x86_64 | baseline, avx2, avx512, amx |
| Windows | aarch64 | baseline, dotprod |
| Windows | x86_64 | baseline, avx2, avx512, amx |

Rebuild dist libraries:
```bash
zig build -p dist -Dbaseline_combined=true -Disa_combined=true \
    -Dall_targets=true -Doptimize=ReleaseFast --cache-dir /tmp/fresh
```

## API overview

### Core lifecycle

```c
lfg_backend_init();
struct lfg_model *model = lfg_load_model(&load_config);
lfg_session *session    = lfg_session_create(model, &session_config);

// ... use session ...

lfg_session_free(session);
lfg_model_free(model);
```

### Generate loop

The library-owned generate loop handles decode, sample, and tool execution in a single call:

```c
lfg_generate_config gc = lfg_generate_default_config();
gc.max_tokens      = 256;
gc.token_cb        = my_token_cb;        // Per-token streaming
gc.tool_call_cb    = my_tool_observer;   // Auto-executed tool call notification
gc.max_tool_rounds = 3;                  // 0 = default (5)

lfg_generate_result result = lfg_session_chat_generate(session, messages, n, gc);
// result.n_tokens, result.n_tool_calls, result.n_tool_rounds,
// result.stop_reason (EOS, max_tokens, callback, tool_call)
```

### Three-signal RAG

```c
// Entropy retrieval loop (external orchestration):
// 1) pop event, 2) vector search, 3) rewind+inject on owner thread.
lfg_entropy_event ev;
if (lfg_session_entropy_pop(session, &ev, embd_buf, n_embd)) {
    const char *inject = vector_db_search(embd_buf, n_embd);
    if (inject && lfg_session_rewind(session, ev.checkpoint_id)) {
        // tokenize+ingest inject
    }
}

// Surprise/confidence are queue-based informational events.
lfg_surprise_event sev;
while (lfg_session_surprise_pop(session, &sev, embd_buf, n_embd)) { /* store */ }

lfg_confidence_event cev;
while (lfg_session_confidence_pop(session, &cev, embd_buf, n_embd)) { /* store */ }
```

Each monitor supports adaptive gating (OFF/FIXED/AUTO). AUTO mode fires relative to a running mean, adapting to the model's natural entropy distribution.

### Tool ranking and auto execution

```c
// Register tools with optional execution callbacks
lfg_tool_desc tools[] = {
    { "calculator", "Evaluate a math expression",
      "{\"type\":\"object\",\"properties\":{\"expression\":{\"type\":\"string\"}}}",
      calculator_fn, NULL },
    { "get_weather", "Get weather for a location", NULL, weather_fn, NULL },
};
lfg_session_register_tools(session, tools, 2, /*top_k=*/3);

// Tool score gating controls whether tools are injected at all
session_config.tool_score_mode = LFG_TOOL_SCORE_AUTO;  // Skip if relevance is low
```

When tools have non-NULL `fn` callbacks, the engine auto-executes tool calls during generation: parse the model output, call the callback, inject the result as a continuation into the KV cache, and resume. No session reset, no re-encoding. The `tool_call_cb` observer fires per execution for logging.

Tools without callbacks cause generation to stop with `LFG_STOP_TOOL_CALL`, returning parsed calls via `lfg_session_get_tool_calls()` for consumer-side execution.

### Structured decoding

```c
// JSON schema — auto-detected when string starts with '{'
lfg_session_configure_structured(session,
    "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"}}}", NULL);

// Or explicit GBNF grammar
lfg_session_configure_structured(session, "root ::= \"yes\" | \"no\"", "root");
```

### Reasoning blocks

Grammar constraints are automatically suspended inside `<think>...</think>` blocks:

```c
// Configure reasoning delimiters
lfg_session_configure_reasoning(session, start_tokens, n_start, end_tokens, n_end);

// Enforce a token budget on thinking (optional)
session_config.reasoning_budget = 512;
```

### Checkpointing

```c
lfg_checkpoint *cp = lfg_session_create_checkpoint(session);
// ... generate, branch, explore ...
lfg_session_restore_checkpoint(session, cp);  // Restores KV cache + sampler + grammar
lfg_checkpoint_free(cp);
```

### Stop sequences

```c
// Token-level stop sequences
lfg_session_configure_stop_sequences(session, sequences, lengths, n_sequences);

// Text-level stop strings (encoding-independent — same text always matches)
const char *stops[] = { "\n\n", "END" };
lfg_session_configure_stop_strings(session, stops, 2);
```

### Embeddings

```c
float embd[2048];
int32_t n = lfg_session_embed(session, "some text", 9, embd, 2048);
// Mean-pooled, L2-normalized — ready for cosine similarity (dot product)
```

### C++ RAII wrapper

```cpp
#include "lfg_api.hpp"

liquid::Session session(model);
session.ConfigureStructured(R"({"type":"object","properties":{"answer":{"type":"string"}}})");

auto checkpoint = session.CreateCheckpoint();
session.RestoreCheckpoint(checkpoint);
```

## Demo application

The `tools/demo/` directory contains a full-featured ImGui desktop application showcasing every lfg.cpp feature:

- **Chat** — multi-turn streaming chat with thinking block display and tool call visualization
- **Surprise** — input novelty events per turn with mean/max surprise metrics
- **Retrieval** — entropy trigger events with normalized entropy, position, and generated context
- **Confidence** — low-entropy span events with span text and position range
- **Context** — exact formatted prompts and raw model output per turn (with special tokens)
- **Tools** — tool ranking results, parsed tool calls, auto-execution logs
- **Memory** — in-memory vector DB viewer with stored entries and retrieval log

The demo implements a complete in-memory vector DB using the three-signal RAG system with 13 working tool callbacks.

```bash
zig build -Ddemo=true -Doptimize=ReleaseFast
./zig-out/bin/lfg-demo
```

## Project structure

```
src/
  inference/
    lfg_api.h         # High-level session API (C11)
    lfg_api.hpp       # C++ RAII wrapper
    lfg_api.cpp       # Implementation
    lfg_inference.h   # Low-level C API (model, context, sampling primitives)
  ggml/               # Vendored ggml tensor library (CPU, Metal)
  vision/             # CLIP and SigLIP vision encoders
  tests/              # 37 doctest integration tests
  eval/               # Evaluation tools
  benchmarks/         # Performance benchmarks
tools/demo/           # ImGui interactive demo
dist/                 # Prebuilt libraries for all platforms
build.zig             # Zig build configuration
```

## Architecture

The engine is built around an opaque `lfg_session` handle that owns all state: KV cache position, sampler chain, grammar constraints, entropy/confidence/surprise monitors, tool embeddings, and checkpoint history. Sessions are created from a shared `lfg_model` (thread-safe, read-only after load).

Key design decisions:

- **Zero-alloc hot path** — all monitors use pre-allocated ring buffers with atomic write indices. Tool embeddings are cached in flat arrays by FNV-1a hash. No heap allocations during generation.
- **Shared computation** — entropy and confidence monitors share the same softmax + Shannon entropy calculation. Enabling both costs no more than enabling one.
- **Lazy embeddings** — confidence monitor computes mean-pooled embeddings on `pop()`, not during sampling, keeping the sampling loop at ~5% overhead instead of 84%.
- **KV cache continuation** — auto tool execution injects results directly into the existing KV cache and continues. No session reset, no re-encoding of prior context.
- **C11 public API** — the header is parseable by a C11 compiler. Implementation is C++17 internally.

## Benchmarks

All benchmarks run on Apple M3 Max (CPU only, 8 threads), LFM-2.5-1.2B-Thinking Q4_K_M, Zig ReleaseFast. Both lfg.cpp and vendored llama.cpp compiled by the same Zig/clang toolchain in a single binary.

### Raw generation: lfg.cpp vs llama.cpp

Greedy decode from BOS, median of 5 iterations after 2 warmup runs.

| Tokens | llama.cpp (tok/s) | lfg.cpp (tok/s) | Difference |
|--------|-------------------|-----------------|------------|
| 128    | 135.9             | 137.0           | +0.8%      |
| 256    | 135.4             | 137.1           | +1.3%      |
| 512    | 124.7             | 132.9           | +6.6%      |

Same model, same compiler, same hardware. The session API adds no overhead to raw decode performance.

### Structured decoding overhead

Standard generation vs JSON-schema-constrained generation, 100 tokens.

| Mode | tok/s |
|---|---|
| Standard | 134 |
| JSON schema constrained | 84 |

Grammar constraint overhead: ~37%. This is the cost of per-token grammar masking, not the engine layer.

### Monitor overhead

Median tok/s generating 128 tokens with various monitor combinations enabled.

| Configuration | tok/s | Overhead |
|---|---|---|
| Baseline (no monitors) | 128.3 | — |
| Entropy only | 124.5 | -3.0% |
| Confidence only | 123.9 | -3.4% |
| Entropy + Confidence | 121.4 | -5.4% |

Entropy and confidence share the softmax computation, so enabling both is cheaper than the sum of each.

### Surprise monitor overhead

Prompt ingestion and generation with surprise monitor enabled. Average of 3 trials.

| Prompt length | Baseline ingest (tok/s) | Surprise ingest (tok/s) | Ingest overhead | Gen overhead |
|---|---|---|---|---|
| 73 tokens | 176 | 134 | -24% | ~0% |
| 286 tokens | 178 | 135 | -24% | ~0% |
| 570 tokens | 175 | 134 | -24% | ~0% |

Surprise adds ~24% overhead to prompt ingestion (enables all-position logits) but zero overhead to generation.

### Entropy monitor: event overhead

Generation overhead scales with how many events fire. Default settings fire rarely; worst-case (threshold=0.01, firing every token) is pathological but included for completeness.

| Configuration | Gen tok/s | Events | Overhead |
|---|---|---|---|
| No entropy | 109 | 0 | — |
| Default (t=0.7, cd=16) | 105 | 0 | -4% |
| Aggressive (t=0.3, cd=4) | 105 | 0-1 | -4% |
| Worst case (t=0.01, cd=1) | 5 | 26-42 | -95% |

At default settings, entropy monitoring costs ~4%. Each event that fires computes an embedding (~80ms), so the cost is proportional to the number of retrievals triggered. In practice, meaningful thresholds fire 0-2 events per generation.

### Tool ranking overhead

End-to-end generation with tool ranking. Tool embeddings cached after first registration.

| Configuration | Register | TTFT | Gen ms/tok | Notes |
|---|---|---|---|---|
| No tools (baseline) | — | 0.2 ms | 9.2 | — |
| Manual inject all 5 tools | — | 2109 ms | 9.4 | +2s from 373 extra prompt tokens |
| Ranked top_k=3 (cache hit) | 0.2 ms | 0.2 ms | 9.3 | Embedding lookup only |
| Ranked top_k=1 (cache hit) | 0.2 ms | 0.2 ms | 9.2 | Minimal context injection |
| Ranked top_k=3 (cold cache) | 2696 ms | 0.2 ms | 9.2 | One-time embedding cost |

Tool registration with cold cache costs ~2.7s (computes embeddings for all 5 tool descriptions). After that, ranking is a 0.2ms hash lookup. Generation speed is unaffected — the only cost is the extra tokens injected into the prompt.

### Long context scaling

Generation speed as the KV cache fills. 64 tokens generated at each context depth, median of 5 iterations after 2 warmup runs.

| Context depth | Prompt tokens | tok/s | Degradation |
|---|---|---|---|
| 512 | 512 | 104.4 | — |
| 1024 | 1024 | 101.9 | -2.4% |
| 2048 | 2048 | 96.3 | -7.7% |
| 4096 | 4096 | 86.2 | -17.4% |
| 8192 | 8192 | 70.0 | -32.9% |

Attention cost is O(n) in context length. At 8K context the engine still delivers 70 tok/s — a 33% drop from the 512-token baseline, consistent with the linear attention cost scaling.

### Rewind throughput

Effective throughput under entropy-triggered KV cache rewinds. Each retrieval injects a fixed 50-token context. 128 tokens generated, median of 5 iterations.

| Configuration | tok/s | Rewinds | Wall time (ms) | Effective tok/s |
|---|---|---|---|---|
| Baseline (no entropy) | 109.6 | 0 | 1167.6 | 109.6 |
| Light rewind (t=0.5, cd=32) | 104.1 | 0 | 1229.2 | 104.1 |
| Heavy rewind (t=0.2, cd=8) | 0.7 | 51 | 181700.1 | 0.7 |

At practical thresholds (t=0.5) no rewinds fire and overhead is just the monitor cost (~5%). The heavy rewind case (t=0.2, cd=8) is deliberately pathological — 51 rewinds in 128 tokens means the engine spends almost all time re-encoding injected context. This confirms that threshold tuning is critical: rewinds are cheap individually but catastrophic when they cascade.

### Parallel session scaling

Aggregate throughput with N concurrent sessions sharing one model. 128 tokens per session, 1 thread per session, median of 3 iterations.

| Sessions | Wall time (ms) | Per-thread tok/s | Aggregate tok/s | Efficiency |
|---|---|---|---|---|
| 1 | 3549.2 | 36.1 | 36.1 | 100.0% |
| 2 | 3822.1 | 33.5 | 67.0 | 92.9% |
| 4 | 3962.7 | 32.5 | 129.2 | 89.6% |
| 8 | 4912.7 | 26.3 | 208.4 | 72.2% |

The model is loaded once and shared read-only across sessions. At 4 sessions the engine sustains 89.6% scaling efficiency (129 aggregate tok/s vs 144 ideal). At 8 sessions contention on shared memory bandwidth reduces efficiency to 72%, but aggregate throughput still reaches 208 tok/s — 5.8x the single-session rate.

### Reproducing

```bash
zig build install -Dall_targets=false -Doptimize=ReleaseFast

# lfg.cpp vs llama.cpp
./zig-out/bin/benchmark_perf_compare models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf \
    --n-predict 256 --n-threads 8 --warmup 2 --iters 5

# Monitor overhead
./zig-out/bin/bench-confidence-overhead models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf \
    --n-predict 128 --n-threads 8 --warmup 2 --iters 5

# Surprise overhead
./zig-out/bin/bench-surprise-overhead

# Tool ranking + entropy
./zig-out/bin/bench-tool-ranking

# Structured decoding
./zig-out/bin/benchmark_json_schema models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf

# Long context scaling
./zig-out/bin/bench-long-context models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf

# Rewind throughput
./zig-out/bin/bench-rewind models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf

# Parallel session scaling
./zig-out/bin/bench-parallel-sessions models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf
```

## Testing

Tests use [doctest](https://github.com/doctest/doctest) and require a model file at `models/lfm2-350M.gguf`.

```bash
zig build test -Dall_targets=false                      # Run all tests
./zig-out/bin/test_entropy_monitor -tc="Entropy event fires"  # Filter by name
```

## License

See [LICENSE](LICENSE) for details.
