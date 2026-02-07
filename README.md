# lfg.cpp

A C inference engine for GGUF language models with built-in structured decoding, entropy-driven retrieval, confidence-based knowledge store detection, embedding-based tool ranking, and cross-platform prebuilt libraries.

## Features

- **Pure C API** with opaque handles (`lfg_session*`, `lfg_checkpoint*`) — no C++ in the public interface
- **Structured decoding** — constrained generation via GBNF grammars or JSON schemas
- **Reasoning/thinking blocks** — token-delimited reasoning spans with budget enforcement, grammar constraints suspended inside thinking blocks
- **Entropy monitor** — real-time Shannon entropy tracking with SPSC ring buffer, triggers retrieval when the model is uncertain
- **Confidence monitor** — inverse entropy signal detecting sustained low-entropy spans for knowledge store candidates
- **Tool ranking** — embedding-based semantic ranking of registered tools, auto-injected into context
- **Embeddings** — mean-pooled, L2-normalized text embeddings via `lfg_session_embed()`
- **Generate loop** — library-owned decode+sample loop with per-token, entropy, and confidence callbacks
- **Checkpointing** — snapshot and restore KV cache + sampler state for speculative/structured decoding
- **Token healing** — corrects tokenization boundary artifacts at generation boundaries
- **Stop sequences** — arbitrary token-sequence-based stop conditions
- **Cross-platform** — prebuilt static and shared libraries for macOS, Linux, and Windows with ISA-specific variants

## Quick Start

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

The project uses the **Zig build system** for cross-compilation. A standard C++17 compiler is required for the implementation (the public API is C11).

```bash
# Build everything
zig build

# Build optimized
zig build -Doptimize=ReleaseFast

# Build and install executables (tests, evals, benchmarks)
zig build install -Doptimize=ReleaseFast
```

### Prebuilt Libraries

The `dist/` directory contains prebuilt static and shared libraries for all supported targets:

| Platform | Architecture | Variants |
|----------|-------------|----------|
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

## Testing

Tests use the [doctest](https://github.com/doctest/doctest) framework and require a model file at `models/lfm2-350M.gguf`.

```bash
# Run all tests
zig build test

# Run a specific test
./zig-out/bin/test_entropy_monitor

# Filter test cases
./zig-out/bin/test_entropy_monitor -tc="Entropy event fires"
```

## API Overview

### Core Lifecycle

```c
lfg_backend_init();
struct lfg_model *model = lfg_load_model(&load_config);
lfg_session *session    = lfg_session_create(model, &session_config);

// ... use session ...

lfg_session_free(session);
lfg_model_free(model);
```

### Structured Decoding

Constrain output to a GBNF grammar or JSON schema:

```c
// JSON schema — auto-detected when string starts with '{'
lfg_session_configure_structured(session,
    "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"}}}", NULL);

// Or explicit GBNF grammar
lfg_session_configure_structured(session, "root ::= \"yes\" | \"no\"", "root");
```

### Entropy Monitor (Retrieval Signal)

Detect high-uncertainty tokens in real time. When entropy exceeds a threshold, the generate loop can rewind and inject retrieved context:

```c
lfg_entropy_monitor_config ecfg = lfg_entropy_monitor_default_config();
ecfg.threshold = 0.7f;
ecfg.cooldown_tokens = 16;
int32_t n_embd = lfg_session_configure_entropy_monitor(session, &ecfg);

// In the generate loop, the entropy callback receives the event + embedding:
const char *my_entropy_cb(const lfg_entropy_event *event,
                          const float *embedding, void *user_data) {
    // Use embedding to find relevant context in your knowledge base
    // Return a string to inject, or NULL to skip
    return retrieved_context;
}
```

### Confidence Monitor (Store Signal)

Detect sustained low-entropy spans where the model is confidently generating. These spans are candidates for storing into a knowledge base:

```c
lfg_confidence_monitor_config ccfg = lfg_confidence_monitor_default_config();
ccfg.threshold = 0.3f;
ccfg.min_span = 5;
int32_t n_embd = lfg_session_configure_confidence_monitor(session, &ccfg);

// In the generate loop, the confidence callback fires per span:
void my_confidence_cb(const lfg_confidence_event *event,
                      const float *embedding, void *user_data) {
    // event->span_length consecutive low-entropy tokens
    // embedding is mean-pooled over the span — use for KB indexing
}
```

### Tool Ranking

Register tools with descriptions; the engine ranks them by semantic similarity to the prompt and injects the top-k into context:

```c
lfg_tool_desc tools[] = {
    {"get_weather",  "Get current weather for a location", NULL},
    {"send_email",   "Send an email message",              NULL},
    {"search_web",   "Search the web for information",     NULL},
};
lfg_session_register_tools(session, tools, 3, /*top_k=*/2);
```

### Generate Loop

The library-owned generate loop handles decode, sample, entropy retrieval, and confidence detection in a single call:

```c
lfg_generate_config gc = lfg_generate_default_config();
gc.max_tokens        = 256;
gc.token_cb          = my_token_cb;
gc.token_cb_data     = &my_state;
gc.entropy_cb        = my_entropy_cb;
gc.entropy_cb_data   = &my_kb;
gc.confidence_cb     = my_confidence_cb;
gc.confidence_cb_data = &my_store;

lfg_generate_result result = lfg_session_generate(session, gc);
// result.n_tokens, result.n_retrievals, result.n_confidence_spans, result.stop_reason
```

### C++ RAII Wrapper

A lightweight header-only wrapper is provided in `lfg_api.hpp`:

```cpp
#include "lfg_api.hpp"
liquid::Session session(model);
session.IngestTokens(tokens);
session.Decode();
lfg_token tok = session.Sample();
```

## Project Structure

```
src/
  inference/
    lfg_inference.h   # Low-level C API (model, context, sampling primitives)
    lfg_api.h         # High-level session API (structured decoding, monitors, tools)
    lfg_api.hpp       # C++ RAII wrapper
    lfg_api.cpp       # Implementation
  ggml/               # Vendored ggml tensor library
  tests/              # doctest integration tests
  eval/               # Benchmarks and evaluation tools
dist/                 # Prebuilt libraries for all platforms
build.zig             # Zig build system
```

## Architecture

The engine is built around an opaque `lfg_session` handle that owns all state: KV cache position, sampler chain, grammar constraints, entropy/confidence ring buffers, tool embeddings, and checkpoint history. Sessions are created from a shared `lfg_model` (thread-safe, read-only after load).

Key design decisions:
- **Zero-alloc hot path** — entropy and confidence monitors use pre-allocated SPSC ring buffers with atomic write indices. No heap allocations during generation.
- **Lazy embeddings** — confidence monitor computes mean-pooled embeddings on `pop()`, not during sampling. This keeps the sampling loop at ~5% overhead instead of 84%.
- **Shared computation** — entropy and confidence monitors share the same softmax + Shannon entropy calculation. Enabling both costs no more than enabling one.
- **C11 public API** — the header is parseable by a C11 compiler. Implementation is C++17 internally.

## License

See [LICENSE](LICENSE) for details.
