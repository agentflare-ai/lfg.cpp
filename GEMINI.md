# LFG

**LFG** (Liquid Foundation Generation) is a high-performance, CPU-only inference engine and server designed for the Liquid AI LFM2.5 family of models. It provides a unified "omni" runtime capable of processing text, vision, and audio prompts by orchestrating separate GGUF models and presenting them as a single virtual model via an Open Responses-compatible API.

## Project Overview

*   **Goal:** Run LFM2.5 (Text, Vision, Audio) models efficiently on general-purpose CPUs (ARM64/x86_64) without GPU dependencies.
*   **Architecture:**
    *   **Language:** C++17
    *   **Build System:** Zig
    *   **Core Inference:** Leverages optimized kernels from `llama.cpp` (via `src/ggml`).
    *   **API:** Open Responses compliant HTTP server.
*   **Key Models:**
    *   `LFM2.5-1.2B-Thinking-GGUF` (Text)
    *   `LFM2.5-VL-1.6B-GGUF` (Vision)
    *   `LFM2.5-Audio-1.5B-GGUF` (Audio)

## Building and Running

### Prerequisites
*   **Zig** (0.15+)
*   **C++ toolchain** (Zig uses Clang under the hood)
*   **pthreads**

### Build Instructions

```bash
zig build -Doptimize=ReleaseFast
```

Build outputs (static libraries):
*   `zig-out/lib/liblfg.a` (combined: liquid core + ggml + vision)
*   `zig-out/lib/liblfm_core.a`
*   `zig-out/lib/libggml.a`
*   `zig-out/lib/liblfm_vision.a`

Optional flags:
*   `-Dmetal=true|false` (defaults to `true` on macOS)
*   `-Dmetal_embed=true|false` (defaults to `true` when Metal is enabled)
*   `-Daccelerate=true|false` (defaults to `true` on macOS)
*   `-Dopenmp=true|false` (defaults to `false`)
*   `-Dnative=true|false` (defaults to `true` only for native builds; set `false` for cross-compile)

### Cross-Compile

```bash
zig build -Doptimize=ReleaseFast -Dtarget=x86_64-linux-gnu -Dnative=false
```

### Running the CLI
The main CLI tool is `lfg_cli`, currently set up to verify model loading.

```bash
./zig-out/bin/lfg_cli <path_to_model.gguf>
```

### Running the Server
The server implementation resides in `src/server`. 
*Note: Currently, the server subdirectory might need to be explicitly built or added to the root build configuration.*

## Directory Structure

*   `src/`: Core source code.
    *   `main.cpp`: Entry point for the CLI tool.
    *   `inference/`: Inference logic and model wrappers.
    *   `loader/`: GGUF model loading and validation.
    *   `server/`: HTTP API server implementation.
    *   `ggml/`: Low-level tensor library (forked/adapted from llama.cpp).
    *   `vision/`: Vision encoders (CLIP/SigLIP).
*   `third_party/`: External dependencies (e.g., `llama.cpp`).
*   `docs/`: Project documentation and specifications (`spec.md`, `plan.md`).
*   `models/`: Directory for storing GGUF model files.
*   `tests/`: Unit and integration tests.

## Development Conventions

*   **Code Style:** Modern C++17.
*   **Warnings:** The project enforces strict warnings (`-Wall -Wextra -Wpedantic -Werror`) for internal code.
*   **Optimization:** Optimizes for specific CPU architectures (AVX, AVX2, AVX-512, NEON) automatically.
*   **Model Format:** strictly **GGUF**.
*   **No GPU:** The project is explicitly designed for CPU execution; do not add CUDA/ROCm dependencies.

## Structured Decoding (Tooling-Friendly)

LFG supports JSON-schema-driven structured decoding for reliable tool calling and agent workflows.

*   **Schema to grammar:** `lfm_session_configure_structured(session, grammar_or_schema, root_rule)` accepts either a GBNF grammar or a JSON schema string (if it begins with `{`).
*   **Structured checkpointing:** `lfm_session_config.structured_checkpointing` (default `true`) snapshots sampler state, including grammar parse state, so restores and token healing remain deterministic under structured decoding.
*   **Checkpoint restore options:** use `lfm_session_restore_checkpoint_ex` with `lfm_checkpoint_restore_options` to control:
    *   `restore_sampler_state` (default `true`)
    *   `restore_grammar` (default `true`)

These defaults prioritize high success rates for tool calls and agent-style orchestration by keeping grammar state aligned across healing and checkpoint restore.