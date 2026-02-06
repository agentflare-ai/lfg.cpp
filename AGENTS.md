# LFG.cpp Agent Guidelines

This repository contains the `lfg.cpp` inference engine, a C++ codebase built using the Zig build system. It integrates `ggml` and provides advanced sampling capabilities (grammar, reasoning, etc.).

## 1. Build and Test Commands

The project uses `zig build` as the primary build system.

### Build
- **Build all artifacts (libraries and tests):**
  ```bash
  zig build
  ```
- **Build for release (optimized):**
  ```bash
  zig build -Doptimize=ReleaseFast
  ```
  *Note: Use ReleaseFast for performance testing or if Debug mode triggers excessive assertions.*

### Test
- **Run all tests:**
  ```bash
  zig build test
  ```
- **Run a single test file:**
  Tests are compiled into individual executables in `zig-out/bin/`.
  1. Build the tests:
     ```bash
     zig build install
     ```
  2. Run the specific test executable:
     ```bash
     ./zig-out/bin/test_reasoning_gate
     ```
- **Filter test cases (Doctest):**
  The tests use `doctest`. You can filter specific test cases within a binary:
  ```bash
  ./zig-out/bin/test_reasoning_gate -tc="Reasoning Budget Enforcement"
  ```

### Benchmarks
- **Run benchmarks:**
  ```bash
  zig build bench
  ```

## 2. Code Style & Conventions

### Language Standard
- **C++:** C++17 (`-std=c++17`).
- **Zig:** Used only for build orchestration (`build.zig`).

### Formatting
- **Indentation:** 4 spaces.
- **Braces:** Same line for functions/control structures (K&R/Stroustrup style).
  ```cpp
  if (condition) {
      // ...
  }
  ```

### Naming Conventions
- **General:** Heavily `snake_case`.
- **Types/Structs:** `snake_case` (e.g., `lfg_context`, `lfg_sampler_chain`).
- **Functions:** `snake_case` (e.g., `lfg_sampler_init_grammar`).
- **Prefixes:** Public API types and functions are prefixed with `lfg_`.
- **Members:** `snake_case` (e.g., `cparams.n_ctx`).
- **Macros/Constants:** `UPPER_CASE` (e.g., `LFG_LOG_INFO`, `LFG_TOKEN_NULL`).

### Error Handling
- **Internal C++ Logic:** Use exceptions (e.g., `throw std::runtime_error("message")`).
- **C API Boundaries:** Catch exceptions and report via `lfg_set_last_error`.
  ```cpp
  try {
      // ...
  } catch (const std::exception & err) {
      lfg_set_last_error(LFG_ERROR_INTERNAL, "%s: failed: %s", __func__, err.what());
      return nullptr;
  }
  ```

### Logging
- Use `spdlog` macros wrapped by LFG macros:
  - `LFG_LOG_INFO(...)`
  - `LFG_LOG_WARN(...)`
  - `LFG_LOG_ERROR(...)`
  - `LFG_LOG_DEBUG(...)`

### Memory Management
- Prefer `std::unique_ptr` for ownership.
- Use raw pointers for non-owning references (observers) passed to functions.
- Manual memory management (`new`/`delete`) is rare and usually encapsulated in `lfg_sampler` constructors/destructors.

### Project Structure
- `src/inference/`: Core inference logic (`lfg_context`, `lfg_sampling`, etc.).
- `src/ggml/`: Low-level tensor operations (vendored/adapted).
- `src/tests/`: Integration tests using `doctest`.
- `build.zig`: Build configuration.

### Best Practices
1. **Safety:** Always check pointers before use if they can be null.
2. **Const Correctness:** Use `const` for read-only references and pointers.
3. **Headers:** Use `#pragma once`. Include local headers first, then system headers.
4. **Git:** Do not commit `models/` or `zig-out/` or `zig-cache/`.
