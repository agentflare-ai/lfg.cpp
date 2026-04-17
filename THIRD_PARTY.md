# Third-Party Software

This repository contains third-party code and vendored components that remain under their upstream licenses.

## Major bundled components

| Component | Location | License |
|---|---|---|
| llama.cpp / ggml | `third_party/llama.cpp`, `src/ggml/` | MIT |
| Dear ImGui | `third_party/imgui/` | MIT |
| doctest | `src/tests/doctest.h` | MIT |
| stb_image | `src/vision/stb_image.h` | MIT or public domain |
| Intel / LLVM-derived SYCL backend snippets | `src/ggml/ggml-sycl/` | MIT and Apache-2.0 WITH LLVM-exception notices in-file |

## Notes

- `src/ggml/` contains vendored and adapted low-level backend code. Preserve in-file notices when redistributing modified versions.
- Some individual source files carry their own SPDX identifiers or upstream copyright blocks. Those file-level notices take precedence for those files.
- `third_party/macos_sdk_stubs/` contains local compatibility stubs authored for this repository and is covered by the repository license unless noted otherwise.

## Upstream license files

- `third_party/llama.cpp/LICENSE`
- `third_party/imgui/LICENSE.txt`
- `third_party/llama.cpp/licenses/LICENSE-jsonhpp`
- `third_party/llama.cpp/gguf-py/LICENSE`
