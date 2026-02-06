#!/usr/bin/env python3
import argparse
import os
import sys


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ggml Metal embedded sources")
    parser.add_argument("--common", required=True)
    parser.add_argument("--metal", required=True)
    parser.add_argument("--impl", required=True)
    parser.add_argument("--out-metal", required=True)
    parser.add_argument("--out-asm", required=True)
    args = parser.parse_args()

    common = load_text(args.common)
    metal = load_text(args.metal)
    impl = load_text(args.impl)

    merged = metal.replace("__embed_ggml-common.h__", common)
    merged = merged.replace('#include "ggml-metal-impl.h"', impl)

    write_text(args.out_metal, merged)

    asm_path = os.path.abspath(args.out_asm)
    metal_path = os.path.abspath(args.out_metal)

    asm = (
        ".section __DATA,__ggml_metallib\n"
        ".globl _ggml_metallib_start\n"
        "_ggml_metallib_start:\n"
        f".incbin \"{metal_path}\"\n"
        ".globl _ggml_metallib_end\n"
        "_ggml_metallib_end:\n"
    )

    write_text(asm_path, asm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
