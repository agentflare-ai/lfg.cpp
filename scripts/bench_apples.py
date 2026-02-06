#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import time


def run_cmd(cmd, env=None):
    proc = subprocess.run(cmd, capture_output=True, env=env)
    stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
    stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
    return proc.returncode, stdout, stderr


def parse_metrics(text):
    metrics = {}
    tps_match = re.search(r"^TPS:\s*([0-9.]+)", text, re.MULTILINE)
    if tps_match:
        metrics["tps"] = float(tps_match.group(1))
    tokens_match = re.search(r"^Tokens generated:\s*(\d+)", text, re.MULTILINE)
    if tokens_match:
        metrics["tokens"] = int(tokens_match.group(1))
    time_match = re.search(r"^Time:\s*([0-9.]+)\s*ms", text, re.MULTILINE)
    if time_match:
        metrics["time_ms"] = float(time_match.group(1))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run apples-to-apples liquid vs llama benchmark")
    parser.add_argument("model_path")
    parser.add_argument("n_predict", nargs="?", type=int, default=1024)
    parser.add_argument("seed", nargs="?", type=int, default=42)
    parser.add_argument("n_threads", nargs="?", type=int, default=4)
    parser.add_argument("ngl", nargs="?", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--igpu", action="store_true")
    parser.add_argument("--no-kv-offload", action="store_true")
    parser.add_argument("--no-op-offload", action="store_true")
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument("--force-flash-attn", action="store_true")
    parser.add_argument("--swap-order", action="store_true", help="Run llama before liquid")
    parser.add_argument("--bin-dir", default="build/bin")
    parser.add_argument("--snapshot-dir", default=".snapshot")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    if args.cpu + args.gpu + args.igpu > 1:
        print("error: choose only one of --cpu/--gpu/--igpu", file=sys.stderr)
        return 2
    if args.no_flash_attn and args.force_flash_attn:
        print("error: choose only one of --no-flash-attn/--force-flash-attn", file=sys.stderr)
        return 2

    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"error: model not found: {model_path}", file=sys.stderr)
        return 2

    lfm_bin = os.path.join(args.bin_dir, "liquid-compare")
    llama_bin = os.path.join(args.bin_dir, "llama-compare")

    if not os.path.exists(lfm_bin):
        print(f"error: missing {lfm_bin}", file=sys.stderr)
        return 2
    if not os.path.exists(llama_bin):
        print(f"error: missing {llama_bin}", file=sys.stderr)
        return 2

    os.makedirs(args.snapshot_dir, exist_ok=True)

    device_flag = None
    if args.cpu:
        device_flag = "--cpu"
    elif args.gpu:
        device_flag = "--gpu"
    elif args.igpu:
        device_flag = "--igpu"
    extra_flags = []
    if args.no_kv_offload:
        extra_flags.append("--no-kv-offload")
    if args.no_op_offload:
        extra_flags.append("--no-op-offload")
    if args.no_flash_attn:
        extra_flags.append("--no-flash-attn")
    if args.force_flash_attn:
        extra_flags.append("--force-flash-attn")

    results = {
        "timestamp": int(time.time()),
        "model_path": model_path,
        "n_predict": args.n_predict,
        "seed": args.seed,
        "n_threads": args.n_threads,
        "ngl": args.ngl,
        "device": device_flag or "auto",
        "repeat": args.repeat,
        "runs": [],
    }

    base_args = [
        model_path,
        str(args.n_predict),
        str(args.seed),
        str(args.n_threads),
        str(args.ngl),
    ]

    for i in range(args.repeat):
        run_entry = {"iteration": i + 1}

        lfm_cmd = [lfm_bin] + base_args
        llama_cmd = [llama_bin] + base_args
        if device_flag:
            lfm_cmd.append(device_flag)
            llama_cmd.append(device_flag)
        if extra_flags:
            lfm_cmd += extra_flags
            llama_cmd += extra_flags
        lfm_cmd.append("--quiet")
        llama_cmd.append("--quiet")

        if args.swap_order:
            rc, out, err = run_cmd(llama_cmd)
            run_entry["llama"] = {
                "cmd": llama_cmd,
                "returncode": rc,
                "metrics": parse_metrics(err),
                "stderr": err,
            }

            rc, out, err = run_cmd(lfm_cmd)
            run_entry["liquid"] = {
                "cmd": lfm_cmd,
                "returncode": rc,
                "metrics": parse_metrics(err),
                "stderr": err,
            }
        else:
            rc, out, err = run_cmd(lfm_cmd)
            run_entry["liquid"] = {
                "cmd": lfm_cmd,
                "returncode": rc,
                "metrics": parse_metrics(err),
                "stderr": err,
            }

            rc, out, err = run_cmd(llama_cmd)
            run_entry["llama"] = {
                "cmd": llama_cmd,
                "returncode": rc,
                "metrics": parse_metrics(err),
                "stderr": err,
            }

        results["runs"].append(run_entry)

    snapshot_path = os.path.join(
        args.snapshot_dir, f"bench_apples_{results['timestamp']}.json"
    )
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {snapshot_path}")

    for run in results["runs"]:
        lfm_tps = run["liquid"]["metrics"].get("tps")
        llama_tps = run["llama"]["metrics"].get("tps")
        ratio = None
        if lfm_tps is not None and llama_tps:
            ratio = lfm_tps / llama_tps
        print(
            f"Run {run['iteration']}: Liquid TPS {lfm_tps} | Llama TPS {llama_tps}"
            + (f" | Ratio {ratio:.3f}" if ratio is not None else "")
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
