# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run Default vLLM-BoN vs TailBlend over real prompt sets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT_DIR / "benchmarks" / "benchmark_taper_plus.py"


DEFAULT_DATASETS = {
    "gsm8k": "benchmark_outputs/tail_blend_generalization_prompts/gsm8k_40.jsonl",
    "mbpp": "benchmark_outputs/tail_blend_generalization_prompts/mbpp_40.jsonl",
    "longbench_qasper": (
        "benchmark_outputs/tail_blend_generalization_prompts/"
        "longbench_qasper_40.jsonl"
    ),
    "chat_ultrachat": (
        "benchmark_outputs/tail_blend_generalization_prompts/"
        "chat_ultrachat_40.jsonl"
    ),
}


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_dataset_list(value: str) -> dict[str, str]:
    if not value:
        return dict(DEFAULT_DATASETS)
    datasets: dict[str, str] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        name, path = item.split("=", 1)
        datasets[name.strip()] = path.strip()
    return datasets


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_result(config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    row = dict(config)
    row["mode"] = result["mode"]
    row.update(result["metrics"])
    row["status"] = "ok"
    row["error"] = ""
    return row


def failure_row(
    config: dict[str, Any],
    error: str,
    elapsed_s: float,
    log_path: Path,
) -> dict[str, Any]:
    row = dict(config)
    row.update({
        "mode": "both_tail_blend",
        "status": "failed",
        "error": error,
        "elapsed_s": elapsed_s,
        "log_path": str(log_path),
    })
    return row


def run_command(
    cmd: list[str],
    env: dict[str, str],
    timeout_s: int,
    log_path: Path,
) -> tuple[int, float]:
    start = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
            text=True,
        )
    return process.returncode, time.perf_counter() - start


def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--bon-values", default="8,32")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--num-requests", type=int, default=40)
    parser.add_argument("--request-rate", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--output-len", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=5120)
    parser.add_argument("--long-output-len", type=int, default=2048)
    parser.add_argument("--long-max-model-len", type=int, default=8192)
    parser.add_argument("--ttft-slo", type=float, default=180.0)
    parser.add_argument("--tpot-slo", type=float, default=0.20)
    parser.add_argument("--parallel-slo-mode", default="logical")
    parser.add_argument("--timeout-s", type=int, default=5400)
    parser.add_argument(
        "--output-root",
        default="benchmark_outputs/tail_blend_generalization_full_vs_baseline",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--log-level", default="WARNING")
    return parser


def main() -> int:
    parser = argparse.ArgumentParser()
    args = add_cli_args(parser).parse_args()

    datasets = parse_dataset_list(args.datasets)
    bon_values = parse_int_list(args.bon_values)
    seeds = parse_int_list(args.seeds)
    output_root = Path(args.output_root)
    rows: list[dict[str, Any]] = []

    env = os.environ.copy()
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    for dataset_name, prompt_path in datasets.items():
        is_long = dataset_name.startswith("longbench")
        output_len = args.long_output_len if is_long else args.output_len
        max_model_len = args.long_max_model_len if is_long else args.max_model_len
        for bon_n in bon_values:
            for seed in seeds:
                run_name = f"{dataset_name}_seed{seed}_bon{bon_n}"
                run_dir = output_root / run_name
                comparison_path = run_dir / "comparison.json"
                log_path = run_dir / "run.log"
                config = {
                    "dataset": dataset_name,
                    "prompt_jsonl": prompt_path,
                    "seed": seed,
                    "bon_n": bon_n,
                    "num_requests": args.num_requests,
                    "request_rate": args.request_rate,
                    "output_len": output_len,
                    "max_model_len": max_model_len,
                    "ttft_slo": args.ttft_slo,
                    "tpot_slo": args.tpot_slo,
                }

                if args.resume and comparison_path.exists():
                    with comparison_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    rows.extend(
                        flatten_result(config, result)
                        for result in data["results"]
                    )
                    write_rows(output_root / "sweep_summary.csv", rows)
                    continue

                cmd = [
                    sys.executable,
                    str(BENCHMARK),
                    "--mode",
                    "both_tail_blend",
                    "--enable-tail-blend-preemption",
                    "--tail-blend-preemption-mode",
                    "full",
                    "--model",
                    args.model,
                    "--dtype",
                    args.dtype,
                    "--max-model-len",
                    str(max_model_len),
                    "--gpu-memory-utilization",
                    str(args.gpu_memory_utilization),
                    "--num-requests",
                    str(args.num_requests),
                    "--request-rate",
                    str(args.request_rate),
                    "--bon-n",
                    str(bon_n),
                    "--serial-fraction",
                    "0.0",
                    "--output-len",
                    str(output_len),
                    "--prompt-jsonl",
                    prompt_path,
                    "--ttft-slo",
                    str(args.ttft_slo),
                    "--tpot-slo",
                    str(args.tpot_slo),
                    "--parallel-slo-mode",
                    args.parallel_slo_mode,
                    "--sampling-seed",
                    str(seed),
                    "--workload-seed",
                    str(seed),
                    "--output-dir",
                    str(run_dir),
                    "--log-level",
                    args.log_level,
                ]
                if args.enforce_eager:
                    cmd.append("--enforce-eager")
                if args.ignore_eos:
                    cmd.append("--ignore-eos")

                print(f"Running {run_name}", flush=True)
                try:
                    returncode, elapsed_s = run_command(
                        cmd,
                        env,
                        args.timeout_s,
                        log_path,
                    )
                except subprocess.TimeoutExpired:
                    rows.append(
                        failure_row(config, "timeout", args.timeout_s, log_path)
                    )
                    write_rows(output_root / "sweep_summary.csv", rows)
                    continue

                if returncode != 0 or not comparison_path.exists():
                    rows.append(
                        failure_row(
                            config,
                            f"returncode={returncode}",
                            elapsed_s,
                            log_path,
                        )
                    )
                    write_rows(output_root / "sweep_summary.csv", rows)
                    continue

                with comparison_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                rows.extend(
                    flatten_result(config, result) for result in data["results"]
                )
                write_rows(output_root / "sweep_summary.csv", rows)

    write_rows(output_root / "sweep_summary.csv", rows)
    with (output_root / "sweep_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
