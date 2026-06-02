# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run an IRP-OFF / IRP-EAGER / TAPER+ sweep for BoN serving.

The runner wraps ``benchmark_taper_plus.py --mode all`` over a grid of BoN
widths, output lengths, request counts, and arrival rates. It records successful
metrics and failed/OOM runs in machine-readable files so long experiments can be
resumed and audited.
"""

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


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


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


def flatten_result(
    config: dict[str, Any],
    mode_result: dict[str, Any],
) -> dict[str, Any]:
    row = dict(config)
    row["mode"] = mode_result["mode"]
    for key, value in mode_result["metrics"].items():
        row[key] = value
    row["status"] = "ok"
    row["error"] = ""
    return row


def failure_row(
    config: dict[str, Any],
    returncode: int | str,
    elapsed_s: float,
    log_path: Path,
) -> dict[str, Any]:
    row = dict(config)
    row.update({
        "mode": "all",
        "status": "failed",
        "error": str(returncode),
        "elapsed_s": elapsed_s,
        "log_path": str(log_path),
    })
    return row


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--bon-values", default="8,16,32,64")
    parser.add_argument("--output-lens", default="256,1024,4096")
    parser.add_argument("--num-requests-values", default="16,32,64")
    parser.add_argument("--request-rates", default="2,4,8")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--compare-mode",
        choices=["both", "both_plus", "both_bon", "both_bon_plus", "all"],
        default="both",
        help=(
            "Comparison mode passed to benchmark_taper_plus.py: both is "
            "IRP-EAGER vs TAPER, both_plus is IRP-EAGER vs TAPER+, all "
            "includes IRP-OFF, IRP-EAGER, TAPER, and TAPER+. both_bon and "
            "both_bon_plus compare IRP-EAGER against BoN-TAPER variants."
        ),
    )
    parser.add_argument("--serial-fraction", type=float, default=0.5)
    parser.add_argument("--synthetic-prompt-words", type=int, default=256)
    parser.add_argument("--workload-seed", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument("--sampling-seed-stride", type=int, default=100000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--ttft-slo", type=float, default=2.0)
    parser.add_argument("--tpot-slo", type=float, default=0.05)
    parser.add_argument(
        "--parallel-slo-mode",
        choices=["logical", "branch"],
        default="logical",
    )
    parser.add_argument("--taper-step-target-ms", type=float, default=50.0)
    parser.add_argument("--taper-latency-per-seq-ms", type=float, default=0.5)
    parser.add_argument("--taper-latency-profile", type=str, default="")
    parser.add_argument("--taper-rho", type=float, default=1.0)
    parser.add_argument(
        "--taper-utility",
        choices=["linear", "concave"],
        default="linear",
    )
    parser.add_argument("--taper-calibration-alpha", type=float, default=0.08)
    parser.add_argument("--taper-safety-factor", type=float, default=1.10)
    parser.add_argument("--disable-taper-online-calibration", action="store_true")
    parser.add_argument("--enable-taper-eager-when-unpressured", action="store_true")
    parser.add_argument("--disable-taper-branch-overdue-boost", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--output-root", default="benchmark_outputs/taper_plus_sweep")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra raw argument passed to benchmark_taper_plus.py.",
    )
    return parser


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep TAPER+ against IRP-OFF and IRP-EAGER."
    )
    args = add_cli_args(parser).parse_args()

    bon_values = parse_int_list(args.bon_values)
    output_lens = parse_int_list(args.output_lens)
    request_counts = parse_int_list(args.num_requests_values)
    request_rates = parse_float_list(args.request_rates)
    output_root = Path(args.output_root)
    rows: list[dict[str, Any]] = []

    env = os.environ.copy()
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    for repeat in range(args.repeats):
        for bon_n in bon_values:
            repeat_workload_seed = args.workload_seed + repeat
            repeat_sampling_seed = (
                None if args.sampling_seed is None else args.sampling_seed + repeat
            )
            for output_len in output_lens:
                for num_requests in request_counts:
                    for request_rate in request_rates:
                        max_model_len = output_len + args.synthetic_prompt_words + 64
                        run_name = (
                            f"rep{repeat}_bon{bon_n}_out{output_len}_"
                            f"req{num_requests}_rate{request_rate:g}"
                        )
                        run_dir = output_root / run_name
                        comparison_path = run_dir / "comparison.json"
                        log_path = run_dir / "run.log"

                        config = {
                            "repeat": repeat,
                            "model": args.model,
                            "bon_n": bon_n,
                            "output_len": output_len,
                            "num_requests": num_requests,
                            "request_rate": request_rate,
                            "serial_fraction": args.serial_fraction,
                            "workload_seed": repeat_workload_seed,
                            "sampling_seed": repeat_sampling_seed,
                            "taper_latency_profile": args.taper_latency_profile,
                            "max_model_len": max_model_len,
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
                            args.compare_mode,
                            "--model",
                            args.model,
                            "--dtype",
                            args.dtype,
                            "--max-model-len",
                            str(max_model_len),
                            "--gpu-memory-utilization",
                            str(args.gpu_memory_utilization),
                            "--num-requests",
                            str(num_requests),
                            "--request-rate",
                            str(request_rate),
                            "--bon-n",
                            str(bon_n),
                            "--serial-fraction",
                            str(args.serial_fraction),
                            "--workload-seed",
                            str(repeat_workload_seed),
                            "--output-len",
                            str(output_len),
                            "--synthetic-prompt-words",
                            str(args.synthetic_prompt_words),
                            "--ttft-slo",
                            str(args.ttft_slo),
                            "--tpot-slo",
                            str(args.tpot_slo),
                            "--parallel-slo-mode",
                            args.parallel_slo_mode,
                            "--taper-step-target-ms",
                            str(args.taper_step_target_ms),
                            "--taper-latency-per-seq-ms",
                            str(args.taper_latency_per_seq_ms),
                            "--taper-latency-profile",
                            args.taper_latency_profile,
                            "--taper-rho",
                            str(args.taper_rho),
                            "--taper-utility",
                            args.taper_utility,
                            "--taper-calibration-alpha",
                            str(args.taper_calibration_alpha),
                            "--taper-safety-factor",
                            str(args.taper_safety_factor),
                            "--output-dir",
                            str(run_dir),
                            "--log-level",
                            "INFO",
                        ]
                        if args.disable_taper_online_calibration:
                            cmd.append("--disable-taper-online-calibration")
                        if args.enable_taper_eager_when_unpressured:
                            cmd.append("--enable-taper-eager-when-unpressured")
                        if args.disable_taper_branch_overdue_boost:
                            cmd.append("--disable-taper-branch-overdue-boost")
                        if repeat_sampling_seed is not None:
                            cmd.extend(["--sampling-seed", str(repeat_sampling_seed)])
                            cmd.extend([
                                "--sampling-seed-stride",
                                str(args.sampling_seed_stride),
                            ])
                        if args.enforce_eager:
                            cmd.append("--enforce-eager")
                        if args.ignore_eos:
                            cmd.append("--ignore-eos")
                        cmd.extend(args.extra_arg)

                        print("Running", run_name, flush=True)
                        try:
                            returncode, elapsed_s = run_command(
                                cmd, env, args.timeout_s, log_path
                            )
                        except subprocess.TimeoutExpired:
                            rows.append(
                                failure_row(config, "timeout", args.timeout_s, log_path)
                            )
                            write_rows(output_root / "sweep_summary.csv", rows)
                            continue

                        if returncode != 0 or not comparison_path.exists():
                            rows.append(
                                failure_row(config, returncode, elapsed_s, log_path)
                            )
                            write_rows(output_root / "sweep_summary.csv", rows)
                            continue

                        with comparison_path.open("r", encoding="utf-8") as f:
                            data = json.load(f)
                        rows.extend(
                            flatten_result(config, result)
                            for result in data["results"]
                        )
                        write_rows(output_root / "sweep_summary.csv", rows)

    write_rows(output_root / "sweep_summary.csv", rows)
    with (output_root / "sweep_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
