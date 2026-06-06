# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark TAPER+ against stock eager BoN scheduling.

This benchmark drives ``LLMEngine`` directly so it can submit requests with a
Poisson arrival process and timestamp streaming delta outputs at the client
side. It supports:

* irp_off: admit only the protected one branch per BoN parent per step
* irp_eager: stock vLLM behavior
* taper: TAPER+ scheduler enabled via environment variables
* tail_blend: stock eager admission plus BoN-aware preemption
* bon_taper: BoN parent-level TAPER-inspired controller
* bon_taper_len_predictor: BoN-TAPER plus cheap live-branch length hints
* both/all: run comparison modes in separate subprocesses, then print a table
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402

logger = logging.getLogger("benchmark_taper_plus")


MODE_LABELS = {
    "irp_off": "IRP-OFF",
    "irp_eager": "IRP-EAGER",
    "taper": "TAPER",
    "taper_plus": "TAPER+",
    "tail_blend": "TailBlend",
    "eager_tail_blend": "TailBlend",
    "bon_taper": "BoN-TAPER",
    "bon_taper_plus": "BoN-TAPER+",
    "bon_taper_len_predictor": "BoN-TAPER-LEN",
}


def canonical_mode(mode: str) -> str:
    if mode == "baseline":
        return "irp_eager"
    if mode == "eager_tail_blend":
        return "tail_blend"
    return mode


@dataclass
class WorkloadRequest:
    request_id: str
    prompt: str
    arrival_s: float
    bon_n: int
    sampling_seed: int | None = None


@dataclass
class RequestTrace:
    request_id: str
    scheduled_arrival_s: float
    submit_time_s: float
    expected_branches: int
    prompt_len: int | None = None
    first_token_time_s: float | None = None
    finish_time_s: float | None = None
    output_tokens: int = 0
    branch_tokens: dict[int, int] = field(default_factory=dict)
    branch_last_token_time_s: dict[int, float] = field(default_factory=dict)
    branch_inter_token_latencies_s: list[float] = field(default_factory=list)
    logical_inter_token_latencies_s: list[float] = field(default_factory=list)
    logical_last_token_time_s: float | None = None
    finished: bool = False

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_time_s is None:
            return None
        return self.first_token_time_s - self.submit_time_s

    @property
    def latency_s(self) -> float | None:
        if self.finish_time_s is None:
            return None
        return self.finish_time_s - self.submit_time_s

    @property
    def max_tpot_s(self) -> float:
        if not self.logical_inter_token_latencies_s:
            return 0.0
        return max(self.logical_inter_token_latencies_s)

    @staticmethod
    def _percentile(values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        if len(sorted_values) == 1:
            return sorted_values[0]
        rank = (len(sorted_values) - 1) * pct / 100.0
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return sorted_values[int(rank)]
        lower_value = sorted_values[lower]
        upper_value = sorted_values[upper]
        return lower_value + (upper_value - lower_value) * (rank - lower)

    @property
    def mean_tpot_s(self) -> float:
        if not self.logical_inter_token_latencies_s:
            return 0.0
        return statistics.fmean(self.logical_inter_token_latencies_s)

    @property
    def p95_tpot_s(self) -> float:
        return self._percentile(self.logical_inter_token_latencies_s, 95)

    @property
    def p99_tpot_s(self) -> float:
        return self._percentile(self.logical_inter_token_latencies_s, 99)

    @property
    def branch_max_tpot_s(self) -> float:
        if not self.branch_inter_token_latencies_s:
            return 0.0
        return max(self.branch_inter_token_latencies_s)

    @property
    def branch_mean_tpot_s(self) -> float:
        if not self.branch_inter_token_latencies_s:
            return 0.0
        return statistics.fmean(self.branch_inter_token_latencies_s)

    @property
    def branch_p95_tpot_s(self) -> float:
        return self._percentile(self.branch_inter_token_latencies_s, 95)

    @property
    def branch_p99_tpot_s(self) -> float:
        return self._percentile(self.branch_inter_token_latencies_s, 99)

    def tpot_s_for_slo(
        self,
        parallel_slo_mode: str,
        tpot_slo_aggregation: str,
    ) -> float:
        prefix = "branch_" if parallel_slo_mode == "branch" else ""
        return getattr(self, f"{prefix}{tpot_slo_aggregation}_tpot_s")

    def met_slo(
        self,
        ttft_slo_s: float,
        tpot_slo_s: float,
        parallel_slo_mode: str,
        tpot_slo_aggregation: str,
    ) -> bool:
        tpot_s = self.tpot_s_for_slo(parallel_slo_mode, tpot_slo_aggregation)
        return (
            self.finished
            and len(self.branch_tokens) == self.expected_branches
            and self.ttft_s is not None
            and self.ttft_s <= ttft_slo_s
            and tpot_s <= tpot_slo_s
        )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def make_synthetic_prompt(index: int, target_words: int) -> str:
    base = (
        "You are analyzing a production incident for a language model serving "
        "cluster. Explain the root cause, user impact, mitigation steps, and "
        "follow-up actions in a concise but technically specific report."
    )
    words = base.split()
    repeated = [words[(index + i) % len(words)] for i in range(target_words)]
    return " ".join(repeated)


def load_sharegpt_prompts(path: Path, limit: int) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    prompts: list[str] = []
    for row in data:
        conversations = row.get("conversations") or row.get("conversation")
        if not conversations:
            continue
        for turn in conversations:
            role = str(turn.get("from") or turn.get("role") or "").lower()
            text = turn.get("value") or turn.get("content")
            if text and role in {"human", "user"}:
                prompts.append(str(text))
                break
        if len(prompts) >= limit:
            break
    return prompts


def load_jsonl_prompts(path: Path, limit: int) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if prompt is None:
                problem = row.get("problem") or row.get("Problem")
                if problem is not None:
                    prompt = str(problem)
            if prompt:
                prompts.append(str(prompt))
            if len(prompts) >= limit:
                break
    return prompts


def build_workload(args: argparse.Namespace) -> list[WorkloadRequest]:
    rng = random.Random(args.workload_seed)
    if args.prompt_jsonl:
        prompts = load_jsonl_prompts(Path(args.prompt_jsonl), args.num_requests)
        if len(prompts) < args.num_requests:
            raise ValueError(
                f"Only found {len(prompts)} JSONL prompts, need {args.num_requests}."
            )
    elif args.sharegpt_path:
        prompts = load_sharegpt_prompts(Path(args.sharegpt_path), args.num_requests)
        if len(prompts) < args.num_requests:
            raise ValueError(
                f"Only found {len(prompts)} ShareGPT prompts, need {args.num_requests}."
            )
    else:
        prompts = [
            make_synthetic_prompt(i, args.synthetic_prompt_words)
            for i in range(args.num_requests)
        ]

    arrivals: list[float] = []
    elapsed = 0.0
    for i in range(args.num_requests):
        if i == 0 or math.isinf(args.request_rate):
            interarrival = 0.0
        else:
            interarrival = rng.expovariate(args.request_rate)
        elapsed += interarrival
        arrivals.append(elapsed)

    serial_target = round(args.num_requests * args.serial_fraction)
    serial_indices = set(rng.sample(range(args.num_requests), serial_target))

    return [
        WorkloadRequest(
            request_id=(
                f"serial-{i}" if i in serial_indices else f"bon{args.bon_n}-{i}"
            ),
            prompt=prompt,
            arrival_s=arrivals[i],
            bon_n=1 if i in serial_indices else args.bon_n,
            sampling_seed=(
                None
                if args.sampling_seed is None
                else args.sampling_seed + i * args.sampling_seed_stride
            ),
        )
        for i, prompt in enumerate(prompts)
    ]


def configure_mode_env(mode: str, args: argparse.Namespace) -> None:
    mode = canonical_mode(mode)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ENABLE_TAPER_PLUS"] = (
        "1"
        if mode
        in {
            "irp_off",
            "taper",
            "taper_plus",
            "tail_blend",
            "bon_taper",
            "bon_taper_plus",
            "bon_taper_len_predictor",
        }
        else "0"
    )
    if mode in {
        "taper_plus",
        "tail_blend",
        "bon_taper",
        "bon_taper_plus",
        "bon_taper_len_predictor",
    }:
        os.environ["VLLM_TAPER_PLUS_POLICY"] = mode
    else:
        os.environ["VLLM_TAPER_PLUS_POLICY"] = "taper"
    step_target_ms = 0.0 if mode == "irp_off" else args.taper_step_target_ms
    os.environ["VLLM_TAPER_PLUS_STEP_TARGET_MS"] = str(step_target_ms)
    os.environ["VLLM_TAPER_PLUS_LATENCY_PER_SEQ_MS"] = str(
        args.taper_latency_per_seq_ms
    )
    os.environ["VLLM_TAPER_PLUS_LATENCY_PROFILE_PATH"] = (
        args.taper_latency_profile or ""
    )
    os.environ["VLLM_TAPER_PLUS_TTFT_SLO_MS"] = str(args.ttft_slo * 1000.0)
    os.environ["VLLM_TAPER_PLUS_TPOT_SLO_MS"] = str(args.tpot_slo * 1000.0)
    os.environ["VLLM_TAPER_PLUS_RHO"] = str(args.taper_rho)
    os.environ["VLLM_TAPER_PLUS_UTILITY"] = args.taper_utility
    os.environ["VLLM_TAPER_PLUS_MEMORY_THRESHOLD_BLOCKS"] = str(
        args.taper_memory_threshold_blocks
    )
    os.environ["VLLM_TAPER_PLUS_ONLINE_CALIBRATION"] = (
        "0" if args.disable_taper_online_calibration else "1"
    )
    os.environ["VLLM_TAPER_PLUS_RECORD_LATENCY_SAMPLES"] = (
        "1"
        if args.record_taper_latency_samples or bool(args.write_taper_latency_profile)
        else "0"
    )
    os.environ["VLLM_TAPER_PLUS_CALIBRATION_ALPHA"] = str(args.taper_calibration_alpha)
    os.environ["VLLM_TAPER_PLUS_SAFETY_FACTOR"] = str(args.taper_safety_factor)
    os.environ["VLLM_TAPER_PLUS_EAGER_WHEN_UNPRESSURED"] = (
        "1" if args.enable_taper_eager_when_unpressured else "0"
    )
    os.environ["VLLM_TAPER_PLUS_BRANCH_OVERDUE_BOOST"] = (
        "0" if args.disable_taper_branch_overdue_boost else "1"
    )
    os.environ["VLLM_TAIL_BLEND_PILOT_K"] = str(args.tail_blend_pilot_k)
    os.environ["VLLM_TAIL_BLEND_PREEMPTION"] = (
        "1" if args.enable_tail_blend_preemption else "0"
    )
    os.environ["VLLM_TAIL_BLEND_PREEMPTION_MODE"] = args.tail_blend_preemption_mode
    os.environ["VLLM_TAIL_BLEND_RESERVATION_Q"] = str(args.tail_blend_reservation_q)
    os.environ["VLLM_TAIL_BLEND_ADMISSION_MODE"] = args.tail_blend_admission_mode


def make_sampling_params(args: argparse.Namespace, bon_n: int, seed: int | None = None):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    return SamplingParams(
        n=bon_n,
        temperature=args.temperature,
        max_tokens=args.output_len,
        ignore_eos=args.ignore_eos,
        seed=seed,
        detokenize=not args.disable_detokenize,
        output_kind=RequestOutputKind.DELTA,
    )


def get_preemption_stats(engine: Any) -> dict[str, float]:
    try:
        core_client = engine.engine_core
        engine_core = getattr(core_client, "engine_core", None)
        scheduler = getattr(engine_core, "scheduler", None)
        controller = getattr(scheduler, "tail_blend_controller", None)
        adaptive_counts = getattr(controller, "adaptive_choice_counts", {})
        return {
            "preemption_count": float(
                getattr(scheduler, "num_cumulative_preemptions", 0)
            ),
            "predictor_preemption_count": float(
                getattr(scheduler, "num_cumulative_predictor_preemptions", 0)
            ),
            "preempted_computed_tokens": float(
                getattr(
                    scheduler,
                    "num_cumulative_preempted_computed_tokens",
                    0,
                )
            ),
            "preempted_output_tokens": float(
                getattr(
                    scheduler,
                    "num_cumulative_preempted_output_tokens",
                    0,
                )
            ),
            "adaptive_default_count": float(adaptive_counts.get("default", 0)),
            "adaptive_full_count": float(adaptive_counts.get("full", 0)),
            "adaptive_prefill_aware_count": float(
                adaptive_counts.get("prefill_aware", 0)
            ),
            "adaptive_full_prefix_guard_count": float(
                adaptive_counts.get("full_prefix_guard", 0)
            ),
        }
    except Exception:
        return {
            "preemption_count": 0.0,
            "predictor_preemption_count": 0.0,
            "preempted_computed_tokens": 0.0,
            "preempted_output_tokens": 0.0,
            "adaptive_default_count": 0.0,
            "adaptive_full_count": 0.0,
            "adaptive_prefill_aware_count": 0.0,
            "adaptive_full_prefix_guard_count": 0.0,
        }


def get_taper_latency_profile(
    engine: Any, args: argparse.Namespace
) -> dict[str, Any] | None:
    try:
        core_client = engine.engine_core
        engine_core = getattr(core_client, "engine_core", None)
        scheduler = getattr(engine_core, "scheduler", None)
        if scheduler is None:
            return None
        samples = list(getattr(scheduler, "taper_plus_latency_samples", []))
        fitted_coefficients = fit_taper_latency_coefficients(samples)
        if fitted_coefficients is None:
            fitted_coefficients = {
                "base_ms": float(scheduler.taper_plus_latency_base_ms),
                "width_ms": float(scheduler.taper_plus_latency_width_ms),
                "context_ms_per_1k": float(
                    scheduler.taper_plus_latency_context_ms_per_1k
                ),
            }
            fit_method = "online_final"
        else:
            fit_method = "offline_least_squares"
        fitted_coefficients["safety_factor"] = float(scheduler.taper_plus_safety_factor)
        return {
            "format": "vllm_taper_latency_profile_v1",
            "model": args.model,
            "fit_method": fit_method,
            "coefficients": fitted_coefficients,
            "calibration_samples": int(
                getattr(scheduler, "taper_plus_calibration_samples", 0)
            ),
            "profile_samples": len(samples),
            "source_config": {
                "taper_latency_per_seq_ms": args.taper_latency_per_seq_ms,
                "taper_calibration_alpha": args.taper_calibration_alpha,
                "taper_safety_factor": args.taper_safety_factor,
                "output_len": args.output_len,
                "synthetic_prompt_words": args.synthetic_prompt_words,
                "prompt_jsonl": args.prompt_jsonl,
                "ignore_eos": args.ignore_eos,
            },
        }
    except Exception:
        return None


def update_trace_from_output(
    trace: RequestTrace,
    request_output: Any,
    now: float,
) -> None:
    if trace.prompt_len is None and request_output.prompt_token_ids is not None:
        trace.prompt_len = len(request_output.prompt_token_ids)

    total_new_tokens = sum(
        len(completion.token_ids) for completion in request_output.outputs
    )
    if total_new_tokens > 0:
        if trace.logical_last_token_time_s is not None:
            trace.logical_inter_token_latencies_s.append(
                (now - trace.logical_last_token_time_s) / total_new_tokens
            )
        trace.logical_last_token_time_s = now

    for completion in request_output.outputs:
        num_tokens = len(completion.token_ids)
        if num_tokens == 0:
            continue

        if trace.first_token_time_s is None:
            trace.first_token_time_s = now

        branch_index = int(completion.index)
        last_time = trace.branch_last_token_time_s.get(branch_index)
        if last_time is not None:
            trace.branch_inter_token_latencies_s.append((now - last_time) / num_tokens)
        trace.branch_last_token_time_s[branch_index] = now
        trace.branch_tokens[branch_index] = (
            trace.branch_tokens.get(branch_index, 0) + num_tokens
        )
        trace.output_tokens += num_tokens

    if request_output.finished:
        trace.finished = True
        trace.finish_time_s = now


def run_one_mode(args: argparse.Namespace) -> dict[str, Any]:
    mode = canonical_mode(args.mode)
    configure_mode_env(mode, args)

    from vllm import LLMEngine

    logger.info("Starting %s run", mode)
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args, enable_multiprocessing=False)
    workload = build_workload(args)
    sampling_params_by_bon = {}
    if args.sampling_seed is None:
        sampling_params_by_bon = {
            bon_n: make_sampling_params(args, bon_n)
            for bon_n in sorted({item.bon_n for item in workload})
        }
    logger.info(
        "Prepared workload: %d requests, n=%d, serial_fraction=%.2f, "
        "request_rate=%s req/s",
        len(workload),
        args.bon_n,
        args.serial_fraction,
        args.request_rate,
    )

    traces: dict[str, RequestTrace] = {}
    next_request_idx = 0
    start_time = time.perf_counter()

    while next_request_idx < len(workload) or engine.has_unfinished_requests():
        now = time.perf_counter()
        elapsed = now - start_time

        while (
            next_request_idx < len(workload)
            and workload[next_request_idx].arrival_s <= elapsed
        ):
            item = workload[next_request_idx]
            submit_time = time.perf_counter()
            scheduler_arrival_time = time.time()
            engine.add_request(
                item.request_id,
                item.prompt,
                sampling_params_by_bon[item.bon_n]
                if item.sampling_seed is None
                else make_sampling_params(args, item.bon_n, item.sampling_seed),
                arrival_time=scheduler_arrival_time,
            )
            traces[item.request_id] = RequestTrace(
                request_id=item.request_id,
                scheduled_arrival_s=item.arrival_s,
                submit_time_s=submit_time,
                expected_branches=item.bon_n,
            )
            logger.debug(
                "Submitted request %s at %.3fs",
                item.request_id,
                submit_time - start_time,
            )
            next_request_idx += 1

        if engine.has_unfinished_requests():
            for request_output in engine.step():
                output_time = time.perf_counter()
                trace = traces[request_output.request_id]
                update_trace_from_output(trace, request_output, output_time)
        elif next_request_idx < len(workload):
            next_arrival = workload[next_request_idx].arrival_s
            sleep_s = max(
                0.0,
                min(0.01, next_arrival - (time.perf_counter() - start_time)),
            )
            if sleep_s > 0:
                time.sleep(sleep_s)

    end_time = time.perf_counter()
    duration_s = end_time - start_time
    preemption_stats = get_preemption_stats(engine)
    taper_latency_profile = get_taper_latency_profile(engine, args)

    records = [trace_to_record(t, args) for t in traces.values()]
    metrics = compute_metrics(records, duration_s, preemption_stats, args)
    result = {
        "mode": mode,
        "config": {
            "model": args.model,
            "num_requests": args.num_requests,
            "request_rate": args.request_rate,
            "bon_n": args.bon_n,
            "temperature": args.temperature,
            "sampling_seed": args.sampling_seed,
            "sampling_seed_stride": args.sampling_seed_stride,
            "output_len": args.output_len,
            "ttft_slo_s": args.ttft_slo,
            "tpot_slo_s": args.tpot_slo,
            "taper_step_target_ms": args.taper_step_target_ms,
            "taper_latency_per_seq_ms": args.taper_latency_per_seq_ms,
            "taper_latency_profile": args.taper_latency_profile,
            "taper_rho": args.taper_rho,
            "taper_utility": args.taper_utility,
            "taper_memory_threshold_blocks": args.taper_memory_threshold_blocks,
            "taper_policy": (
                "disabled"
                if mode == "irp_eager"
                else "taper_plus"
                if mode == "taper_plus"
                else "tail_blend"
                if mode == "tail_blend"
                else "taper"
            ),
            "taper_online_calibration": (not args.disable_taper_online_calibration),
            "taper_calibration_alpha": args.taper_calibration_alpha,
            "taper_safety_factor": args.taper_safety_factor,
            "taper_eager_when_unpressured": (args.enable_taper_eager_when_unpressured),
            "taper_branch_overdue_boost": (not args.disable_taper_branch_overdue_boost),
            "tail_blend_preemption": args.enable_tail_blend_preemption,
            "tail_blend_preemption_mode": args.tail_blend_preemption_mode,
            "tail_blend_pilot_k": args.tail_blend_pilot_k,
            "tail_blend_reservation_q": args.tail_blend_reservation_q,
            "tail_blend_admission_mode": args.tail_blend_admission_mode,
            "parallel_slo_mode": args.parallel_slo_mode,
            "workload_seed": args.workload_seed,
        },
        "metrics": metrics,
        "requests": records,
    }
    if taper_latency_profile is not None:
        result["taper_latency_profile"] = taper_latency_profile

    if args.raw_csv:
        write_csv(Path(args.raw_csv), records)
    if args.result_json:
        write_json(Path(args.result_json), result)
    if args.write_taper_latency_profile:
        if taper_latency_profile is None:
            raise RuntimeError("No TAPER scheduler profile was available to write.")
        write_json(Path(args.write_taper_latency_profile), taper_latency_profile)
    logger.info(
        "Finished %s run: %.3fs, %.0f/%d completed, %.2f%% SLO",
        mode,
        duration_s,
        metrics["completed_requests"],
        len(records),
        metrics["slo_attainment_rate_pct"],
    )
    return result


def trace_to_record(trace: RequestTrace, args: argparse.Namespace) -> dict[str, Any]:
    ttft_s = trace.ttft_s
    latency_s = trace.latency_s
    return {
        "request_id": trace.request_id,
        "request_class": "serial" if trace.expected_branches == 1 else "parallel",
        "expected_branches": trace.expected_branches,
        "scheduled_arrival_s": trace.scheduled_arrival_s,
        "submit_time_s": trace.submit_time_s,
        "first_token_time_s": trace.first_token_time_s,
        "finish_time_s": trace.finish_time_s,
        "prompt_len": trace.prompt_len,
        "output_tokens": trace.output_tokens,
        "num_finished_branches": len(trace.branch_tokens),
        "branch_tokens": trace.branch_tokens,
        "ttft_s": ttft_s,
        "latency_s": latency_s,
        "mean_tpot_s": trace.mean_tpot_s,
        "p95_tpot_s": trace.p95_tpot_s,
        "p99_tpot_s": trace.p99_tpot_s,
        "max_tpot_s": trace.max_tpot_s,
        "branch_mean_tpot_s": trace.branch_mean_tpot_s,
        "branch_p95_tpot_s": trace.branch_p95_tpot_s,
        "branch_p99_tpot_s": trace.branch_p99_tpot_s,
        "branch_max_tpot_s": trace.branch_max_tpot_s,
        "finished": trace.finished,
        "met_slo": trace.met_slo(
            args.ttft_slo,
            args.tpot_slo,
            args.parallel_slo_mode,
            args.tpot_slo_aggregation,
        ),
    }


def compute_metrics(
    records: list[dict[str, Any]],
    duration_s: float,
    preemption_stats: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, float]:
    completed = [r for r in records if r["finished"]]
    good = [r for r in completed if r["met_slo"]]
    serial = [r for r in records if r["request_class"] == "serial"]
    parallel = [r for r in records if r["request_class"] == "parallel"]
    serial_good = [r for r in serial if r["met_slo"]]
    parallel_good = [r for r in parallel if r["met_slo"]]
    ttfts = [r["ttft_s"] for r in completed if r["ttft_s"] is not None]
    total_tokens = sum(int(r["output_tokens"]) for r in records)
    good_tokens = sum(int(r["output_tokens"]) for r in good)
    denom = max(duration_s, 1e-9)
    return {
        "avg_ttft_s": statistics.fmean(ttfts) if ttfts else 0.0,
        "p99_ttft_s": percentile(ttfts, 99),
        "system_throughput_tokens_s": total_tokens / denom,
        "user_throughput_requests_s": len(completed) / denom,
        "user_goodput_tokens_s": good_tokens / denom,
        "slo_attainment_rate_pct": (
            100.0 * len(good) / len(records) if records else 0.0
        ),
        "serial_slo_attainment_rate_pct": (
            100.0 * len(serial_good) / len(serial) if serial else 0.0
        ),
        "parallel_slo_attainment_rate_pct": (
            100.0 * len(parallel_good) / len(parallel) if parallel else 0.0
        ),
        "preemption_count": float(preemption_stats["preemption_count"]),
        "predictor_preemption_count": float(
            preemption_stats["predictor_preemption_count"]
        ),
        "preempted_computed_tokens": float(
            preemption_stats["preempted_computed_tokens"]
        ),
        "preempted_output_tokens": float(preemption_stats["preempted_output_tokens"]),
        "adaptive_default_count": float(preemption_stats["adaptive_default_count"]),
        "adaptive_full_count": float(preemption_stats["adaptive_full_count"]),
        "adaptive_prefill_aware_count": float(
            preemption_stats["adaptive_prefill_aware_count"]
        ),
        "adaptive_full_prefix_guard_count": float(
            preemption_stats["adaptive_full_prefix_guard_count"]
        ),
        "duration_s": duration_s,
        "completed_requests": float(len(completed)),
        "total_requests": float(len(records)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def solve_3x3(a: list[list[float]], b: list[float]) -> list[float] | None:
    matrix = [row[:] + [rhs] for row, rhs in zip(a, b)]
    n = 3
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(matrix[row][col]))
        if abs(matrix[pivot][col]) < 1e-9:
            return None
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        scale = matrix[col][col]
        matrix[col] = [value / scale for value in matrix[col]]
        for row in range(n):
            if row == col:
                continue
            factor = matrix[row][col]
            matrix[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(matrix[row], matrix[col])
            ]
    return [matrix[row][n] for row in range(n)]


def fit_taper_latency_coefficients(
    samples: list[dict[str, float]],
) -> dict[str, float] | None:
    if len(samples) < 3:
        return None

    xtx = [[0.0 for _ in range(3)] for _ in range(3)]
    xty = [0.0 for _ in range(3)]
    for sample in samples:
        features = [
            1.0,
            float(sample["decode_width"]),
            float(sample["context_k"]),
        ]
        observed_ms = float(sample["observed_ms"])
        for i in range(3):
            xty[i] += features[i] * observed_ms
            for j in range(3):
                xtx[i][j] += features[i] * features[j]

    # Tiny ridge term keeps width-only profile runs numerically stable.
    for i in range(3):
        xtx[i][i] += 1e-6

    coeffs = solve_3x3(xtx, xty)
    if coeffs is None:
        return None
    return {
        "base_ms": max(0.0, coeffs[0]),
        "width_ms": max(0.001, coeffs[1]),
        "context_ms_per_1k": max(0.0, coeffs[2]),
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "request_id",
        "request_class",
        "expected_branches",
        "scheduled_arrival_s",
        "submit_time_s",
        "first_token_time_s",
        "finish_time_s",
        "prompt_len",
        "output_tokens",
        "num_finished_branches",
        "ttft_s",
        "latency_s",
        "mean_tpot_s",
        "p95_tpot_s",
        "p99_tpot_s",
        "max_tpot_s",
        "branch_mean_tpot_s",
        "branch_p95_tpot_s",
        "branch_p99_tpot_s",
        "branch_max_tpot_s",
        "finished",
        "met_slo",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fieldnames})


def format_metric(value: float, suffix: str = "") -> str:
    if suffix == "%":
        return f"{value:.2f}%"
    if abs(value) >= 100:
        return f"{value:.2f}{suffix}"
    return f"{value:.4f}{suffix}"


def print_comparison(results: Iterable[dict[str, Any]]) -> None:
    by_mode = {result["mode"]: result["metrics"] for result in results}
    modes = [
        mode
        for mode in (
            "irp_off",
            "irp_eager",
            "taper",
            "taper_plus",
            "tail_blend",
            "bon_taper",
            "bon_taper_plus",
            "bon_taper_len_predictor",
        )
        if mode in by_mode
    ]
    rows = [
        ("Avg TTFT", "avg_ttft_s", "s"),
        ("P99 TTFT", "p99_ttft_s", "s"),
        ("System throughput", "system_throughput_tokens_s", " tok/s"),
        ("User throughput", "user_throughput_requests_s", " req/s"),
        ("User-level goodput", "user_goodput_tokens_s", " tok/s"),
        ("SLO attainment", "slo_attainment_rate_pct", "%"),
        ("Serial SLO attainment", "serial_slo_attainment_rate_pct", "%"),
        ("Parallel SLO attainment", "parallel_slo_attainment_rate_pct", "%"),
        ("Preemption count", "preemption_count", ""),
        ("Predictor preemptions", "predictor_preemption_count", ""),
        ("Preempted computed toks", "preempted_computed_tokens", ""),
        ("Preempted output toks", "preempted_output_tokens", ""),
        ("Adaptive default", "adaptive_default_count", ""),
        ("Adaptive full", "adaptive_full_count", ""),
        ("Adaptive prefill", "adaptive_prefill_aware_count", ""),
        ("Adaptive full guard", "adaptive_full_prefix_guard_count", ""),
    ]
    print("\nBoN Scheduling Benchmark")
    width = 28 + 23 * len(modes)
    print("=" * width)
    header = f"{'Metric':<28}" + "".join(f" {MODE_LABELS[mode]:>22}" for mode in modes)
    print(header)
    print("-" * width)
    for label, key, suffix in rows:
        values = "".join(
            f" {format_metric(float(by_mode[mode].get(key, 0.0)), suffix):>22}"
            for mode in modes
        )
        print(f"{label:<28}{values}")
    print("=" * width)


def print_single(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    rows = [
        ("Avg TTFT", "avg_ttft_s", "s"),
        ("P99 TTFT", "p99_ttft_s", "s"),
        ("System throughput", "system_throughput_tokens_s", " tok/s"),
        ("User throughput", "user_throughput_requests_s", " req/s"),
        ("User-level goodput", "user_goodput_tokens_s", " tok/s"),
        ("SLO attainment", "slo_attainment_rate_pct", "%"),
        ("Serial SLO attainment", "serial_slo_attainment_rate_pct", "%"),
        ("Parallel SLO attainment", "parallel_slo_attainment_rate_pct", "%"),
        ("Preemption count", "preemption_count", ""),
        ("Predictor preemptions", "predictor_preemption_count", ""),
        ("Preempted computed toks", "preempted_computed_tokens", ""),
        ("Preempted output toks", "preempted_output_tokens", ""),
        ("Adaptive default", "adaptive_default_count", ""),
        ("Adaptive full", "adaptive_full_count", ""),
        ("Adaptive prefill", "adaptive_prefill_aware_count", ""),
        ("Adaptive full guard", "adaptive_full_prefix_guard_count", ""),
    ]
    print(f"\nBoN Scheduling Benchmark ({result['mode']})")
    print("=" * 54)
    for label, key, suffix in rows:
        print(f"{label:<28} {format_metric(float(metrics[key]), suffix):>22}")
    print("=" * 54)
    print(f"Duration: {metrics['duration_s']:.3f}s")
    print(
        "Completed: "
        f"{int(metrics['completed_requests'])}/{int(metrics['total_requests'])}"
    )


def child_args(argv: list[str], mode: str, output_dir: Path) -> list[str]:
    skip_next = False
    filtered: list[str] = []
    drop_with_value = {"--mode", "--result-json", "--raw-csv"}
    drop_prefix = tuple(f"{opt}=" for opt in drop_with_value)
    for index, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg in drop_with_value:
            skip_next = index + 1 < len(argv)
            continue
        if arg.startswith(drop_prefix):
            continue
        filtered.append(arg)
    result_json = output_dir / f"{mode}_summary.json"
    raw_csv = output_dir / f"{mode}_requests.csv"
    return [
        *filtered,
        "--mode",
        mode,
        "--result-json",
        str(result_json),
        "--raw-csv",
        str(raw_csv),
    ]


def run_both(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    modes = (
        ("irp_off", "irp_eager", "taper", "taper_plus")
        if args.mode == "all"
        else ("irp_eager", "bon_taper")
        if args.mode == "both_bon"
        else ("irp_eager", "bon_taper_plus")
        if args.mode == "both_bon_plus"
        else ("bon_taper", "bon_taper_len_predictor")
        if args.mode == "both_bon_len_predictor"
        else ("irp_eager", "tail_blend")
        if args.mode == "both_tail_blend"
        else ("irp_eager", "taper_plus")
        if args.mode == "both_plus"
        else ("irp_eager", "taper")
    )
    for mode in modes:
        cmd = [sys.executable, *child_args(sys.argv, mode, output_dir)]
        env = os.environ.copy()
        env["VLLM_ENABLE_TAPER_PLUS"] = (
            "1"
            if mode
            in {
                "irp_off",
                "taper",
                "taper_plus",
                "tail_blend",
                "bon_taper",
                "bon_taper_plus",
                "bon_taper_len_predictor",
            }
            else "0"
        )
        if mode in {
            "taper_plus",
            "tail_blend",
            "bon_taper",
            "bon_taper_plus",
            "bon_taper_len_predictor",
        }:
            env["VLLM_TAPER_PLUS_POLICY"] = mode
        else:
            env["VLLM_TAPER_PLUS_POLICY"] = "taper"
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        logger.info("Running %s: %s", mode, " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)
        with (output_dir / f"{mode}_summary.json").open("r", encoding="utf-8") as f:
            results.append(json.load(f))

    comparison_path = output_dir / "comparison.json"
    write_json(comparison_path, {"results": results})
    print_comparison(results)
    logger.info("Raw outputs written under: %s", output_dir)


def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--mode",
        choices=[
            "baseline",
            "irp_off",
            "irp_eager",
            "taper",
            "taper_plus",
            "tail_blend",
            "eager_tail_blend",
            "bon_taper",
            "bon_taper_plus",
            "bon_taper_len_predictor",
            "both",
            "both_plus",
            "both_tail_blend",
            "both_bon",
            "both_bon_plus",
            "both_bon_len_predictor",
            "all",
        ],
        default="both",
        help=(
            "Run one mode, IRP-EAGER vs TAPER with --mode both, "
            "IRP-EAGER vs TAPER+ with --mode both_plus, "
            "IRP-EAGER vs TailBlend with --mode both_tail_blend, "
            "or all policies."
        ),
    )
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument(
        "--request-rate",
        type=float,
        default=8.0,
        help="Poisson arrival rate in requests/s. Use inf for all-at-once.",
    )
    parser.add_argument("--sharegpt-path", type=str, default=None)
    parser.add_argument(
        "--prompt-jsonl",
        type=str,
        default=None,
        help=(
            "JSONL prompt file. Each line should contain a 'prompt' field; "
            "'problem'/'Problem' is accepted as a fallback."
        ),
    )
    parser.add_argument("--synthetic-prompt-words", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--bon-n", type=int, default=32)
    parser.add_argument(
        "--serial-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of logical requests that use n=1. The rest use --bon-n. "
            "Use this to reproduce mixed serial/parallel TAPER workloads."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help=(
            "Base sampling seed. When set, logical request i uses "
            "seed + i, and vLLM parallel sampling assigns deterministic "
            "child branch seeds from that parent seed."
        ),
    )
    parser.add_argument(
        "--sampling-seed-stride",
        type=int,
        default=100000,
        help="Stride between logical request parent seeds.",
    )
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--disable-detokenize", action="store_true")
    parser.add_argument("--ttft-slo", type=float, default=2.0)
    parser.add_argument("--tpot-slo", type=float, default=0.05)
    parser.add_argument(
        "--tpot-slo-aggregation",
        choices=["max", "p99", "p95", "mean"],
        default="max",
        help=(
            "TPOT statistic used for SLO pass/fail. max preserves the "
            "original strict SLO; p99/p95/mean are useful diagnostics for "
            "separating one-step stalls from sustained slowdowns."
        ),
    )
    parser.add_argument(
        "--parallel-slo-mode",
        choices=["logical", "branch"],
        default="logical",
        help=(
            "logical matches TAPER's request-level deadline; branch keeps the "
            "stricter per-branch TPOT diagnostic."
        ),
    )
    parser.add_argument("--workload-seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_outputs/taper_plus",
        help="Directory used by --mode both.",
    )
    parser.add_argument("--result-json", type=str, default=None)
    parser.add_argument("--raw-csv", type=str, default=None)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("--taper-step-target-ms", type=float, default=50.0)
    parser.add_argument("--taper-latency-per-seq-ms", type=float, default=2.0)
    parser.add_argument(
        "--taper-latency-profile",
        type=str,
        default="",
        help="Offline TAPER latency coefficient JSON used to warm-start T(S).",
    )
    parser.add_argument(
        "--write-taper-latency-profile",
        type=str,
        default="",
        help="Write the final TAPER latency coefficients from this run.",
    )
    parser.add_argument(
        "--record-taper-latency-samples",
        action="store_true",
        help=(
            "Record decode-only step samples and fit offline TAPER latency "
            "coefficients. This is implied by --write-taper-latency-profile."
        ),
    )
    parser.add_argument(
        "--taper-rho",
        type=float,
        default=1.0,
        help="Fraction of residual slack spent on opportunistic width.",
    )
    parser.add_argument(
        "--taper-utility",
        type=str,
        default="linear",
        choices=["linear", "concave"],
        help="Utility curve for the Algorithm 1 greedy planner.",
    )
    parser.add_argument("--taper-memory-threshold-blocks", type=int, default=-1)
    parser.add_argument("--taper-calibration-alpha", type=float, default=0.08)
    parser.add_argument("--taper-safety-factor", type=float, default=1.10)
    parser.add_argument(
        "--disable-taper-online-calibration",
        action="store_true",
        help="Disable observed-step online latency calibration.",
    )
    parser.add_argument(
        "--enable-taper-eager-when-unpressured",
        action="store_true",
        help="Enable the old experimental eager bypass outside Algorithm 1.",
    )
    parser.add_argument(
        "--disable-taper-branch-overdue-boost",
        action="store_true",
        help=(
            "Disable the BoN branch-starvation boost so --mode taper follows "
            "the paper-style Algorithm 1 admission rule more closely."
        ),
    )
    parser.add_argument(
        "--tail-blend-pilot-k",
        "--eager-tail-blend-pilot-k",
        dest="tail_blend_pilot_k",
        type=int,
        default=0,
        help=(
            "For --mode tail_blend, admit only this many initial BoN "
            "siblings per logical parent under pressure, then expand parents "
            "using finished-sibling length hints. 0 keeps stock eager admission."
        ),
    )
    parser.add_argument(
        "--tail-blend-reservation-q",
        "--eager-tail-blend-reservation-q",
        dest="tail_blend_reservation_q",
        type=float,
        default=0.0,
        help=(
            "For --mode tail_blend, use an online parent output-length "
            "quantile as a virtual KV reservation gate for waiting admission. "
            "0 disables it; values like 0.5 or 0.75 reserve less than max_tokens."
        ),
    )
    parser.add_argument(
        "--tail-blend-admission-mode",
        dest="tail_blend_admission_mode",
        type=str,
        default="off",
        choices=["off", "ttft_guarded"],
        help=(
            "For --mode tail_blend, keep eager admission by default or enable "
            "a guarded TTFT-risk controller that limits extra BoN siblings "
            "only under clear first-token pressure."
        ),
    )
    parser.add_argument(
        "--enable-tail-blend-preemption",
        "--enable-eager-tail-blend-preemption",
        dest="enable_tail_blend_preemption",
        action="store_true",
        help=(
            "For --mode tail_blend, choose KV-pressure preemption "
            "victims with online BoN length hints instead of default LIFO."
        ),
    )
    parser.add_argument(
        "--tail-blend-preemption-mode",
        "--eager-tail-blend-preemption-mode",
        dest="tail_blend_preemption_mode",
        type=str,
        default="full",
        choices=[
            "remaining",
            "remaining_recompute",
            "remaining_parent",
            "full",
            "full_v2",
            "full_slo",
            "complicated",
            "simplified",
            "prefill_aware",
            "prefill_aware_gated",
            "adaptive",
        ],
        help="Ablation mode for predictor-aware preemption victim scoring.",
    )
    parser = EngineArgs.add_cli_args(parser)
    parser.set_defaults(disable_log_stats=False)
    return parser


def main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark stock BoN scheduling against TAPER+."
    )
    parser = add_cli_args(parser)
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.mode in {
        "both",
        "both_plus",
        "both_tail_blend",
        "both_bon",
        "both_bon_plus",
        "both_bon_len_predictor",
        "all",
    }:
        run_both(args)
        return

    result = run_one_mode(args)
    print_single(result)


if __name__ == "__main__":
    main()
