# SPDX-License-Identifier: Apache-2.0
"""Evaluate online output-length prediction for BoN sibling branches.

This script is intentionally separate from the scheduler. It answers a smaller
question first: after a few BoN sibling branches finish, can their observed
lengths help predict the final lengths of the remaining sibling branches?

Typical flow:

1. Generate BoN samples on MATH-500 and write branch lengths to JSONL.
2. Evaluate predictors offline from the JSONL, with no GPU required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


MATH500_DATASET = "HuggingFaceH4/MATH-500"

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def prompt_from_math500_row(row: dict[str, Any]) -> str:
    problem = row.get("problem") or row.get("Problem") or row.get("question")
    if not problem:
        raise ValueError(f"Could not find problem text in row keys: {row.keys()}")
    return (
        "Solve the following math problem. Show your reasoning and put the final "
        "answer in \\boxed{}.\n\n"
        f"{problem}"
    )


def load_math500_prompts(limit: int | None) -> list[tuple[int, str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The generate command needs the `datasets` package. Install it or "
            "prepare an input JSONL and run the eval command only."
        ) from exc

    dataset = load_dataset(MATH500_DATASET, split="test")
    rows: list[tuple[int, str]] = []
    for idx, row in enumerate(dataset):
        if limit is not None and len(rows) >= limit:
            break
        rows.append((idx, prompt_from_math500_row(dict(row))))
    return rows


def load_jsonl_prompts(path: Path, limit: int | None) -> list[tuple[int, str, dict[str, Any]]]:
    rows: list[tuple[int, str, dict[str, Any]]] = []
    for idx, record in enumerate(read_jsonl(path)):
        if limit is not None and len(rows) >= limit:
            break
        prompt = record.get("prompt") or record.get("problem") or record.get("Problem")
        if not prompt:
            raise ValueError(f"No prompt/problem field in JSONL row {idx}: {record.keys()}")
        problem_index = int(record.get("problem_index", idx))
        rows.append((problem_index, str(prompt), record))
    return rows


def load_prompts_for_generation(
    prompt_jsonl: str | None,
    limit: int | None,
) -> list[tuple[int, str, dict[str, Any] | None]]:
    if prompt_jsonl:
        return load_jsonl_prompts(Path(prompt_jsonl), limit)
    return [(idx, prompt, None) for idx, prompt in load_math500_prompts(limit)]


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def avg_logprob(cumulative_logprob: float | None, length: int) -> float | None:
    if cumulative_logprob is None or length <= 0:
        return None
    return cumulative_logprob / length


def window_mean(values: list[float], window: int, from_end: bool) -> float | None:
    if not values:
        return None
    selected = values[-window:] if from_end else values[:window]
    return statistics.fmean(selected)


def score_generated_token_logprobs(
    model: Any,
    sequence: Any,
    prompt_len: int,
) -> list[float]:
    """Return log p(token_t | prefix_<t) for generated tokens in one sequence."""
    import torch

    device = next(model.parameters()).device
    input_ids = sequence.unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0]
        generated = input_ids[0, prompt_len:]
        if generated.numel() == 0:
            return []
        pred_logits = logits[prompt_len - 1:-1]
        token_logprobs = torch.log_softmax(pred_logits, dim=-1).gather(
            1, generated.unsqueeze(1)
        ).squeeze(1)
    result = [float(x) for x in token_logprobs.detach().cpu()]
    del logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def round_optional(value: float | None, ndigits: int = 6) -> float | None:
    value = finite_or_none(value)
    return round(value, ndigits) if value is not None else None


def score_generated_token_stats(
    model: Any,
    sequence: Any,
    prompt_len: int,
    eos_token_id: int | None,
) -> dict[str, list[float]]:
    """Return per-generated-token scoring signals for one sequence.

    This is intentionally generation-time data capture for BoN-TAPER offline
    analysis. It stores raw branch-specific signals so later replay scripts do
    not need to infer or leave feature columns blank.
    """
    import torch

    device = next(model.parameters()).device
    input_ids = sequence.unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0]
        generated = input_ids[0, prompt_len:]
        if generated.numel() == 0:
            return {
                "token_logprobs": [],
                "token_entropies": [],
                "token_eos_logprobs": [],
            }
        pred_logits = logits[prompt_len - 1:-1]
        log_probs = torch.log_softmax(pred_logits, dim=-1)
        probs = log_probs.exp()
        token_logprobs = log_probs.gather(1, generated.unsqueeze(1)).squeeze(1)
        entropies = -(probs * log_probs).sum(dim=-1)
        if eos_token_id is None:
            eos_logprobs = torch.empty(0, device=log_probs.device)
        else:
            eos_logprobs = log_probs[:, eos_token_id]

    result = {
        "token_logprobs": [float(x) for x in token_logprobs.detach().cpu()],
        "token_entropies": [float(x) for x in entropies.detach().cpu()],
        "token_eos_logprobs": (
            [float(x) for x in eos_logprobs.detach().cpu()]
            if eos_token_id is not None
            else []
        ),
    }
    del logits, log_probs, probs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def prefix_signal_summary(
    *,
    token_logprobs: list[float] | None,
    token_entropies: list[float] | None,
    token_eos_logprobs: list[float] | None,
    token_ids: list[int] | None,
    output_len: int,
    checkpoints: list[int],
    window: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        if checkpoint <= 0 or output_len <= checkpoint:
            continue
        logprob_prefix = (
            token_logprobs[:checkpoint] if token_logprobs is not None else []
        )
        entropy_prefix = (
            token_entropies[:checkpoint] if token_entropies is not None else []
        )
        eos_prefix = (
            token_eos_logprobs[:checkpoint]
            if token_eos_logprobs is not None
            else []
        )
        token_prefix = token_ids[:checkpoint] if token_ids is not None else []
        first_lp = (
            window_mean(logprob_prefix, window, from_end=False)
            if logprob_prefix
            else None
        )
        recent_lp = (
            window_mean(logprob_prefix, window, from_end=True)
            if logprob_prefix
            else None
        )
        first_entropy = (
            window_mean(entropy_prefix, window, from_end=False)
            if entropy_prefix
            else None
        )
        recent_entropy = (
            window_mean(entropy_prefix, window, from_end=True)
            if entropy_prefix
            else None
        )
        recent_eos_logprob = (
            window_mean(eos_prefix, window, from_end=True) if eos_prefix else None
        )
        repeat_4gram_rate = None
        if len(token_prefix) >= 4:
            grams = [
                tuple(token_prefix[i : i + 4])
                for i in range(0, len(token_prefix) - 3)
            ]
            repeat_4gram_rate = 1.0 - len(set(grams)) / max(1, len(grams))
        summaries.append({
            "prefix_len": checkpoint,
            "avg_logprob": round_optional(safe_mean(logprob_prefix)),
            "first_logprob_mean": round_optional(first_lp),
            "recent_logprob_mean": round_optional(recent_lp),
            "logprob_delta": round_optional(
                recent_lp - first_lp
                if recent_lp is not None and first_lp is not None
                else None
            ),
            "avg_entropy": round_optional(safe_mean(entropy_prefix)),
            "first_entropy_mean": round_optional(first_entropy),
            "recent_entropy_mean": round_optional(recent_entropy),
            "entropy_delta": round_optional(
                recent_entropy - first_entropy
                if recent_entropy is not None and first_entropy is not None
                else None
            ),
            "recent_eos_logprob_mean": round_optional(recent_eos_logprob),
            "recent_eos_prob_mean": round_optional(
                math.exp(recent_eos_logprob)
                if recent_eos_logprob is not None
                else None
            ),
            "repeat_4gram_rate": round_optional(repeat_4gram_rate),
        })
    return summaries


def generate_math500_bon(args: argparse.Namespace) -> None:
    if args.backend == "transformers":
        generate_math500_bon_transformers(args)
        return

    if args.record_token_stats:
        raise ValueError(
            "--record-token-stats currently requires --backend transformers "
            "because it captures entropy and EOS probabilities from logits."
        )

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "The generate command must be run from an environment where vLLM is "
            "importable. The eval command does not require vLLM."
        ) from exc

    prompts = load_prompts_for_generation(args.prompt_jsonl, args.limit)
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    output_path = Path(args.output_jsonl)
    all_records: list[dict[str, Any]] = []
    count = 0
    output_handle = None
    if args.stream_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w", encoding="utf-8")
    try:
        for bon_n in args.bon_ns:
            sampling_params = SamplingParams(
                n=bon_n,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=args.seed,
                logprobs=args.logprobs,
            )
            for batch in chunks(prompts, args.batch_size):
                batch_prompts = [prompt for _, prompt, _ in batch]
                outputs = llm.generate(batch_prompts, sampling_params)
                for (problem_index, prompt, source_record), request_output in zip(batch, outputs):
                    branches: list[dict[str, Any]] = []
                    for fallback_index, completion in enumerate(request_output.outputs):
                        token_ids = completion.token_ids or []
                        cumulative = getattr(completion, "cumulative_logprob", None)
                        branch_index = getattr(completion, "index", fallback_index)
                        branches.append({
                            "branch_index": branch_index,
                            "output_len": len(token_ids),
                            "finish_reason": getattr(
                                completion, "finish_reason", None
                            ),
                            "stop_reason": getattr(completion, "stop_reason", None),
                            "cumulative_logprob": cumulative,
                            "avg_logprob": avg_logprob(cumulative, len(token_ids)),
                        })
                    record = {
                        "dataset": MATH500_DATASET,
                        "source_dataset": (
                            source_record.get("dataset")
                            if source_record is not None
                            else MATH500_DATASET
                        ),
                        "problem_index": problem_index,
                        "source_id": (
                            source_record.get("id")
                            if source_record is not None
                            else None
                        ),
                        "year": (
                            source_record.get("year")
                            if source_record is not None
                            else None
                        ),
                        "request_id": f"math500-{problem_index}-bon{bon_n}",
                        "bon_n": bon_n,
                        "max_tokens": args.max_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "seed": args.seed,
                        "prompt": None if args.omit_prompts else prompt,
                        "branches": sorted(
                            branches, key=lambda item: item["branch_index"]
                        ),
                    }
                    count += 1
                    if output_handle is not None:
                        write_jsonl_record(output_handle, record)
                    else:
                        all_records.append(record)
    finally:
        if output_handle is not None:
            output_handle.close()

    if output_handle is None:
        write_jsonl(output_path, all_records)
    print(f"Wrote {count if args.stream_output else len(all_records)} parent records to {args.output_jsonl}")


def generate_math500_bon_transformers(args: argparse.Namespace) -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "The transformers backend needs torch and transformers installed."
        ) from exc

    if args.record_token_stats:
        args.record_token_logprobs = True
        args.store_token_ids = True

    prompts = load_prompts_for_generation(args.prompt_jsonl, args.limit)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map not in ("", "none", "None"):
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.device_map in ("", "none", "None"):
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    output_path = Path(args.output_jsonl)
    all_records: list[dict[str, Any]] = []
    count = 0
    output_handle = None
    if args.stream_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w", encoding="utf-8")
    try:
        for bon_n in args.bon_ns:
            for batch in chunks(prompts, args.batch_size):
                model_prompts: list[str] = []
                for _, prompt, _ in batch:
                    model_prompt = prompt
                    if args.use_chat_template:
                        model_prompt = tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    model_prompts.append(model_prompt)
                inputs = tokenizer(
                    model_prompts,
                    return_tensors="pt",
                    padding=True,
                ).to(model.device)
                generated_start = int(inputs["input_ids"].shape[-1])
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        do_sample=args.temperature > 0,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_tokens,
                        num_return_sequences=bon_n,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                for batch_index, (problem_index, prompt, source_record) in enumerate(batch):
                    start = batch_index * bon_n
                    end = start + bon_n
                    branches: list[dict[str, Any]] = []
                    for branch_index, sequence in enumerate(outputs[start:end]):
                        token_ids = sequence[generated_start:].tolist()
                        if tokenizer.eos_token_id is not None:
                            try:
                                eos_pos = token_ids.index(tokenizer.eos_token_id)
                                output_len = eos_pos + 1
                                finish_reason = "eos"
                            except ValueError:
                                output_len = len(token_ids)
                                finish_reason = "length"
                        else:
                            output_len = len(token_ids)
                            finish_reason = "length"
                        token_stats = None
                        if args.record_token_stats:
                            token_stats = score_generated_token_stats(
                                model,
                                sequence,
                                generated_start,
                                tokenizer.eos_token_id,
                            )
                            token_logprobs = token_stats["token_logprobs"]
                        elif args.record_token_logprobs:
                            token_logprobs = score_generated_token_logprobs(
                                model, sequence, generated_start
                            )
                        else:
                            token_logprobs = None
                        trimmed_logprobs = (
                            token_logprobs[:output_len]
                            if token_logprobs is not None
                            else None
                        )
                        trimmed_entropies = (
                            token_stats["token_entropies"][:output_len]
                            if token_stats is not None
                            else None
                        )
                        trimmed_eos_logprobs = (
                            token_stats["token_eos_logprobs"][:output_len]
                            if token_stats is not None
                            and token_stats["token_eos_logprobs"]
                            else None
                        )
                        trimmed_token_ids = token_ids[:output_len]
                        branch_record = {
                            "branch_index": branch_index,
                            "output_len": output_len,
                            "finish_reason": finish_reason,
                            "stop_reason": None,
                            "cumulative_logprob": None,
                            "avg_logprob": (
                                safe_mean(trimmed_logprobs)
                                if trimmed_logprobs
                                else None
                            ),
                        }
                        if args.store_token_ids:
                            branch_record["token_ids"] = trimmed_token_ids
                        if trimmed_logprobs is not None:
                            branch_record["token_logprobs"] = [
                                round(value, 6) for value in trimmed_logprobs
                            ]
                            branch_record["first_logprob_window_mean"] = window_mean(
                                trimmed_logprobs, args.logprob_window, from_end=False
                            )
                            branch_record["last_logprob_window_mean"] = window_mean(
                                trimmed_logprobs, args.logprob_window, from_end=True
                            )
                            first_lp = branch_record["first_logprob_window_mean"]
                            last_lp = branch_record["last_logprob_window_mean"]
                            branch_record["logprob_window_delta"] = (
                                last_lp - first_lp
                                if first_lp is not None and last_lp is not None
                                else None
                            )
                        if trimmed_entropies is not None:
                            branch_record["token_entropies"] = [
                                round(value, 6) for value in trimmed_entropies
                            ]
                            branch_record["first_entropy_window_mean"] = (
                                round_optional(
                                    window_mean(
                                        trimmed_entropies,
                                        args.logprob_window,
                                        from_end=False,
                                    )
                                )
                            )
                            branch_record["last_entropy_window_mean"] = (
                                round_optional(
                                    window_mean(
                                        trimmed_entropies,
                                        args.logprob_window,
                                        from_end=True,
                                    )
                                )
                            )
                            first_entropy = branch_record[
                                "first_entropy_window_mean"
                            ]
                            last_entropy = branch_record[
                                "last_entropy_window_mean"
                            ]
                            branch_record["entropy_window_delta"] = (
                                round_optional(last_entropy - first_entropy)
                                if first_entropy is not None
                                and last_entropy is not None
                                else None
                            )
                        if trimmed_eos_logprobs is not None:
                            branch_record["token_eos_logprobs"] = [
                                round(value, 6) for value in trimmed_eos_logprobs
                            ]
                            recent_eos_logprob = window_mean(
                                trimmed_eos_logprobs,
                                args.logprob_window,
                                from_end=True,
                            )
                            branch_record["last_eos_logprob_window_mean"] = (
                                round_optional(recent_eos_logprob)
                            )
                            branch_record["last_eos_prob_window_mean"] = (
                                round_optional(
                                    math.exp(recent_eos_logprob)
                                    if recent_eos_logprob is not None
                                    else None
                                )
                            )
                        if args.record_token_stats:
                            branch_record["prefix_signal_summaries"] = (
                                prefix_signal_summary(
                                    token_logprobs=trimmed_logprobs,
                                    token_entropies=trimmed_entropies,
                                    token_eos_logprobs=trimmed_eos_logprobs,
                                    token_ids=(
                                        trimmed_token_ids
                                        if args.store_token_ids
                                        else None
                                    ),
                                    output_len=output_len,
                                    checkpoints=args.feature_checkpoints,
                                    window=args.logprob_window,
                                )
                            )
                        branches.append(branch_record)
                    record = {
                        "dataset": MATH500_DATASET,
                        "source_dataset": (
                            source_record.get("dataset")
                            if source_record is not None
                            else MATH500_DATASET
                        ),
                        "backend": "transformers",
                        "problem_index": problem_index,
                        "source_id": (
                            source_record.get("id")
                            if source_record is not None
                            else None
                        ),
                        "year": (
                            source_record.get("year")
                            if source_record is not None
                            else None
                        ),
                        "request_id": f"math500-{problem_index}-bon{bon_n}",
                        "bon_n": bon_n,
                        "max_tokens": args.max_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "seed": args.seed,
                        "prompt": None if args.omit_prompts else prompt,
                        "used_chat_template": args.use_chat_template,
                        "branches": branches,
                    }
                    count += 1
                    if output_handle is not None:
                        write_jsonl_record(output_handle, record)
                    else:
                        all_records.append(record)
    finally:
        if output_handle is not None:
            output_handle.close()

    if output_handle is None:
        write_jsonl(output_path, all_records)
    print(f"Wrote {count if args.stream_output else len(all_records)} parent records to {args.output_jsonl}")


def safe_mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def softmax_weighted_length(
    finished: list[dict[str, Any]],
    temperature: float,
) -> float | None:
    scored = [
        (float(branch["output_len"]), branch.get("avg_logprob"))
        for branch in finished
        if branch.get("avg_logprob") is not None
    ]
    if not scored:
        return None
    max_score = max(float(score) for _, score in scored)
    weights = [
        math.exp((float(score) - max_score) / max(temperature, 1e-6))
        for _, score in scored
    ]
    denom = sum(weights)
    if denom <= 0.0:
        return None
    return sum(length * weight for (length, _), weight in zip(scored, weights)) / denom


def build_predictions(
    finished: list[dict[str, Any]],
    kth_finished_len: float,
    prior_len: float,
    shrink_k: float,
    logprob_temperature: float,
    conservative_floor: float,
) -> tuple[dict[str, float], dict[str, float]]:
    finished_lengths = [float(branch["output_len"]) for branch in finished]
    weight = len(finished_lengths) / (len(finished_lengths) + shrink_k)
    sibling_mean = safe_mean(finished_lengths)
    sibling_median = statistics.median(finished_lengths)
    sibling_std = (
        statistics.pstdev(finished_lengths)
        if len(finished_lengths) > 1
        else 0.0
    )
    sibling_cv = sibling_std / max(sibling_mean, 1.0)

    mean_shrink = (1.0 - weight) * prior_len + weight * sibling_mean
    median_shrink = (1.0 - weight) * prior_len + weight * sibling_median

    logprob_weighted = softmax_weighted_length(finished, logprob_temperature)
    if logprob_weighted is None:
        logprob_shrink = mean_shrink
    else:
        logprob_shrink = (1.0 - weight) * prior_len + weight * logprob_weighted

    # If a branch has not finished by the time the kth sibling has finished,
    # its final length is censored from below: it is at least kth_finished_len.
    lower_bound = kth_finished_len + conservative_floor
    tail_mean = 0.5 * (lower_bound + prior_len)
    tail_p75 = lower_bound + 0.75 * max(0.0, prior_len - lower_bound)
    tail_p90 = lower_bound + 0.90 * max(0.0, prior_len - lower_bound)

    # V2: order-statistics aware predictors. Finished siblings are the early
    # order statistics, so their raw mean is biased short. These estimates use
    # the fact that unfinished branches are in the tail above kth_finished_len.
    v2_tail_blend = (1.0 - weight) * tail_mean + weight * sibling_mean
    v2_tail_median_blend = (1.0 - weight) * tail_mean + weight * sibling_median
    v2_logprob_blend = (
        (1.0 - weight) * tail_mean
        + weight * (
            logprob_weighted
            if logprob_weighted is not None
            else sibling_mean
        )
    )

    # V3 confidence is a lightweight online signal for scheduler use. It is
    # high when we have more finished siblings, their lengths are consistent,
    # and the kth finished branch already gives a meaningful lower bound.
    evidence_confidence = weight
    consistency_confidence = 1.0 / (1.0 + sibling_cv)
    lower_bound_confidence = clamp(lower_bound / max(prior_len, 1.0), 0.0, 1.0)
    confidence = clamp(
        evidence_confidence
        * consistency_confidence
        * (0.5 + 0.5 * lower_bound_confidence),
        0.0,
        1.0,
    )

    v2_tail_median_prediction = max(v2_tail_median_blend, lower_bound)
    v3_confidence_tail_blend = (
        confidence * v2_tail_median_prediction
        + (1.0 - confidence) * tail_mean
    )
    risk_adjustment = (1.0 - confidence) * 0.15 * max(0.0, prior_len - lower_bound)
    v3_risk_adjusted = min(prior_len, v2_tail_median_prediction + risk_adjustment)
    v3_confidence_prior_blend = (
        confidence * v2_tail_median_prediction
        + (1.0 - confidence) * prior_len
    )

    predictions = {
        "baseline_prior": prior_len,
        "v1_sibling_mean": sibling_mean,
        "v1_sibling_mean_shrink": mean_shrink,
        "v1_sibling_mean_censored": max(mean_shrink, lower_bound),
        "v1_sibling_median_censored": max(median_shrink, lower_bound),
        "v1_logprob_weighted_censored": max(logprob_shrink, lower_bound),
        "v2_order_tail_mean": tail_mean,
        "v2_order_tail_p75": tail_p75,
        "v2_order_tail_p90": tail_p90,
        "v2_tail_mean_blend": max(v2_tail_blend, lower_bound),
        "v2_tail_median_blend": max(v2_tail_median_blend, lower_bound),
        "v2_logprob_tail_blend": max(v2_logprob_blend, lower_bound),
        "v3_confidence_tail_blend": v3_confidence_tail_blend,
        "v3_risk_adjusted_tail_blend": v3_risk_adjusted,
        "v3_confidence_prior_blend": v3_confidence_prior_blend,
    }
    confidence_info = {
        "confidence": confidence,
        "evidence_confidence": evidence_confidence,
        "consistency_confidence": consistency_confidence,
        "lower_bound_confidence": lower_bound_confidence,
        "finished_mean": sibling_mean,
        "finished_std": sibling_std,
        "finished_cv": sibling_cv,
        "lower_bound": lower_bound,
    }
    return predictions, confidence_info


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def branch_logprob_predictions(
    branch: dict[str, Any],
    finished: list[dict[str, Any]],
    kth_finished_len: float,
    lower_bound: float,
    prior_len: float,
    v2_prediction: float,
    confidence: float,
    window: int,
    scale: float,
) -> tuple[dict[str, float], dict[str, float | None]]:
    token_logprobs = branch.get("token_logprobs")
    if not token_logprobs:
        return {}, {
            "live_prefix_logprob_mean": None,
            "live_recent_logprob_mean": None,
            "live_logprob_delta": None,
            "finished_recent_logprob_mean": None,
            "logprob_long_probability": None,
        }

    cutoff = min(int(kth_finished_len), len(token_logprobs))
    prefix_logprobs = [float(value) for value in token_logprobs[:cutoff]]
    live_prefix_mean = safe_mean(prefix_logprobs)
    live_recent_mean = window_mean(prefix_logprobs, window, from_end=True)
    live_first_mean = window_mean(prefix_logprobs, window, from_end=False)
    live_delta = (
        live_recent_mean - live_first_mean
        if live_recent_mean is not None and live_first_mean is not None
        else None
    )

    finished_recent_values: list[float] = []
    for item in finished:
        item_logprobs = item.get("token_logprobs")
        if item_logprobs:
            item_recent = window_mean(
                [float(value) for value in item_logprobs],
                window,
                from_end=True,
            )
            if item_recent is not None:
                finished_recent_values.append(item_recent)
    finished_recent_mean = (
        safe_mean(finished_recent_values) if finished_recent_values else None
    )

    if live_recent_mean is None or finished_recent_mean is None:
        return {}, {
            "live_prefix_logprob_mean": live_prefix_mean,
            "live_recent_logprob_mean": live_recent_mean,
            "live_logprob_delta": live_delta,
            "finished_recent_logprob_mean": finished_recent_mean,
            "logprob_long_probability": None,
        }

    # If the live branch's recent logprob is worse than branches that already
    # finished, treat it as more likely long-lived. This is intentionally simple
    # and training-free; `scale` controls how quickly the probability changes.
    lp_gap = finished_recent_mean - live_recent_mean
    trend_penalty = max(0.0, -(live_delta or 0.0))
    p_long = sigmoid((lp_gap + 0.5 * trend_penalty) / max(scale, 1e-6))
    logprob_tail = lower_bound + p_long * max(0.0, prior_len - lower_bound)
    logprob_v2_blend = confidence * v2_prediction + (1.0 - confidence) * logprob_tail
    logprob_risk_blend = 0.7 * v2_prediction + 0.3 * logprob_tail

    features = {
        "live_prefix_logprob_mean": live_prefix_mean,
        "live_recent_logprob_mean": live_recent_mean,
        "live_logprob_delta": live_delta,
        "finished_recent_logprob_mean": finished_recent_mean,
        "logprob_long_probability": p_long,
    }
    predictions = {
        "v3_logprob_tail": logprob_tail,
        "v3_logprob_conf_blend": logprob_v2_blend,
        "v3_logprob_risk_blend": logprob_risk_blend,
    }
    return predictions, features


def summarize_errors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], list[float]] = {}
    signed: dict[tuple[int, int, str], list[float]] = {}
    confidences: dict[tuple[int, int, str], list[float]] = {}
    under: dict[tuple[int, int, str], list[float]] = {}
    for row in rows:
        key = (int(row["bon_n"]), int(row["k_finished"]), row["method"])
        grouped.setdefault(key, []).append(abs(float(row["error"])))
        signed.setdefault(key, []).append(float(row["error"]))
        under.setdefault(key, []).append(max(0.0, -float(row["error"])))
        if row.get("confidence") is not None:
            confidences.setdefault(key, []).append(float(row["confidence"]))

    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped):
        bon_n, k_finished, method = key
        abs_errors = grouped[key]
        signed_errors = signed[key]
        under_errors = under[key]
        rmse = math.sqrt(safe_mean([err * err for err in signed_errors]))
        summaries.append({
            "bon_n": bon_n,
            "k_finished": k_finished,
            "method": method,
            "num_predictions": len(abs_errors),
            "mae": safe_mean(abs_errors),
            "rmse": rmse,
            "median_abs_error": percentile(abs_errors, 0.5),
            "p90_abs_error": percentile(abs_errors, 0.9),
            "bias": safe_mean(signed_errors),
            "avg_confidence": safe_mean(confidences.get(key, [])),
            "underprediction_rate": safe_mean([
                1.0 if err > 0.0 else 0.0 for err in under_errors
            ]),
            "p90_underprediction": percentile(under_errors, 0.9),
        })
    return summaries


def evaluate_jsonl(args: argparse.Namespace) -> None:
    records = read_jsonl(Path(args.input_jsonl))
    rows: list[dict[str, Any]] = []

    for record in records:
        branches = [
            branch for branch in record.get("branches", [])
            if branch.get("output_len") is not None
        ]
        if len(branches) < 2:
            continue

        bon_n = int(record.get("bon_n", len(branches)))
        prior_len = float(record.get("max_tokens") or args.default_prior_len)
        sorted_branches = sorted(branches, key=lambda item: item["output_len"])

        for k_finished in args.k_finished:
            if k_finished <= 0 or k_finished >= len(sorted_branches):
                continue
            finished = sorted_branches[:k_finished]
            remaining = sorted_branches[k_finished:]
            kth_finished_len = float(finished[-1]["output_len"])
            predictions, confidence_info = build_predictions(
                finished=finished,
                kth_finished_len=kth_finished_len,
                prior_len=prior_len,
                shrink_k=args.shrink_k,
                logprob_temperature=args.logprob_temperature,
                conservative_floor=args.conservative_floor,
            )
            for branch in remaining:
                actual_len = float(branch["output_len"])
                branch_predictions, logprob_features = branch_logprob_predictions(
                    branch=branch,
                    finished=finished,
                    kth_finished_len=kth_finished_len,
                    lower_bound=confidence_info["lower_bound"],
                    prior_len=prior_len,
                    v2_prediction=predictions["v2_tail_median_blend"],
                    confidence=confidence_info["confidence"],
                    window=args.logprob_window,
                    scale=args.logprob_scale,
                )
                all_predictions = {**predictions, **branch_predictions}
                for method, predicted_len in all_predictions.items():
                    rows.append({
                        "request_id": record.get("request_id"),
                        "problem_index": record.get("problem_index"),
                        "bon_n": bon_n,
                        "k_finished": k_finished,
                        "method": method,
                        "branch_index": branch.get("branch_index"),
                        "actual_len": actual_len,
                        "predicted_len": predicted_len,
                        "error": predicted_len - actual_len,
                        "abs_error": abs(predicted_len - actual_len),
                        **confidence_info,
                        **logprob_features,
                    })

    summaries = summarize_errors(rows)
    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "bon_n",
                "k_finished",
                "method",
                "num_predictions",
                "mae",
                "rmse",
                "median_abs_error",
                "p90_abs_error",
                "bias",
                "avg_confidence",
                "underprediction_rate",
                "p90_underprediction",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)

    if args.predictions_csv:
        predictions_csv = Path(args.predictions_csv)
        predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        with predictions_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "request_id",
                "problem_index",
                "bon_n",
                "k_finished",
                "method",
                "branch_index",
                "actual_len",
                "predicted_len",
                "error",
                "abs_error",
                "confidence",
                "evidence_confidence",
                "consistency_confidence",
                "lower_bound_confidence",
                "finished_mean",
                "finished_std",
                "finished_cv",
                "lower_bound",
                "live_prefix_logprob_mean",
                "live_recent_logprob_mean",
                "live_logprob_delta",
                "finished_recent_logprob_mean",
                "logprob_long_probability",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    print_summary_table(summaries)
    if args.output_csv:
        print(f"\nWrote summary CSV to {args.output_csv}")
    if args.predictions_csv:
        print(f"Wrote per-branch predictions to {args.predictions_csv}")


def split_records(
    records: list[dict[str, Any]],
    train_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        records,
        key=lambda item: (
            int(item.get("problem_index", 0)),
            int(item.get("bon_n", 0)),
            str(item.get("request_id", "")),
        ),
    )
    problem_ids = sorted({int(item.get("problem_index", 0)) for item in ordered})
    split_at = max(1, min(len(problem_ids) - 1, int(len(problem_ids) * train_fraction)))
    train_ids = set(problem_ids[:split_at])
    train = [item for item in ordered if int(item.get("problem_index", 0)) in train_ids]
    test = [item for item in ordered if int(item.get("problem_index", 0)) not in train_ids]
    return train, test


def calibration_samples(
    records: list[dict[str, Any]],
    k_finished_values: list[int],
    num_bins: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record in records:
        branches = [
            branch for branch in record.get("branches", [])
            if branch.get("output_len") is not None
        ]
        if len(branches) < 2:
            continue
        bon_n = int(record.get("bon_n", len(branches)))
        prior_len = float(record.get("max_tokens") or 1.0)
        sorted_branches = sorted(branches, key=lambda item: item["output_len"])
        for k_finished in k_finished_values:
            if k_finished <= 0 or k_finished >= len(sorted_branches):
                continue
            finished = sorted_branches[:k_finished]
            remaining = sorted_branches[k_finished:]
            kth_finished_len = float(finished[-1]["output_len"])
            kth_norm = kth_finished_len / max(prior_len, 1.0)
            bin_index = min(num_bins - 1, max(0, int(kth_norm * num_bins)))
            finished_lengths = [float(item["output_len"]) for item in finished]
            for branch in remaining:
                actual_len = float(branch["output_len"])
                samples.append({
                    "request_id": record.get("request_id"),
                    "problem_index": int(record.get("problem_index", 0)),
                    "bon_n": bon_n,
                    "k_finished": k_finished,
                    "prior_len": prior_len,
                    "kth_finished_len": kth_finished_len,
                    "kth_norm": kth_norm,
                    "bin_index": bin_index,
                    "finished_mean_norm": safe_mean(finished_lengths) / prior_len,
                    "finished_std_norm": (
                        statistics.pstdev(finished_lengths) / prior_len
                        if len(finished_lengths) > 1
                        else 0.0
                    ),
                    "actual_norm": actual_len / prior_len,
                    "actual_len": actual_len,
                    "branch_index": branch.get("branch_index"),
                })
    return samples


def fit_order_calibration(
    samples: list[dict[str, Any]],
    min_bin_samples: int,
) -> dict[tuple[str, int | None, int | None, int | None], dict[str, float]]:
    grouped: dict[tuple[str, int | None, int | None, int | None], list[float]] = {}

    def add(key: tuple[str, int | None, int | None, int | None],
            value: float) -> None:
        grouped.setdefault(key, []).append(value)

    for sample in samples:
        bon_n = int(sample["bon_n"])
        k_finished = int(sample["k_finished"])
        bin_index = int(sample["bin_index"])
        value = float(sample["actual_norm"])
        add(("exact", bon_n, k_finished, bin_index), value)
        add(("nk", bon_n, k_finished, None), value)
        add(("n", bon_n, None, None), value)
        add(("global", None, None, None), value)

    table: dict[tuple[str, int | None, int | None, int | None], dict[str, float]] = {}
    for key, values in grouped.items():
        if key[0] == "exact" and len(values) < min_bin_samples:
            continue
        table[key] = {
            "count": float(len(values)),
            "mean": safe_mean(values),
            "median": percentile(values, 0.5),
            "p75": percentile(values, 0.75),
            "p90": percentile(values, 0.9),
        }
    return table


def lookup_calibration(
    table: dict[tuple[str, int | None, int | None, int | None], dict[str, float]],
    bon_n: int,
    k_finished: int,
    bin_index: int,
) -> dict[str, float]:
    for key in (
        ("exact", bon_n, k_finished, bin_index),
        ("nk", bon_n, k_finished, None),
        ("n", bon_n, None, None),
        ("global", None, None, None),
    ):
        if key in table:
            return table[key]
    return {"count": 0.0, "mean": 1.0, "median": 1.0, "p75": 1.0, "p90": 1.0}


def calibrated_predictions_for_sample(
    sample: dict[str, Any],
    table: dict[tuple[str, int | None, int | None, int | None], dict[str, float]],
) -> dict[str, float]:
    stats = lookup_calibration(
        table,
        int(sample["bon_n"]),
        int(sample["k_finished"]),
        int(sample["bin_index"]),
    )
    prior_len = float(sample["prior_len"])
    lower_bound = float(sample["kth_finished_len"]) + 1.0
    return {
        "v4_calib_mean": max(lower_bound, stats["mean"] * prior_len),
        "v4_calib_median": max(lower_bound, stats["median"] * prior_len),
        "v4_calib_p75": max(lower_bound, stats["p75"] * prior_len),
        "v4_calib_p90": max(lower_bound, stats["p90"] * prior_len),
    }


def calibrate_jsonl(args: argparse.Namespace) -> None:
    records = read_jsonl(Path(args.input_jsonl))
    train_records, test_records = split_records(records, args.train_fraction)
    train_samples = calibration_samples(
        train_records,
        args.k_finished,
        args.num_bins,
    )
    test_samples = calibration_samples(
        test_records,
        args.k_finished,
        args.num_bins,
    )
    table = fit_order_calibration(train_samples, args.min_bin_samples)

    rows: list[dict[str, Any]] = []
    for sample in test_samples:
        predictions = calibrated_predictions_for_sample(sample, table)
        # Include baseline/v2 context on the held-out split for direct comparison.
        pseudo_record = next(
            record for record in test_records
            if record.get("request_id") == sample["request_id"]
        )
        sorted_branches = sorted(
            pseudo_record["branches"], key=lambda item: item["output_len"]
        )
        finished = sorted_branches[:int(sample["k_finished"])]
        kth_finished_len = float(sample["kth_finished_len"])
        base_predictions, confidence_info = build_predictions(
            finished=finished,
            kth_finished_len=kth_finished_len,
            prior_len=float(sample["prior_len"]),
            shrink_k=args.shrink_k,
            logprob_temperature=args.logprob_temperature,
            conservative_floor=args.conservative_floor,
        )
        selected_base = {
            key: value for key, value in base_predictions.items()
            if key in {
                "baseline_prior",
                "v1_sibling_mean",
                "v1_sibling_mean_censored",
                "v2_tail_median_blend",
                "v2_logprob_tail_blend",
            }
        }
        all_predictions = {**selected_base, **predictions}
        actual_len = float(sample["actual_len"])
        for method, predicted_len in all_predictions.items():
            rows.append({
                "request_id": sample["request_id"],
                "problem_index": sample["problem_index"],
                "bon_n": sample["bon_n"],
                "k_finished": sample["k_finished"],
                "method": method,
                "branch_index": sample["branch_index"],
                "actual_len": actual_len,
                "predicted_len": predicted_len,
                "error": predicted_len - actual_len,
                "abs_error": abs(predicted_len - actual_len),
                "bin_index": sample["bin_index"],
                **confidence_info,
            })

    summaries = summarize_errors(rows)
    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "bon_n",
                "k_finished",
                "method",
                "num_predictions",
                "mae",
                "rmse",
                "median_abs_error",
                "p90_abs_error",
                "bias",
                "avg_confidence",
                "underprediction_rate",
                "p90_underprediction",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)

    if args.predictions_csv:
        predictions_csv = Path(args.predictions_csv)
        predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        with predictions_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "request_id",
                "problem_index",
                "bon_n",
                "k_finished",
                "method",
                "branch_index",
                "actual_len",
                "predicted_len",
                "error",
                "abs_error",
                "bin_index",
                "confidence",
                "lower_bound",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    if args.table_json:
        serializable = [
            {
                "kind": key[0],
                "bon_n": key[1],
                "k_finished": key[2],
                "bin_index": key[3],
                **value,
            }
            for key, value in sorted(table.items(), key=lambda item: str(item[0]))
        ]
        Path(args.table_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.table_json).write_text(
            json.dumps(serializable, indent=2),
            encoding="utf-8",
        )

    print(
        f"Calibration records: train={len(train_records)}, test={len(test_records)}, "
        f"train_samples={len(train_samples)}, test_samples={len(test_samples)}, "
        f"table_entries={len(table)}"
    )
    print_summary_table(summaries)
    if args.output_csv:
        print(f"\nWrote calibration summary CSV to {args.output_csv}")
    if args.predictions_csv:
        print(f"Wrote calibration predictions CSV to {args.predictions_csv}")
    if args.table_json:
        print(f"Wrote calibration table JSON to {args.table_json}")


def print_summary_table(summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        print("No valid prediction rows.")
        return
    print("\nBoN sibling output-length prediction")
    print("=" * 92)
    print(
        f"{'N':>4} {'k':>4} {'method':<28} {'n':>8} "
        f"{'MAE':>10} {'RMSE':>10} {'P90 AE':>10} {'bias':>10} "
        f"{'under%':>8} {'conf':>8}"
    )
    print("-" * 92)
    for row in summaries:
        print(
            f"{row['bon_n']:>4} {row['k_finished']:>4} "
            f"{row['method']:<28} {row['num_predictions']:>8} "
            f"{row['mae']:>10.2f} {row['rmse']:>10.2f} "
            f"{row['p90_abs_error']:>10.2f} {row['bias']:>10.2f} "
            f"{100.0 * row['underprediction_rate']:>7.1f}% "
            f"{row['avg_confidence']:>8.3f}"
        )


def add_generate_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "generate",
        help="Generate MATH-500 BoN branch-length records with vLLM.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--backend",
        choices=["vllm", "transformers"],
        default="vllm",
        help="Use vLLM when available; transformers is enough for length data.",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Write each parent record immediately instead of at process exit.",
    )
    parser.add_argument(
        "--prompt-jsonl",
        default=None,
        help="Optional JSONL prompt file with prompt/problem fields.",
    )
    parser.add_argument("--bon-ns", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logprobs", type=int, default=0)
    parser.add_argument("--record-token-logprobs", action="store_true")
    parser.add_argument(
        "--record-token-stats",
        action="store_true",
        help=(
            "With --backend transformers, store token logprobs, entropies, "
            "EOS logprobs, and prefix checkpoint summaries for BoN-TAPER "
            "feature analysis."
        ),
    )
    parser.add_argument("--store-token-ids", action="store_true")
    parser.add_argument("--logprob-window", type=int, default=32)
    parser.add_argument(
        "--feature-checkpoints",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256, 512, 1024],
        help="Prefix lengths summarized when --record-token-stats is enabled.",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--device-map",
        default="auto",
        help=(
            "Transformers device_map. Use 'none' to avoid accelerate and move "
            "the model to cuda/cpu directly."
        ),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="For transformers backend, wrap each problem as a chat user turn.",
    )
    parser.add_argument("--omit-prompts", action="store_true")
    parser.set_defaults(func=generate_math500_bon)


def add_eval_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "eval",
        help="Evaluate online sibling-length predictors from generated JSONL.",
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--predictions-csv", default=None)
    parser.add_argument("--k-finished", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--default-prior-len",
        type=float,
        default=2048.0,
        help="Fallback prior if a JSONL record does not contain max_tokens.",
    )
    parser.add_argument(
        "--shrink-k",
        type=float,
        default=4.0,
        help="Larger values trust finished siblings more slowly.",
    )
    parser.add_argument("--logprob-temperature", type=float, default=0.25)
    parser.add_argument("--logprob-window", type=int, default=32)
    parser.add_argument("--logprob-scale", type=float, default=0.5)
    parser.add_argument(
        "--conservative-floor",
        type=float,
        default=1.0,
        help=(
            "Extra lower-bound tokens added after the kth finished sibling. "
            "This encodes that unfinished branches are censored from below."
        ),
    )
    parser.set_defaults(func=evaluate_jsonl)


def add_calibrate_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "calibrate",
        help=(
            "Fit an order-statistics lookup predictor on a calibration split "
            "and evaluate on held-out BoN parents."
        ),
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--predictions-csv", default=None)
    parser.add_argument("--table-json", default=None)
    parser.add_argument("--k-finished", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--min-bin-samples", type=int, default=8)
    parser.add_argument("--shrink-k", type=float, default=4.0)
    parser.add_argument("--logprob-temperature", type=float, default=0.25)
    parser.add_argument("--conservative-floor", type=float, default=1.0)
    parser.set_defaults(func=calibrate_jsonl)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BoN online output-length prediction experiment."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_generate_args(subparsers)
    add_eval_args(subparsers)
    add_calibrate_args(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
