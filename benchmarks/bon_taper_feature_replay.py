# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build offline feature tables for BoN-TAPER lifetime prediction.

The input is the BoN branch JSONL produced by ``bon_output_length_prediction.py``.
This script replays scheduler-like decision points without modifying vLLM:

* event_k_finished: after k sibling branches have finished, build one row for
  each still-live branch using sibling stats plus optional prefix features.
* fresh_after_k: after k sibling branches have finished, build one row for a
  hypothetical unstarted sibling. This uses sibling stats only and is useful for
  BoN-TAPER admission of deferred branches.

If the JSONL includes ``token_logprobs`` or ``token_ids`` per branch, the live
rows include branch-specific prefix features. Otherwise those fields are blank,
so the same script works with lightweight length-only JSONL.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


FINISH_THRESHOLDS = (32, 64, 128, 256, 512)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def window_mean(values: list[float], window: int, *, from_end: bool) -> float | None:
    if not values:
        return None
    if window <= 0 or len(values) <= window:
        selected = values
    elif from_end:
        selected = values[-window:]
    else:
        selected = values[:window]
    return safe_mean(selected)


def ngram_repeat_rate(token_ids: list[int], ngram: int = 4) -> float | None:
    if len(token_ids) < ngram:
        return None
    grams = [
        tuple(token_ids[i : i + ngram])
        for i in range(0, len(token_ids) - ngram + 1)
    ]
    if not grams:
        return None
    return 1.0 - (len(set(grams)) / len(grams))


def finished_stats(finished: list[dict[str, Any]]) -> dict[str, float | int | None]:
    lengths = [float(item["output_len"]) for item in finished]
    if not lengths:
        return {
            "num_finished": 0,
            "finished_min": None,
            "finished_mean": None,
            "finished_median": None,
            "finished_max": None,
            "finished_std": None,
            "finished_cv": None,
            "finished_eos_frac": None,
            "finished_length_capped_frac": None,
        }

    mean = statistics.fmean(lengths)
    std = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
    eos_count = sum(1 for item in finished if item.get("finish_reason") == "eos")
    capped_count = sum(
        1 for item in finished if item.get("finish_reason") == "length"
    )
    return {
        "num_finished": len(lengths),
        "finished_min": min(lengths),
        "finished_mean": mean,
        "finished_median": statistics.median(lengths),
        "finished_max": max(lengths),
        "finished_std": std,
        "finished_cv": std / max(mean, 1.0),
        "finished_eos_frac": eos_count / len(lengths),
        "finished_length_capped_frac": capped_count / len(lengths),
    }


def live_prefix_features(
    branch: dict[str, Any],
    prefix_len: int,
    logprob_window: int,
) -> dict[str, float | int | None]:
    for summary in branch.get("prefix_signal_summaries") or []:
        if int(summary.get("prefix_len", -1)) == prefix_len:
            return {
                "live_prefix_len": prefix_len,
                "live_avg_logprob": summary.get("avg_logprob"),
                "live_first_logprob_mean": summary.get("first_logprob_mean"),
                "live_recent_logprob_mean": summary.get("recent_logprob_mean"),
                "live_logprob_delta": summary.get("logprob_delta"),
                "live_avg_entropy": summary.get("avg_entropy"),
                "live_first_entropy_mean": summary.get("first_entropy_mean"),
                "live_recent_entropy_mean": summary.get("recent_entropy_mean"),
                "live_entropy_delta": summary.get("entropy_delta"),
                "live_recent_eos_logprob_mean": summary.get(
                    "recent_eos_logprob_mean"
                ),
                "live_recent_eos_prob_mean": summary.get(
                    "recent_eos_prob_mean"
                ),
                "live_repeat_4gram_rate": summary.get("repeat_4gram_rate"),
            }

    token_logprobs = branch.get("token_logprobs") or []
    prefix_logprobs = [float(x) for x in token_logprobs[:prefix_len]]
    first_lp = window_mean(prefix_logprobs, logprob_window, from_end=False)
    recent_lp = window_mean(prefix_logprobs, logprob_window, from_end=True)
    token_entropies = branch.get("token_entropies") or []
    prefix_entropies = [float(x) for x in token_entropies[:prefix_len]]
    first_entropy = window_mean(prefix_entropies, logprob_window, from_end=False)
    recent_entropy = window_mean(prefix_entropies, logprob_window, from_end=True)
    token_eos_logprobs = branch.get("token_eos_logprobs") or []
    prefix_eos_logprobs = [float(x) for x in token_eos_logprobs[:prefix_len]]
    recent_eos_logprob = window_mean(
        prefix_eos_logprobs, logprob_window, from_end=True
    )

    token_ids = branch.get("token_ids") or []
    prefix_token_ids = [int(x) for x in token_ids[:prefix_len]]
    return {
        "live_prefix_len": prefix_len,
        "live_avg_logprob": safe_mean(prefix_logprobs),
        "live_first_logprob_mean": first_lp,
        "live_recent_logprob_mean": recent_lp,
        "live_logprob_delta": (
            recent_lp - first_lp
            if recent_lp is not None and first_lp is not None
            else None
        ),
        "live_avg_entropy": safe_mean(prefix_entropies),
        "live_first_entropy_mean": first_entropy,
        "live_recent_entropy_mean": recent_entropy,
        "live_entropy_delta": (
            recent_entropy - first_entropy
            if recent_entropy is not None and first_entropy is not None
            else None
        ),
        "live_recent_eos_logprob_mean": recent_eos_logprob,
        "live_recent_eos_prob_mean": (
            math.exp(recent_eos_logprob)
            if recent_eos_logprob is not None
            else None
        ),
        "live_repeat_4gram_rate": ngram_repeat_rate(prefix_token_ids, 4),
    }


def prediction_features(
    *,
    prefix_len: int,
    max_tokens: int,
    stats: dict[str, float | int | None],
    shrink_k: float,
) -> dict[str, float | None]:
    finished_mean = stats.get("finished_mean")
    num_finished = int(stats.get("num_finished") or 0)
    baseline_final = float(max_tokens)
    sibling_mean_final = float(finished_mean) if finished_mean is not None else None

    if sibling_mean_final is None or num_finished == 0:
        tail_blend_final = baseline_final
    else:
        weight = num_finished / (num_finished + shrink_k)
        lower_bound = float(prefix_len)
        tail_mean = 0.5 * (lower_bound + baseline_final)
        tail_blend_final = (1.0 - weight) * tail_mean + weight * sibling_mean_final
        tail_blend_final = max(lower_bound, tail_blend_final)

    return {
        "pred_baseline_remaining": max(0.0, baseline_final - prefix_len),
        "pred_sibling_mean_remaining": (
            max(0.0, sibling_mean_final - prefix_len)
            if sibling_mean_final is not None
            else None
        ),
        "pred_tail_blend_remaining": max(0.0, tail_blend_final - prefix_len),
    }


def make_row(
    *,
    record: dict[str, Any],
    branch: dict[str, Any],
    decision_type: str,
    k_finished: int,
    prefix_len: int,
    finished: list[dict[str, Any]],
    max_tokens: int,
    logprob_window: int,
    shrink_k: float,
) -> dict[str, Any]:
    actual_len = int(branch["output_len"])
    remaining_len = max(0, actual_len - prefix_len)
    stats = finished_stats(finished)
    row: dict[str, Any] = {
        "request_id": record.get("request_id"),
        "problem_index": record.get("problem_index"),
        "bon_n": record.get("bon_n", len(record.get("branches", []))),
        "branch_index": branch.get("branch_index"),
        "decision_type": decision_type,
        "k_finished": k_finished,
        "max_tokens": max_tokens,
        "actual_len": actual_len,
        "remaining_len": remaining_len,
        "current_output_len": prefix_len,
        "progress_frac": prefix_len / max(1, max_tokens),
        "finish_reason": branch.get("finish_reason"),
        "length_capped": int(
            branch.get("finish_reason") == "length" or actual_len >= max_tokens
        ),
        **stats,
        **live_prefix_features(branch, prefix_len, logprob_window),
    }
    for threshold in FINISH_THRESHOLDS:
        row[f"finish_within_{threshold}"] = int(remaining_len <= threshold)

    row.update(
        prediction_features(
            prefix_len=prefix_len,
            max_tokens=max_tokens,
            stats=stats,
            shrink_k=shrink_k,
        )
    )
    for method in (
        "baseline",
        "sibling_mean",
        "tail_blend",
    ):
        pred_key = f"pred_{method}_remaining"
        pred = row.get(pred_key)
        row[f"{method}_abs_error"] = (
            abs(float(pred) - remaining_len) if pred is not None else None
        )
    return row


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in read_jsonl(Path(args.input_jsonl)):
        branches = [
            branch
            for branch in record.get("branches", [])
            if branch.get("output_len") is not None
        ]
        if len(branches) < 2:
            continue

        max_tokens = int(record.get("max_tokens") or args.default_prior_len)
        sorted_branches = sorted(
            branches,
            key=lambda item: (int(item["output_len"]), int(item["branch_index"])),
        )

        for k_finished in args.k_finished:
            if k_finished <= 0 or k_finished >= len(sorted_branches):
                continue
            finished = sorted_branches[:k_finished]
            kth_finished_len = int(finished[-1]["output_len"])

            for branch in sorted_branches[k_finished:]:
                actual_len = int(branch["output_len"])

                # A live branch at the event point must not have finished yet.
                if actual_len > kth_finished_len:
                    rows.append(
                        make_row(
                            record=record,
                            branch=branch,
                            decision_type="event_k_finished",
                            k_finished=k_finished,
                            prefix_len=kth_finished_len,
                            finished=finished,
                            max_tokens=max_tokens,
                            logprob_window=args.logprob_window,
                            shrink_k=args.shrink_k,
                        )
                    )

                # BoN-TAPER's deferred sibling case: the sibling has not started,
                # so only sibling/parent features are available.
                rows.append(
                    make_row(
                        record=record,
                        branch=branch,
                        decision_type="fresh_after_k",
                        k_finished=k_finished,
                        prefix_len=0,
                        finished=finished,
                        max_tokens=max_tokens,
                        logprob_window=args.logprob_window,
                        shrink_k=args.shrink_k,
                    )
                )

        for prefix_len in args.prefix_lens:
            finished = [
                branch
                for branch in sorted_branches
                if int(branch["output_len"]) <= prefix_len
            ]
            if len(finished) < args.min_finished_for_prefix:
                continue
            for branch in sorted_branches:
                if int(branch["output_len"]) <= prefix_len:
                    continue
                rows.append(
                    make_row(
                        record=record,
                        branch=branch,
                        decision_type="fixed_prefix",
                        k_finished=len(finished),
                        prefix_len=prefix_len,
                        finished=finished,
                        max_tokens=max_tokens,
                        logprob_window=args.logprob_window,
                        shrink_k=args.shrink_k,
                    )
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["decision_type"]), int(row["k_finished"]))].append(row)

    summaries: list[dict[str, Any]] = []
    for (decision_type, k_finished), items in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "decision_type": decision_type,
            "k_finished": k_finished,
            "num_rows": len(items),
            "avg_remaining_len": safe_mean([
                float(item["remaining_len"]) for item in items
            ]),
            "p90_remaining_len": percentile(
                [float(item["remaining_len"]) for item in items], 0.9
            ),
        }
        for method in ("baseline", "sibling_mean", "tail_blend"):
            values = [
                float(item[f"{method}_abs_error"])
                for item in items
                if item.get(f"{method}_abs_error") is not None
            ]
            summary[f"{method}_mae"] = safe_mean(values)
            summary[f"{method}_p90_abs_error"] = percentile(values, 0.9)
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay BoN decision points into feature rows."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--k-finished", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--prefix-lens", type=int, nargs="+", default=[16, 32, 64, 128, 256]
    )
    parser.add_argument("--min-finished-for-prefix", type=int, default=1)
    parser.add_argument("--default-prior-len", type=int, default=2048)
    parser.add_argument("--logprob-window", type=int, default=32)
    parser.add_argument("--shrink-k", type=float, default=4.0)
    args = parser.parse_args()

    rows = build_rows(args)
    write_csv(Path(args.output_csv), rows)
    summaries = summarize(rows)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump({"summaries": summaries}, f, indent=2)

    print(f"Wrote {len(rows)} feature rows to {args.output_csv}")
    for item in summaries:
        print(
            f"{item['decision_type']:>16} k={item['k_finished']:<2} "
            f"n={item['num_rows']:<6} "
            f"tail_mae={item.get('tail_blend_mae')!s:<18} "
            f"baseline_mae={item.get('baseline_mae')!s:<18}"
        )


if __name__ == "__main__":
    main()
