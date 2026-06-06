#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

PYTHON="${PYTHON:-.venv-gpu/bin/python}"
DATASET="${DATASET:-gsm8k}"
PROMPT_JSONL="${PROMPT_JSONL:-benchmark_outputs/tail_blend_generalization_prompts_80/gsm8k_80.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_outputs/order_test_tail_first}"
FIRST_MODE="${FIRST_MODE:-tail_blend}"
SECOND_MODE="${SECOND_MODE:-irp_eager}"
TTFT_SLO="${TTFT_SLO:-180.0}"
TPOT_SLO="${TPOT_SLO:-0.20}"
TPOT_SLO_AGGREGATION="${TPOT_SLO_AGGREGATION:-p99}"

COMMON_ARGS=(
  --model Qwen/Qwen3.5-2B
  --dtype float16
  --max-model-len 8192
  --gpu-memory-utilization 0.50
  --max-num-seqs 256
  --num-requests 80
  --request-rate 1.0
  --bon-n 8
  --serial-fraction 0.0
  --output-len 4096
  --prompt-jsonl "$PROMPT_JSONL"
  --ttft-slo "$TTFT_SLO"
  --tpot-slo "$TPOT_SLO"
  --tpot-slo-aggregation "$TPOT_SLO_AGGREGATION"
  --parallel-slo-mode logical
  --sampling-seed 0
  --workload-seed 0
  --enforce-eager
  --log-level WARNING
)

if [[ "${IGNORE_EOS:-0}" == "1" ]]; then
  COMMON_ARGS+=(--ignore-eos)
fi

run_mode() {
  local mode="$1"
  local out_dir="$OUTPUT_ROOT/$DATASET/$mode"
  local summary_path="$out_dir/${mode}_summary.json"
  local requests_path="$out_dir/${mode}_requests.csv"
  local extra_args=()

  mkdir -p "$out_dir"

  if [[ "$mode" == "tail_blend" ]]; then
    extra_args=(
      --enable-tail-blend-preemption
      --tail-blend-preemption-mode full
    )
  fi

  echo "== Running $mode -> $out_dir =="
  "$PYTHON" benchmarks/benchmark_taper_plus.py \
    --mode "$mode" \
    "${extra_args[@]}" \
    "${COMMON_ARGS[@]}" \
    --output-dir "$out_dir" \
    --result-json "$summary_path" \
    --raw-csv "$requests_path"
}

run_mode "$FIRST_MODE"
run_mode "$SECOND_MODE"

echo
echo "== Summary =="
"$PYTHON" - <<PY
import csv
import json
from pathlib import Path

root = Path("$OUTPUT_ROOT") / "$DATASET"
modes = ["$FIRST_MODE", "$SECOND_MODE"]
ttft_slo = float("$TTFT_SLO")
tpot_slo = float("$TPOT_SLO")
aggregations = ("max", "p99", "p95", "mean")


def as_bool(value):
    return str(value).lower() == "true"


def met_slo(row, aggregation):
    tpot_key = f"{aggregation}_tpot_s"
    return (
        as_bool(row["finished"])
        and int(row["num_finished_branches"]) == int(row["expected_branches"])
        and float(row["ttft_s"]) <= ttft_slo
        and float(row[tpot_key]) <= tpot_slo
    )

for mode in modes:
    summary_path = root / mode / f"{mode}_summary.json"
    requests_path = root / mode / f"{mode}_requests.csv"
    data = json.loads(summary_path.read_text())
    metrics = data["metrics"]
    rows = list(csv.DictReader(requests_path.open()))
    duration_s = float(metrics["duration_s"])

    print(mode)
    for key in (
        "avg_ttft_s",
        "p99_ttft_s",
        "system_throughput_tokens_s",
        "preemption_count",
        "predictor_preemption_count",
        "duration_s",
    ):
        print(f"  {key}: {metrics[key]}")
    print("  SLO by TPOT aggregation:")
    for aggregation in aggregations:
        tpot_key = f"{aggregation}_tpot_s"
        good = [row for row in rows if met_slo(row, aggregation)]
        tpot_over = sum(float(row[tpot_key]) > tpot_slo for row in rows)
        good_tokens = sum(int(row["output_tokens"]) for row in good)
        goodput = good_tokens / max(duration_s, 1e-9)
        rate = 100.0 * len(good) / len(rows) if rows else 0.0
        print(
            f"    {aggregation}: {len(good)}/{len(rows)} "
            f"({rate:.2f}%), goodput={goodput:.2f} tok/s, "
            f"{tpot_key}>{tpot_slo:g}: {tpot_over}/{len(rows)}"
        )
PY
