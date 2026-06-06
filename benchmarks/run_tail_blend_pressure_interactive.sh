#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 && -z "${OUTPUT_ROOT:-}" ]]; then
  echo "usage: $0 <output_root>" >&2
  echo "example: $0 benchmark_outputs/interactive_pressure_eos_gsm8k_bon16" >&2
  exit 2
fi

export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

PYTHON="${PYTHON:-.venv-gpu/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$1}"
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
DTYPE="${DTYPE:-float16}"
PROMPT_DIR="${PROMPT_DIR:-benchmark_outputs/tail_blend_generalization_prompts_80}"

# Keep GPU util fixed; create pressure by admitting more concurrent branches.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
NUM_REQUESTS="${NUM_REQUESTS:-80}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
OUTPUT_LEN="${OUTPUT_LEN:-4096}"

TTFT_SLO="${TTFT_SLO:-180.0}"
TPOT_SLO="${TPOT_SLO:-0.20}"
TPOT_SLO_AGGREGATION="${TPOT_SLO_AGGREGATION:-p99}"
TAIL_BLEND_ADMISSION_MODE="${TAIL_BLEND_ADMISSION_MODE:-off}"
IGNORE_EOS="${IGNORE_EOS:-0}"
FIRST_MODE="${FIRST_MODE:-tail_blend}"
SECOND_MODE="${SECOND_MODE:-irp_eager}"

# Interactive default is one high-pressure smoke case. Override with:
#   DATASETS="gsm8k mbpp longbench_qasper chat_ultrachat" BON_VALUES="8 16"
read -r -a DATASETS <<<"${DATASETS:-gsm8k}"
read -r -a BON_VALUES <<<"${BON_VALUES:-16}"

echo "== host =="
hostname
echo "== gpu =="
nvidia-smi
echo "== config =="
cat <<EOF
OUTPUT_ROOT=$OUTPUT_ROOT
DATASETS=${DATASETS[*]}
BON_VALUES=${BON_VALUES[*]}
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
MAX_MODEL_LEN=$MAX_MODEL_LEN
MAX_NUM_SEQS=$MAX_NUM_SEQS
NUM_REQUESTS=$NUM_REQUESTS
REQUEST_RATE=$REQUEST_RATE
OUTPUT_LEN=$OUTPUT_LEN
IGNORE_EOS=$IGNORE_EOS
TPOT_SLO_AGGREGATION=$TPOT_SLO_AGGREGATION
TAIL_BLEND_ADMISSION_MODE=$TAIL_BLEND_ADMISSION_MODE
FIRST_MODE=$FIRST_MODE
SECOND_MODE=$SECOND_MODE
EOF

"$PYTHON" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("cuda version", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
PY

COMMON_ARGS=(
  --model "$MODEL"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --num-requests "$NUM_REQUESTS"
  --request-rate "$REQUEST_RATE"
  --serial-fraction 0.0
  --output-len "$OUTPUT_LEN"
  --ttft-slo "$TTFT_SLO"
  --tpot-slo "$TPOT_SLO"
  --tpot-slo-aggregation "$TPOT_SLO_AGGREGATION"
  --parallel-slo-mode logical
  --sampling-seed 0
  --workload-seed 0
  --enforce-eager
  --log-level WARNING
)

if [[ "$IGNORE_EOS" == "1" ]]; then
  COMMON_ARGS+=(--ignore-eos)
fi

run_mode() {
  local dataset="$1"
  local bon="$2"
  local mode="$3"
  local prompt_jsonl="$PROMPT_DIR/${dataset}_${NUM_REQUESTS}.jsonl"
  local out_dir="$OUTPUT_ROOT/${dataset}_bon${bon}/${mode}"
  local extra_args=()

  if [[ ! -f "$prompt_jsonl" ]]; then
    echo "missing prompt file: $prompt_jsonl" >&2
    exit 1
  fi

  mkdir -p "$out_dir"

  if [[ "$mode" == "tail_blend" ]]; then
    extra_args=(
      --enable-tail-blend-preemption
      --tail-blend-preemption-mode full
      --tail-blend-admission-mode "$TAIL_BLEND_ADMISSION_MODE"
    )
  fi

  {
    echo "dataset=$dataset"
    echo "bon=$bon"
    echo "mode=$mode"
    echo "host=$(hostname)"
    echo "date=$(date -Is)"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
    echo "MODEL=$MODEL"
    echo "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
    echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
    echo "MAX_NUM_SEQS=$MAX_NUM_SEQS"
    echo "NUM_REQUESTS=$NUM_REQUESTS"
    echo "REQUEST_RATE=$REQUEST_RATE"
    echo "OUTPUT_LEN=$OUTPUT_LEN"
    echo "IGNORE_EOS=$IGNORE_EOS"
    echo "TPOT_SLO_AGGREGATION=$TPOT_SLO_AGGREGATION"
    nvidia-smi
  } > "$out_dir/env.log" 2>&1

  echo "== Running dataset=$dataset bon=$bon mode=$mode -> $out_dir =="
  "$PYTHON" benchmarks/benchmark_taper_plus.py \
    --mode "$mode" \
    --bon-n "$bon" \
    --prompt-jsonl "$prompt_jsonl" \
    "${extra_args[@]}" \
    "${COMMON_ARGS[@]}" \
    --output-dir "$out_dir" \
    --result-json "$out_dir/${mode}_summary.json" \
    --raw-csv "$out_dir/${mode}_requests.csv" \
    2>&1 | tee "$out_dir/run.log"
}

for dataset in "${DATASETS[@]}"; do
  for bon in "${BON_VALUES[@]}"; do
    run_mode "$dataset" "$bon" "$FIRST_MODE"
    run_mode "$dataset" "$bon" "$SECOND_MODE"
  done
done

SUMMARY_CSV="$OUTPUT_ROOT/sweep_summary_by_slo.csv"
DATASET_LIST="${DATASETS[*]}"
BON_LIST="${BON_VALUES[*]}"
"$PYTHON" - <<PY
import csv
import json
from pathlib import Path

root = Path("$OUTPUT_ROOT")
datasets = "$DATASET_LIST".split()
bon_values = [int(value) for value in "$BON_LIST".split()]
modes = ["$FIRST_MODE", "$SECOND_MODE"]
ttft_slo = float("$TTFT_SLO")
tpot_slo = float("$TPOT_SLO")
aggregations = ("max", "p99", "p95", "mean")
summary_csv = Path("$SUMMARY_CSV")


def as_bool(value):
    return str(value).lower() == "true"


def met_slo(row, aggregation):
    return (
        as_bool(row["finished"])
        and int(row["num_finished_branches"]) == int(row["expected_branches"])
        and float(row["ttft_s"]) <= ttft_slo
        and float(row[f"{aggregation}_tpot_s"]) <= tpot_slo
    )


fieldnames = [
    "dataset",
    "bon",
    "mode",
    "aggregation",
    "met_slo_requests",
    "total_requests",
    "slo_attainment_pct",
    "goodput_tokens_s",
    "tpot_violations",
    "avg_ttft_s",
    "p99_ttft_s",
    "system_throughput_tokens_s",
    "preemption_count",
    "predictor_preemption_count",
    "duration_s",
    "output_dir",
]
summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for dataset in datasets:
        for bon in bon_values:
            for mode in modes:
                out_dir = root / f"{dataset}_bon{bon}" / mode
                metrics = json.loads((out_dir / f"{mode}_summary.json").read_text())["metrics"]
                rows = list(csv.DictReader((out_dir / f"{mode}_requests.csv").open()))
                duration_s = float(metrics["duration_s"])
                for aggregation in aggregations:
                    tpot_key = f"{aggregation}_tpot_s"
                    good = [row for row in rows if met_slo(row, aggregation)]
                    writer.writerow({
                        "dataset": dataset,
                        "bon": bon,
                        "mode": mode,
                        "aggregation": aggregation,
                        "met_slo_requests": len(good),
                        "total_requests": len(rows),
                        "slo_attainment_pct": 100.0 * len(good) / len(rows) if rows else 0.0,
                        "goodput_tokens_s": sum(int(row["output_tokens"]) for row in good) / max(duration_s, 1e-9),
                        "tpot_violations": sum(float(row[tpot_key]) > tpot_slo for row in rows),
                        "avg_ttft_s": metrics["avg_ttft_s"],
                        "p99_ttft_s": metrics["p99_ttft_s"],
                        "system_throughput_tokens_s": metrics["system_throughput_tokens_s"],
                        "preemption_count": metrics["preemption_count"],
                        "predictor_preemption_count": metrics["predictor_preemption_count"],
                        "duration_s": duration_s,
                        "output_dir": str(out_dir),
                    })
print(summary_csv)
PY

echo "== summary =="
cat "$SUMMARY_CSV"
