# TailBlend Reproducibility Guide

This guide describes how to reproduce the TailBlend vLLM experiments from a fresh checkout of this repository.

TailBlend is implemented as a scheduler-level vLLM modification. It keeps vLLM's normal eager `n=N` Best-of-N execution and changes only the preemption victim selected after KV-cache allocation failure.

## 1. Environment

The experiments require a CUDA-capable GPU supported by vLLM. The preliminary results in the proposal used:

- Model: `Qwen/Qwen3.5-2B`
- dtype: `float16`
- Request rate: `1.0`
- Number of logical requests per dataset: `40`
- Best-of-N values: usually `N in {8, 16}`
- GPU memory utilization: `0.70` unless otherwise specified

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install vLLM from this checkout in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Install benchmark helper dependencies:

```bash
python -m pip install -r requirements/tailblend.txt
```

Notes:

- The repository's `pyproject.toml` defines the vLLM package dependencies. Do not install an unrelated released `vllm` wheel on top of this checkout.
- Hugging Face model and dataset downloads require network access.
- Gated datasets, if used, require `HF_TOKEN`. The default public prompt builder uses UltraChat as the chat workload, so `HF_TOKEN` is not required for the main four-dataset setup.

## 2. Build Prompt Sets

Generate the JSONL prompt files used by the generalization sweep:

```bash
python benchmarks/build_tail_blend_prompt_sets.py \
  --model Qwen/Qwen3.5-2B \
  --limit 40 \
  --output-dir benchmark_outputs/tail_blend_generalization_prompts \
  --max-short-prompt-tokens 896 \
  --max-long-prompt-tokens 4096 \
  --longbench-subset qasper
```

This creates:

- `benchmark_outputs/tail_blend_generalization_prompts/gsm8k_40.jsonl`
- `benchmark_outputs/tail_blend_generalization_prompts/mbpp_40.jsonl`
- `benchmark_outputs/tail_blend_generalization_prompts/longbench_qasper_40.jsonl`
- `benchmark_outputs/tail_blend_generalization_prompts/chat_ultrachat_40.jsonl`

Each row contains a `prompt` field consumed by `benchmarks/benchmark_taper_plus.py`.

## 3. Single Comparison Run

Run Default vLLM-BoN versus TailBlend on one prompt set:

```bash
python benchmarks/benchmark_taper_plus.py \
  --mode both_tail_blend \
  --enable-tail-blend-preemption \
  --tail-blend-preemption-mode full \
  --model Qwen/Qwen3.5-2B \
  --dtype float16 \
  --max-model-len 5120 \
  --gpu-memory-utilization 0.70 \
  --num-requests 40 \
  --request-rate 1.0 \
  --bon-n 16 \
  --serial-fraction 0.0 \
  --output-len 4096 \
  --prompt-jsonl benchmark_outputs/tail_blend_generalization_prompts/gsm8k_40.jsonl \
  --ttft-slo 180.0 \
  --tpot-slo 0.20 \
  --parallel-slo-mode logical \
  --sampling-seed 0 \
  --workload-seed 0 \
  --output-dir benchmark_outputs/tail_blend_repro_gsm8k_bon16_seed0 \
  --enforce-eager \
  --log-level WARNING
```

Outputs:

- `comparison.json`: both Default vLLM-BoN and TailBlend summary metrics
- `irp_eager_summary.json`: Default vLLM-BoN metrics
- `tail_blend_summary.json`: TailBlend metrics
- `*_requests.csv`: per-request traces

## 4. Four-Dataset Sweep

Run the main four-dataset sweep:

```bash
python benchmarks/run_tail_blend_generalization_sweep.py \
  --model Qwen/Qwen3.5-2B \
  --dtype float16 \
  --bon-values 8,16 \
  --seeds 0,1,2 \
  --num-requests 40 \
  --request-rate 1.0 \
  --gpu-memory-utilization 0.70 \
  --output-len 4096 \
  --max-model-len 5120 \
  --long-output-len 2048 \
  --long-max-model-len 8192 \
  --ttft-slo 180.0 \
  --tpot-slo 0.20 \
  --parallel-slo-mode logical \
  --output-root benchmark_outputs/tail_blend_repro_sweep \
  --enforce-eager \
  --log-level WARNING
```

The sweep uses the default prompt paths created in Section 2. It automatically uses the LongBench/Qasper budget (`max_model_len=8192`, `output_len=2048`) for datasets whose name starts with `longbench`; other datasets use `max_model_len=5120`, `output_len=4096`.

To resume an interrupted sweep:

```bash
python benchmarks/run_tail_blend_generalization_sweep.py \
  --output-root benchmark_outputs/tail_blend_repro_sweep \
  --resume
```

## 5. Key Policies and Modes

Default vLLM-BoN baseline:

- Benchmark mode: `irp_eager`
- vLLM runs normal `n=N` sampling.
- On KV allocation failure, vLLM uses its default order/priority-based victim policy.

TailBlend:

- Benchmark mode: `tail_blend`, or comparison mode `both_tail_blend`
- Enable preemption scoring with `--enable-tail-blend-preemption`
- Main scoring mode: `--tail-blend-preemption-mode full`

Available TailBlend preemption modes:

- `remaining`
- `remaining_recompute`
- `remaining_parent`
- `full`
- `full_v2`
- `full_slo`
- `prefill_aware`
- `prefill_aware_gated`
- `adaptive`

The proposal treats `full` TailBlend as the main policy. `prefill_aware` and `adaptive` are exploratory variants.

## 6. Result Metrics

Important fields in summary JSON and sweep CSV:

- `user_goodput_tokens_s`
- `slo_attainment_rate_pct`
- `preemption_count`
- `predictor_preemption_count`
- `preempted_computed_tokens`
- `preempted_output_tokens`
- `adaptive_default_count`
- `adaptive_full_count`
- `adaptive_prefill_aware_count`
- `adaptive_full_prefix_guard_count`

For the proposal's main tables, compare `irp_eager` against `tail_blend` on user-level goodput and SLO attainment.

## 7. Hardware and Reproducibility Notes

- Results are sensitive to GPU type, available memory, CUDA stack, vLLM build configuration, and background GPU load.
- Use the same GPU memory utilization, request rate, seeds, model, and prompt files when comparing policies.
- Do not compare runs produced with different `max_model_len` or `output_len` unless the difference is intentional.
- LongBench/Qasper uses a different context/output budget to preserve long-context prompts while keeping total KV pressure manageable.
- Keep raw `comparison.json`, `*_summary.json`, and `*_requests.csv` files for auditability.

## 8. Minimal Smoke Test

Before launching a long sweep, run a small synthetic test:

```bash
python benchmarks/benchmark_taper_plus.py \
  --mode both_tail_blend \
  --enable-tail-blend-preemption \
  --tail-blend-preemption-mode full \
  --model Qwen/Qwen3.5-2B \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.70 \
  --num-requests 4 \
  --request-rate inf \
  --bon-n 4 \
  --serial-fraction 0.0 \
  --synthetic-prompt-words 64 \
  --output-len 256 \
  --output-dir benchmark_outputs/tail_blend_smoke \
  --enforce-eager \
  --log-level WARNING
```

The smoke test is not meant to reproduce paper numbers. It only checks that the benchmark, model loading, and TailBlend preemption path run end to end.
