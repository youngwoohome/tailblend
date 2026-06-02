# TAPER+ Phase 1-4 Implementation Notes

This document summarizes the local TAPER+ changes made against the current vLLM
v1 scheduler architecture. It is intended to help reviewers compare the modified
code against stock vLLM and understand why each new hook exists.

## Background

Stock vLLM v1 handles `SamplingParams(n > 1)` by fanning one user request out
into multiple child `Request` objects. Each child is then scheduled independently
by `vllm/v1/core/sched/scheduler.py`.

For example, a user-visible request with `n=32` becomes 32 internal branches.
The child request IDs are created in `vllm/v1/engine/parallel_sampling.py` as:

```python
child_req_id = f"{index}_{self.request_id}"
```

The scheduler normally treats these child branches like unrelated requests. The
TAPER+ changes add enough metadata and scheduling logic to regulate branch width
inside each BoN parent while keeping stock behavior as the default.

## Phase 1: Branch Lifetime Predictor

### File Changed

- `vllm/v1/request.py`

The v1 scheduler operates on `Request` objects, so the lifetime predictor state
is stored directly on `Request`.

### New State

`Request.__init__` now initializes:

```python
self.prob_short = 0.0
```

This is the TAPER+ estimate that a branch has a short remaining lifetime. It is
monotonic: once the branch looks short-lived, later updates do not lower the
score.

### New Methods

`Request.update_prob_short(...)` was added:

```python
def update_prob_short(
    self,
    *,
    eos_logprob: float | None = None,
    last_token_id: int | None = None,
) -> None:
```

The update currently uses two signals:

- If the latest token is a stop/EOS token, `prob_short` becomes `1.0`.
- Otherwise, it falls back to output-length progress:

```python
length_progress = min(1.0, self.num_output_tokens / max_tokens)
```

If a future sampler path passes an EOS log-probability, the method can also use
`exp(eos_logprob)` as a direct short-lifetime signal.

`Request.is_short_lived()` was added:

```python
def is_short_lived(self) -> bool:
    return self.prob_short > 0.5
```

### Decode Update Hook

`Request.append_output_token_ids(...)` now calls:

```python
self.update_prob_short(last_token_id=last_token_id)
```

This means the predictor is updated whenever scheduler output processing appends
new generated tokens to a branch.

### Parent Request Metadata

`Request` now also carries:

```python
self.external_req_id = external_req_id
```

The field is copied from `EngineCoreRequest.external_req_id` in
`Request.from_engine_core_request(...)`.

This is necessary because BoN child branches have distinct internal request IDs,
but all children share the same user-visible external request ID. Phase 3 uses
this field to group branches belonging to the same logical BoN request.

## Phase 2: Memory Safety Gate

### File Changed

- `vllm/v1/core/kv_cache_manager.py`

The v1 scheduler owns a `KVCacheManager`, which in turn exposes the active
`BlockPool`. The memory helpers are added to `KVCacheManager` so scheduling
policy code can query KV-cache pressure through a narrow interface.

### Free-Block Helper

`KVCacheManager.get_remaining_free_blocks()` was added:

```python
def get_remaining_free_blocks(self) -> int:
    return self.block_pool.get_num_free_blocks()
```

This exposes the current GPU KV block budget to the scheduler without requiring
the scheduler to reach into `BlockPool` directly.

### Future-Footprint Estimator

`KVCacheManager.estimate_future_footprint(request: Request)` was added.

It estimates how many additional KV blocks a branch may require before it
finishes:

- For long-lived branches, it uses a conservative upper bound:

```python
target_num_tokens = request.num_prompt_tokens + request.max_tokens
```

- For short-lived branches, it uses a tighter remaining-token estimate based on
`prob_short`:

```python
predicted_remaining_tokens = max(
    1,
    ceil(remaining_output_tokens * (1.0 - request.prob_short)),
)
target_num_tokens = request.num_tokens + predicted_remaining_tokens
```

The method then asks the existing KV coordinator for the actual block count:

```python
self.coordinator.get_num_blocks_to_allocate(..., apply_admission_cap=True)
```

Using the coordinator keeps the estimate aligned with vLLM's current block-size,
hybrid KV-cache, sliding-window, and admission-cap logic.

## Phase 3: Dual-Budgeted Planner

### Files Changed

- `vllm/v1/core/sched/scheduler.py`
- `vllm/envs.py`

The scheduler change is gated behind environment variables so stock vLLM behavior
remains the default.

### New Runtime Flags

`vllm/envs.py` adds:

```python
VLLM_ENABLE_TAPER_PLUS
VLLM_TAPER_PLUS_MEMORY_THRESHOLD_BLOCKS
VLLM_TAPER_PLUS_STEP_TARGET_MS
VLLM_TAPER_PLUS_LATENCY_PER_SEQ_MS
VLLM_TAPER_PLUS_TTFT_SLO_MS
VLLM_TAPER_PLUS_TPOT_SLO_MS
VLLM_TAPER_PLUS_RHO
VLLM_TAPER_PLUS_UTILITY
```

Default behavior:

- `VLLM_ENABLE_TAPER_PLUS=0`: scheduler behaves like stock vLLM.
- `VLLM_TAPER_PLUS_MEMORY_THRESHOLD_BLOCKS=-1`: scheduler uses 5% of GPU KV
  blocks as the memory-pressure threshold.
- `VLLM_TAPER_PLUS_STEP_TARGET_MS=50.0`
- `VLLM_TAPER_PLUS_LATENCY_PER_SEQ_MS=2.0`
- `VLLM_TAPER_PLUS_TTFT_SLO_MS=2000.0`
- `VLLM_TAPER_PLUS_TPOT_SLO_MS=50.0`
- `VLLM_TAPER_PLUS_RHO=1.0`
- `VLLM_TAPER_PLUS_UTILITY=linear`

The time predictor is intentionally simple for Phase 3:

```text
T(S) = active_decode_branches * VLLM_TAPER_PLUS_LATENCY_PER_SEQ_MS
```

### Scheduler State

`Scheduler.__init__` now reads the TAPER/TAPER+ env flags into instance fields:

```python
self.enable_taper_plus = envs.VLLM_ENABLE_TAPER_PLUS
self.taper_plus_policy = envs.VLLM_TAPER_PLUS_POLICY
self.taper_plus_memory_threshold_blocks = (
    envs.VLLM_TAPER_PLUS_MEMORY_THRESHOLD_BLOCKS
)
self.taper_plus_step_target_ms = envs.VLLM_TAPER_PLUS_STEP_TARGET_MS
self.taper_plus_latency_per_seq_ms = (
    envs.VLLM_TAPER_PLUS_LATENCY_PER_SEQ_MS
)
self.taper_plus_ttft_slo_s = envs.VLLM_TAPER_PLUS_TTFT_SLO_MS / 1000.0
self.taper_plus_tpot_slo_s = envs.VLLM_TAPER_PLUS_TPOT_SLO_MS / 1000.0
self.taper_plus_rho = envs.VLLM_TAPER_PLUS_RHO
self.taper_plus_utility = envs.VLLM_TAPER_PLUS_UTILITY
```

`VLLM_TAPER_PLUS_POLICY` separates pure TAPER from the extra TAPER+ gates:

| Policy | Enabled behavior |
|---|---|
| `taper` | Baseline protection plus time-slack width regulation only. Deferred branches keep FIFO ordering. Memory gate, future footprint, and lifetime ordering are disabled. |
| `taper_plus` | `taper` plus memory safety gate, future footprint estimate, and lifetime-aware ordering by `prob_short`. |

This split lets us first reproduce the TAPER-style IRP controller in isolation,
then add TAPER+ features one at a time as an ablation.

### Branch Grouping

`Scheduler._taper_plus_group_id(request)` returns:

```python
return request.external_req_id or request.request_id
```

For BoN requests, all child branches share `external_req_id`, so they are
regulated together. Non-BoN requests normally have a unique group and are not
width-regulated.

### Decode Candidate Detection

`Scheduler._is_taper_plus_decode_candidate(request)` restricts TAPER+ to decode
branches that have already produced output and still have work pending:

```python
request.sampling_params is not None
and request.num_output_tokens + request.num_output_placeholders > 0
and request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
    > 0
```

This avoids regulating initial prefill chunks. Prefill admission remains handled
by the existing scheduler path.

### Admission Planner

`Scheduler._get_taper_plus_admitted_running_req_ids(token_budget)` computes the
set of running request IDs allowed to receive compute this step.

The planner implements pure TAPER first, with TAPER+ as an extension:

1. **Baseline protection**

   For every active BoN parent group, the first eligible branch in running-order
   is admitted. This guarantees at least 1-token progress for each active group.

2. **Slack-budgeted time budget**

   The previous prototype used a fixed absolute step target. The current
   implementation follows the TAPER paper's Algorithm 1 more closely:

   ```python
   T0 = T(baseline)
   min_slack = min(request.deadline - now for request in active_requests)
   budget = T0 + rho * max(0, min_slack - T0)
   ```

   vLLM represents one BoN request as multiple child `Request` objects, but
   Algorithm 1 reasons over the logical parent request. The implementation
   therefore computes one deadline per `external_req_id` group. A group with no
   generated branch token uses the TTFT SLO as its next-token deadline:

   ```python
   deadline = min(child.arrival_time for child in group) + TTFT_SLO
   ```

   After the first generated token, the TPOT SLO is measured from the most
   recent branch progress in that logical request:

   ```python
   deadline = max(child.taper_last_token_time for child in group) + TPOT_SLO
   ```

   This is the key paper behavior: if the batch has slack, TAPER can widen
   toward IRP-EAGER; if the most urgent co-batched request is tight, the budget
   collapses toward the protected baseline.

3. **Greedy utility-per-cost planner**

   Instead of scanning deferred branches once, the scheduler repeatedly
   considers one additional branch from each BoN parent and commits the feasible
   candidate with the largest marginal utility per predicted marginal latency:

   ```python
   score = du / (EPS + max(0, T(widened) - T(step)))
   ```

   `VLLM_TAPER_PLUS_UTILITY=linear` matches the throughput-oriented utility
   curve from the paper. `concave` is available as an ablation for fairness-style
   experiments.

4. **Lifetime-aware ordering (`taper_plus` only)**

   Remaining branches are deferred and sorted by:

   ```python
   -request.prob_short, request.arrival_time, request.request_id
   ```

   Higher `prob_short` branches are tried first, encouraging short branches to
   finish and release KV cache.

5. **Memory gate (`taper_plus` only)**

   The scheduler reads current free GPU KV blocks via:

   ```python
   self.kv_cache_manager.get_remaining_free_blocks()
   ```

   It estimates future branch footprint via:

   ```python
   self.kv_cache_manager.estimate_future_footprint(request)
   ```

   If free blocks are under the threshold, or if the estimated future footprint
   exceeds the virtual free-block budget, a branch is admitted only when:

   ```python
   request.is_short_lived()
   ```

6. **Execution by omission**

   Deferred branches are not preempted and their KV cache is not freed. They are
   simply omitted from this step's `scheduled_running_reqs`, so they do not enter
   the current forward pass.

### Integration Point in `schedule()`

Right after `self.kv_cache_manager.new_step_starts()`, the scheduler computes:

```python
taper_plus_admitted_running_req_ids = (
    self._get_taper_plus_admitted_running_req_ids(token_budget)
)
```

Then, while iterating `self.running`, it skips non-admitted branches:

```python
if (
    taper_plus_admitted_running_req_ids is not None
    and request.request_id not in taper_plus_admitted_running_req_ids
):
    req_index += 1
    continue
```

This preserves the existing allocation, preemption, LoRA, encoder-cache,
spec-decode, and scheduler-output code paths for admitted branches.

## Phase 4: Benchmark Harness

### File Added

- `benchmarks/benchmark_taper_plus.py`

The benchmark drives `LLMEngine` directly instead of going through the OpenAI
HTTP server. This keeps the measurement focused on scheduler behavior and avoids
client/server transport noise.

### Execution Modes

The script supports separate TAPER and TAPER+ modes:

```bash
--mode baseline
--mode irp_eager
--mode taper
--mode taper_plus
--mode both
--mode both_plus
--mode all
```

`baseline` sets:

```python
VLLM_ENABLE_TAPER_PLUS=0
```

`taper` sets:

```python
VLLM_ENABLE_TAPER_PLUS=1
VLLM_TAPER_PLUS_POLICY=taper
```

`taper_plus` sets:

```python
VLLM_ENABLE_TAPER_PLUS=1
VLLM_TAPER_PLUS_POLICY=taper_plus
```

`both` runs IRP-EAGER vs pure TAPER. `both_plus` runs IRP-EAGER vs TAPER+.
`all` runs IRP-OFF, IRP-EAGER, TAPER, and TAPER+ in separate subprocesses. This
is deliberate: the scheduler reads environment variables during engine
initialization, so separate processes avoid accidental cross-mode state.

The script also forces:

```python
VLLM_ENABLE_V1_MULTIPROCESSING=0
```

This keeps the `EngineCore` in-process, which lets the benchmark read
`scheduler.num_cumulative_preemptions` directly.

### Workload

The benchmark can use either ShareGPT prompts or a synthetic trace:

```bash
--sharegpt-path /path/to/sharegpt.json
```

If no ShareGPT file is provided, it creates synthetic prompts with
`--synthetic-prompt-words`.

Request arrivals follow a Poisson process:

```bash
--request-rate 8
```

Using `--request-rate inf` submits the whole workload immediately.

BoN fanout is controlled by:

```bash
--bon-n 32
--temperature 0.7
```

The script constructs:

```python
SamplingParams(
    n=args.bon_n,
    temperature=args.temperature,
    max_tokens=args.output_len,
    output_kind=RequestOutputKind.DELTA,
)
```

`RequestOutputKind.DELTA` is important because the benchmark timestamps streamed
token deltas and computes inter-token latency from client-visible output events.

### Metrics

The benchmark records request-level traces and computes:

- Average TTFT.
- P99 TTFT.
- System throughput in raw generated tokens/s across all BoN branches.
- User throughput in completed logical requests/s.
- User-level goodput in generated tokens/s from requests that met both SLOs.
- SLO attainment rate.
- Scheduler preemption count.

For each request, the raw trace includes:

- scheduled arrival time
- actual submit time
- first token time
- finish time
- prompt length
- total output tokens
- number of finished branches
- mean TPOT
- max TPOT
- SLO pass/fail

### Output Files

For `--mode both`, output is written under `--output-dir`:

```text
baseline_summary.json
baseline_requests.csv
taper_summary.json
taper_requests.csv
comparison.json
```

The CSV files are request-level and are intended for CDF plotting.

### Example Command

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
python benchmarks/benchmark_taper_plus.py \
  --mode both \
  --model Qwen/Qwen3.5-2B \
  --dtype float16 \
  --enforce-eager \
  --max-model-len 512 \
  --gpu-memory-utilization 0.60 \
  --num-requests 8 \
  --request-rate inf \
  --bon-n 32 \
  --output-len 32 \
  --synthetic-prompt-words 64 \
  --ttft-slo 5.0 \
  --tpot-slo 0.5 \
  --taper-step-target-ms 8 \
  --taper-latency-per-seq-ms 2 \
  --output-dir benchmark_outputs/taper_plus_wsl_n32_r8
```

`VLLM_USE_FLASHINFER_SAMPLER=0` was needed in the local WSL environment because
FlashInfer sampler JIT required `nvcc`, while the machine had CUDA runtime
available but not the CUDA toolkit under `/usr/local/cuda`.

### Local Validation

Validation was run on WSL Ubuntu 24.04 with:

```text
Python: 3.12.3
torch: 2.11.0+cu130
CUDA available: True
GPU: NVIDIA GeForce RTX 4070 Ti SUPER
vLLM: 0.21.1.dev0+gad7125a43.d20260526
```

The WSL environment was prepared with:

```bash
python3 -m venv ~/.venvs/vllm-taper
source ~/.venvs/vllm-taper/bin/activate
cd "/mnt/c/Users/young/OneDrive/바탕 화면/vllm-src"
python -m pip install -U pip setuptools wheel
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=nightly \
python -m pip install -e .
```

Native Windows validation was blocked because `uvloop` does not support Windows.
Installing `torch==2.11.0` and `requirements/common.txt` on Windows succeeded,
but `python benchmarks/benchmark_taper_plus.py --help` stopped at:

```text
RuntimeError: uvloop does not support Windows at the moment
```

The WSL benchmark succeeded. For the `n=32`, 8-request smoke-pressure run:

```text
Metric                         Baseline              TAPER+
Avg TTFT                       0.3036s               0.9093s
P99 TTFT                       0.4858s               2.7146s
System throughput              10666.32 tok/s        1389.24 tok/s
User throughput                10.4354 req/s         1.3567 req/s
User-level goodput             10666.32 tok/s        0.00 tok/s
SLO attainment                 100.00%               0.00%
Preemption count               0                     0
```

This run intentionally used a tight TAPER+ time budget:

```text
VLLM_TAPER_PLUS_STEP_TARGET_MS = 8
VLLM_TAPER_PLUS_LATENCY_PER_SEQ_MS = 2
```

Because the validation model was small relative to the available GPU memory and
the GPU had ample KV capacity, the run validates benchmark execution and width
regulation mechanics rather than demonstrating memory-pressure preemption
avoidance.

`Qwen/Qwen3.5-2B` was also validated with a smaller loading smoke test:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
python benchmarks/benchmark_taper_plus.py \
  --mode taper \
  --model Qwen/Qwen3.5-2B \
  --dtype float16 \
  --enforce-eager \
  --max-model-len 512 \
  --gpu-memory-utilization 0.60 \
  --num-requests 1 \
  --request-rate inf \
  --bon-n 4 \
  --output-len 4 \
  --synthetic-prompt-words 16 \
  --ttft-slo 10.0 \
  --tpot-slo 1.0 \
  --result-json benchmark_outputs/taper_plus_qwen35_2b_smoke/taper_summary.json \
  --raw-csv benchmark_outputs/taper_plus_qwen35_2b_smoke/taper_requests.csv
```

This run resolved `Qwen3_5ForConditionalGeneration`, loaded the model, created
KV cache, and completed generation:

```text
Avg TTFT                 1.6747s
System throughput        9.3228 tok/s
User throughput          0.5827 req/s
SLO attainment           100.00%
Preemption count         0
```

## Research-Grade Benchmarking Notes

The benchmark harness now includes two layers:

- `benchmarks/benchmark_taper_plus.py`: one controlled run comparing
  `irp_off`, `irp_eager`, and `taper`.
- `benchmarks/run_taper_plus_sweep.py`: resumable grid runner over BoN width,
  output length, request count, and request arrival rate. It records both
  successful metrics and failed/OOM/timeout cases.

The default sweep dimensions are intended to match the TAPER-style BoN serving
study:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
python benchmarks/run_taper_plus_sweep.py \
  --model Qwen/Qwen3.5-2B \
  --dtype float16 \
  --enforce-eager \
  --gpu-memory-utilization 0.60 \
  --bon-values 8,16,32,64 \
  --output-lens 256,1024,4096 \
  --num-requests-values 16,32,64 \
  --request-rates 2,4,8 \
  --serial-fraction 0.5 \
  --synthetic-prompt-words 256 \
  --ttft-slo 10.0 \
  --tpot-slo 1.0 \
  --taper-step-target-ms 50 \
  --taper-latency-per-seq-ms 0.5 \
  --ignore-eos \
  --output-root benchmark_outputs/taper_plus_qwen35_2b_full_sweep \
  --timeout-s 7200 \
  --resume
```

For local validation on the RTX 4070 Ti SUPER 16GB setup, a shorter BoN sweep
was executed with `output_len=128`, `num_requests=16`, `request_rate=inf`, and
`serial_fraction=0.5`:

```text
Output directory:
benchmark_outputs/taper_plus_qwen35_2b_bon_sweep_128
```

| BoN | Mode | Avg TTFT | P99 TTFT | Sys tok/s | Req/s | Goodput tok/s | SLO | Serial SLO | Parallel SLO |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | irp_off | 0.327s | 0.461s | 439.5 | 0.771 | 44.8 | 50.0% | 100.0% | 0.0% |
| 8 | irp_eager | 0.363s | 0.546s | 3016.9 | 5.292 | 3016.9 | 100.0% | 100.0% | 100.0% |
| 8 | taper | 0.316s | 0.466s | 3422.3 | 6.003 | 3422.3 | 100.0% | 100.0% | 100.0% |
| 16 | irp_off | 0.612s | 2.824s | 416.3 | 0.383 | 24.5 | 50.0% | 100.0% | 0.0% |
| 16 | irp_eager | 0.595s | 2.937s | 3180.3 | 2.939 | 3180.3 | 100.0% | 100.0% | 100.0% |
| 16 | taper | 0.616s | 3.054s | 3035.5 | 2.805 | 2317.4 | 87.5% | 100.0% | 75.0% |
| 32 | irp_off | 14.191s | 48.827s | 279.5 | 0.132 | 5.3 | 31.2% | 62.5% | 0.0% |
| 32 | irp_eager | 2.859s | 6.786s | 3506.2 | 1.670 | 3506.2 | 100.0% | 100.0% | 100.0% |
| 32 | taper | 2.861s | 6.862s | 3613.8 | 1.716 | 2735.2 | 87.5% | 100.0% | 75.0% |
| 64 | irp_off | 97.737s | 215.714s | 198.3 | 0.048 | 0.1 | 6.2% | 12.5% | 0.0% |
| 64 | irp_eager | 6.732s | 13.504s | 4121.9 | 0.992 | 3082.0 | 68.8% | 62.5% | 75.0% |
| 64 | taper | 6.607s | 12.721s | 3823.5 | 0.920 | 979.2 | 43.8% | 62.5% | 25.0% |

This table is useful, but it is not yet a professor-ready claim that TAPER+
dominates IRP-EAGER. It shows that IRP-OFF collapses as BoN width grows, and it
also exposes that the current placeholder time model can under-admit branches.
With `taper_latency_per_seq_ms=0.5`, TAPER+ becomes too conservative for
BoN 16/32/64 on this local GPU.

A calibration check was then run at BoN 64 with
`taper_latency_per_seq_ms=0.1`, which allows TAPER+ to behave like EAGER when
there is no clear memory-pressure signal:

```text
Output directory:
benchmark_outputs/taper_plus_qwen35_2b_bon64_calibrated_01

IRP-EAGER: Avg TTFT 6.7699s, P99 TTFT 13.9229s, Goodput 2886.49 tok/s, SLO 68.75%
TAPER+:    Avg TTFT 6.8659s, P99 TTFT 14.1946s, Goodput 2909.56 tok/s, SLO 68.75%
```

This supports the expected low-pressure behavior: after calibration, TAPER+
tracks IRP-EAGER instead of artificially throttling useful parallelism.

A long-output feasibility run was also completed with `output_len=4096`,
`bon_n=8`, `num_requests=2`, and `ignore_eos=True`:

```text
Output directory:
benchmark_outputs/taper_plus_qwen35_2b_len4096_feasibility

IRP-OFF:   duration 578.35s, Goodput 7.08 tok/s, SLO 50.00%
IRP-EAGER: duration 70.35s,  Goodput 523.98 tok/s, SLO 100.00%
TAPER+:    duration 73.98s,  Goodput 498.29 tok/s, SLO 100.00%
```

This validates 4096-token generation in the harness. It did not trigger
preemption on the local 16GB GPU under this small request count, so the full
paper-style memory-pressure claim still requires the larger sweep or a larger
GPU-backed experiment.

## Online Latency Calibration

The initial Phase 3 implementation used a fixed linear proxy:

```text
predicted_step_ms = active_decode_width * VLLM_TAPER_PLUS_LATENCY_PER_SEQ_MS
```

That was useful for validating the scheduling hook, but it was not strong
enough for research claims. The local BoN sweep showed that a manual constant
can under-admit useful branches: with `taper_latency_per_seq_ms=0.5`, TAPER+
became too conservative for BoN 16/32/64 even when the GPU still had enough
headroom. Lowering the constant to `0.1` made BoN 64 track IRP-EAGER again,
which means the planner behavior was dominated by predictor calibration rather
than by the scheduling policy itself.

To remove this manual tuning dependency, TAPER+ now uses an online calibrated
step-latency model:

```text
predicted_step_ms =
  safety_factor * (
    base_ms
    + width_ms * decode_width
    + context_ms_per_1k * decode_context_tokens / 1000
  )
```

The coefficients are warm-started from
`VLLM_TAPER_PLUS_LATENCY_PER_SEQ_MS` and then updated after each completed
decode step. The scheduler records lightweight profiling metadata in
`SchedulerOutput`:

- `taper_plus_step_start_s`
- `taper_plus_decode_width`
- `taper_plus_context_tokens`
- `taper_plus_prefill_tokens`
- `taper_plus_predicted_step_ms`

When `Scheduler.update_from_output(...)` receives the model output, it computes
the observed step time and applies a normalized online regression update:

```text
error_ms = observed_ms - predicted_ms
beta_i = beta_i + alpha * error_ms * feature_i / (1 + ||features||^2)
```

Only decode-focused steps are used for calibration. Prefill-heavy steps are
filtered out because prompt computation has a different cost surface and would
make branch admission too conservative immediately after prompt bursts.

The implementation still contains a legacy low-pressure guardrail:

```text
if no waiting backlog and enough free KV blocks:
    admit all running branches
```

This guardrail was useful for the earlier fixed-target prototype, but it is now
disabled by default. In the paper-faithful path, TAPER expands toward EAGER
through the slack-budget equation itself rather than through a separate policy
switch.

New runtime controls:

```bash
export VLLM_TAPER_PLUS_ONLINE_CALIBRATION=1
export VLLM_TAPER_PLUS_CALIBRATION_ALPHA=0.08
export VLLM_TAPER_PLUS_SAFETY_FACTOR=1.10
export VLLM_TAPER_PLUS_EAGER_WHEN_UNPRESSURED=0
```

Benchmark flags mirror these environment variables:

```bash
--taper-calibration-alpha 0.08
--taper-safety-factor 1.10
--disable-taper-online-calibration
--enable-taper-eager-when-unpressured
```

This design is intentionally conservative:

- It keeps the stock scheduler path unchanged when
  `VLLM_ENABLE_TAPER_PLUS=0`.
- It preserves IRP-OFF emulation because `VLLM_TAPER_PLUS_STEP_TARGET_MS=0`
  returns the protected baseline only.
- It avoids learning from prefill outliers and very large latency spikes.
- It gives future work a natural upgrade path to richer features, such as
  prefill tokens, total KV footprint, or model-specific kernel mode.

Validation after adding online calibration:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
python benchmarks/benchmark_taper_plus.py \
  --mode all \
  --model Qwen/Qwen3.5-2B \
  --dtype float16 \
  --enforce-eager \
  --max-model-len 256 \
  --gpu-memory-utilization 0.60 \
  --num-requests 4 \
  --request-rate inf \
  --serial-fraction 0.5 \
  --bon-n 16 \
  --output-len 16 \
  --synthetic-prompt-words 32 \
  --ttft-slo 10.0 \
  --tpot-slo 1.0 \
  --taper-step-target-ms 50 \
  --taper-latency-per-seq-ms 2.0 \
  --taper-calibration-alpha 0.08 \
  --taper-safety-factor 1.10 \
  --output-dir benchmark_outputs/taper_plus_qwen35_2b_online_calibration_smoke
```

```text
Output directory:
benchmark_outputs/taper_plus_qwen35_2b_online_calibration_smoke

IRP-OFF:   Avg TTFT 0.3391s, Goodput 5.69 tok/s,  SLO 50.00%
IRP-EAGER: Avg TTFT 0.3017s, Goodput 793.58 tok/s, SLO 100.00%
TAPER+:    Avg TTFT 0.2946s, Goodput 773.79 tok/s, SLO 100.00%
```

This smoke test verifies that the new profiling fields and online update path
execute under CUDA, that IRP-OFF remains protected from the low-pressure eager
guardrail, and that TAPER+ no longer collapses under the old conservative
warm-start value in a low-pressure run.

### Algorithm 1 Rework Validation

The scheduler was reworked from the earlier fixed-threshold prototype to the
paper-style Algorithm 1 planner:

- build a protected baseline `S0`;
- compute a request-level slack budget from the most urgent logical request;
- greedily admit one branch at a time by marginal utility per marginal latency;
- replan every decode step after online calibration updates the latency model.

Because vLLM expands `n > 1` into child requests, the slack deadline is computed
per `external_req_id` logical group. For a parallel group, the TPOT clock is
based on the most recent branch progress in that group, matching the paper's
statement that a parallel-stage request receives `w_r,t` branch-progress tokens
from a widened step. The benchmark now records both logical TPOT and stricter
per-branch TPOT; `--parallel-slo-mode logical` is the paper-faithful default,
while `--parallel-slo-mode branch` is retained as a diagnostic.

Validation command:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 python benchmarks/benchmark_taper_plus.py \
  --mode both \
  --model Qwen/Qwen3.5-2B \
  --dtype float16 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.70 \
  --num-requests 8 \
  --request-rate 1.0 \
  --bon-n 32 \
  --serial-fraction 0.5 \
  --output-len 128 \
  --synthetic-prompt-words 64 \
  --ttft-slo 2.0 \
  --tpot-slo 0.05 \
  --parallel-slo-mode logical \
  --taper-latency-per-seq-ms 0.5 \
  --taper-rho 1.0 \
  --output-dir benchmark_outputs/taper_algorithm1_logical_lowload_bon32_out128_req8_rate1 \
  --enforce-eager
```

Result:

```text
IRP-EAGER: Avg TTFT 0.1085s, P99 TTFT 0.2659s,
           raw throughput 1607.12 tok/s, goodput 414.79 tok/s,
           SLO 12.50%, serial SLO 0.00%, parallel SLO 25.00%

TAPER:     Avg TTFT 0.0960s, P99 TTFT 0.2332s,
           raw throughput 1380.90 tok/s, goodput 721.58 tok/s,
           SLO 37.50%, serial SLO 25.00%, parallel SLO 50.00%
```

This is the expected throughput-trap pattern: IRP-EAGER produces more raw tokens
per second, but TAPER spends less branch externality and therefore delivers more
SLO-valid user-level work under the logical-request SLO.

A heavier run with 16 requests at 4 req/s produced 0% SLO for both policies on
this WSL/Qwen3.5-2B setup, so it is not a useful comparison point yet. That run
is still saved under:

```text
benchmark_outputs/taper_algorithm1_logical_midload_bon32_out128_req16_rate4
```

## Stock-vLLM Compatibility

With `VLLM_ENABLE_TAPER_PLUS=0`, `_get_taper_plus_admitted_running_req_ids()`
returns `None`, and the running request loop behaves as it did before these
changes.

No public API changes are required to run stock vLLM. TAPER+ is opt-in through
environment variables.

## Current Limitations

- The time model is an online-calibrated linear predictor. It should eventually
  be extended with richer features, such as prefill tokens and KV pressure.
- The predictor uses output-length progress by default. EOS log-probability
  support is present in `update_prob_short(...)`, but the sampler path has not
  yet been wired to pass EOS logprobs.
- TAPER+ currently regulates running decode branches. Initial prefill admission
  remains stock vLLM behavior.
- The local validation run used a small public model and did not create KV-cache
  preemption pressure. A larger model or tighter GPU-memory budget should be used
  for the full performance study.

## How to Enable TAPER+

Example:

```bash
export VLLM_ENABLE_TAPER_PLUS=1
export VLLM_TAPER_PLUS_POLICY=taper
export VLLM_TAPER_PLUS_STEP_TARGET_MS=50.0
export VLLM_TAPER_PLUS_LATENCY_PER_SEQ_MS=2.0
export VLLM_TAPER_PLUS_TTFT_SLO_MS=2000.0
export VLLM_TAPER_PLUS_TPOT_SLO_MS=50.0
export VLLM_TAPER_PLUS_RHO=1.0
export VLLM_TAPER_PLUS_UTILITY=linear
export VLLM_TAPER_PLUS_MEMORY_THRESHOLD_BLOCKS=-1
export VLLM_TAPER_PLUS_ONLINE_CALIBRATION=1
export VLLM_TAPER_PLUS_CALIBRATION_ALPHA=0.08
export VLLM_TAPER_PLUS_SAFETY_FACTOR=1.10
export VLLM_TAPER_PLUS_EAGER_WHEN_UNPRESSURED=0
```

On PowerShell:

```powershell
$env:VLLM_ENABLE_TAPER_PLUS = "1"
$env:VLLM_TAPER_PLUS_POLICY = "taper"
$env:VLLM_TAPER_PLUS_STEP_TARGET_MS = "50.0"
$env:VLLM_TAPER_PLUS_LATENCY_PER_SEQ_MS = "2.0"
$env:VLLM_TAPER_PLUS_TTFT_SLO_MS = "2000.0"
$env:VLLM_TAPER_PLUS_TPOT_SLO_MS = "50.0"
$env:VLLM_TAPER_PLUS_RHO = "1.0"
$env:VLLM_TAPER_PLUS_UTILITY = "linear"
$env:VLLM_TAPER_PLUS_MEMORY_THRESHOLD_BLOCKS = "-1"
$env:VLLM_TAPER_PLUS_ONLINE_CALIBRATION = "1"
$env:VLLM_TAPER_PLUS_CALIBRATION_ALPHA = "0.08"
$env:VLLM_TAPER_PLUS_SAFETY_FACTOR = "1.10"
$env:VLLM_TAPER_PLUS_EAGER_WHEN_UNPRESSURED = "0"
```
