# TailBlend Implementation Handoff

This document is for someone who wants to understand or continue the TailBlend
implementation in this vLLM fork. It focuses on the code changes, runtime
switches, benchmark entry points, and safe extension points.

For the research motivation and experimental claims, read
`docs/design/tailblend_full_research_proposal.md`. For reproduction commands,
read `docs/design/tailblend_reproducibility.md`.

## High-Level Summary

TailBlend is implemented as a scheduler-level modification for Best-of-N (BoN)
serving in vLLM. It preserves vLLM's normal eager `n=N` admission behavior by
default and changes only the victim selection decision when KV-cache allocation
already fails.

In stock vLLM, when `kv_cache_manager.allocate_slots(...)` cannot allocate KV
blocks for a running request, the scheduler preempts one running request using
the normal scheduling policy:

- priority scheduling: preempt the running request with the largest
  `(priority, arrival_time)` key;
- otherwise: preempt `self.running[-1]`, effectively the latest running request
  in the scheduler's running list.

TailBlend inserts a policy hook at this same failure point. If TailBlend
preemption is enabled, the hook scores BoN child branches using online
parent/sibling state and returns a victim. If the hook returns `None`, vLLM's
default victim policy still runs.

## Main Files

### `vllm/v1/core/sched/scheduler.py`

This is the integration point with vLLM's v1 scheduler.

Important changes:

- imports `TailBlendController` from
  `vllm.v1.core.sched.tail_blend_controller`;
- accepts `tail_blend` as a `VLLM_TAPER_PLUS_POLICY`;
- normalizes the old `eager_tail_blend` policy name to `tail_blend` for
  backward compatibility;
- reads TailBlend environment variables into scheduler fields:
  `tail_blend_pilot_k`, `tail_blend_preemption`,
  `tail_blend_preemption_mode`, and `tail_blend_reservation_q`;
- creates `self.tail_blend_controller`;
- calls `tail_blend_controller.select_preemption_victim(...)` only after
  `kv_cache_manager.allocate_slots(...)` returns `None`;
- calls `tail_blend_controller.should_defer_waiting_request(...)` for optional
  admission experiments;
- calls `tail_blend_controller.observe_finished_request(...)` when a request
  finishes so completed sibling lengths can be recorded.

The most important invariant is that TailBlend preemption is pressure-only:
the victim hook is reached only after `allocate_slots` fails.
The main `tail_blend` policy also bypasses TAPER+ decode planning and online
latency calibration so no-KV-pressure execution matches Default vLLM-BoN as
closely as possible.

### `vllm/v1/core/sched/tail_blend_controller.py`

This file contains the TailBlend policy logic.

Core responsibilities:

- maintain per-BoN-parent finished sibling output lengths;
- estimate remaining output length for live siblings;
- optionally provide admission hints for experimental pilot/reservation modes;
- select a preemption victim under KV pressure;
- implement ablation modes and exploratory adaptive/prefill-aware variants;
- expose adaptive selector counters for benchmark logging.

Key methods:

- `observe_finished_request(request)`: records finished sibling output lengths
  by `external_req_id`.
- `should_defer_waiting_request(request)`: optional waiting-admission gate.
  With `VLLM_TAIL_BLEND_PILOT_K=0` and
  `VLLM_TAIL_BLEND_RESERVATION_Q=0`, this keeps normal eager admission.
- `select_preemption_victim(current_request)`: pressure-time victim selector.
  This is the main TailBlend hook.
- `_preemption_victim_key(...)`: implements the score used to rank victims.
  The main `full` score is `remaining_norm - invested_progress -
  completion_protection`, where `invested_progress` averages generated output
  progress and recompute cost, and `completion_protection` is the strongest
  parent-rescue, near-finish, or overdue signal.
- `_predict_remaining_tokens(request)`: online sibling-length estimator.
- `_parent_rescue_score(group_id)`: estimates whether a branch may help finish
  an almost-complete BoN parent.
- `_select_adaptive_preemption_mode(...)`: exploratory selector between
  TailBlend, prefill-aware, and default behavior.

The default main policy is `tail_blend_preemption_mode="full"`.

### `vllm/v1/core/sched/eager_tail_blend_controller.py`

This is only a backward-compatible import shim. New code should import
`TailBlendController` from `tail_blend_controller.py`.

### `vllm/envs.py`

This file defines the environment variables used by the scheduler.

Main TailBlend variables:

- `VLLM_ENABLE_TAPER_PLUS=1`: enables the experimental scheduler policy path.
- `VLLM_TAPER_PLUS_POLICY=tail_blend`: selects TailBlend.
- `VLLM_TAIL_BLEND_PREEMPTION=1`: enables TailBlend victim selection after
  KV allocation failure.
- `VLLM_TAIL_BLEND_PREEMPTION_MODE=full`: uses the main TailBlend score.
- `VLLM_TAIL_BLEND_PILOT_K=0`: keeps normal eager admission by default.
- `VLLM_TAIL_BLEND_RESERVATION_Q=0`: disables virtual reservation by default.

Deprecated `VLLM_EAGER_TAIL_BLEND_*` aliases are still present so older local
scripts do not immediately break, but new scripts and documentation should use
`VLLM_TAIL_BLEND_*`.

### `benchmarks/benchmark_taper_plus.py`

This is the main benchmark driver used for TailBlend experiments. It drives
`LLMEngine` directly, submits BoN requests, records streaming outputs, and
writes summary metrics.

Relevant modes:

- `irp_eager`: Default vLLM-BoN baseline.
- `tail_blend`: TailBlend.
- `both_tail_blend`: runs Default vLLM-BoN and TailBlend for comparison.

Relevant CLI flags:

- `--enable-tail-blend-preemption`
- `--tail-blend-preemption-mode full`
- `--tail-blend-pilot-k 0`
- `--tail-blend-reservation-q 0`

Important metrics emitted by this benchmark include:

- user-level goodput;
- SLO attainment;
- total preemptions;
- predictor-selected preemptions;
- preempted computed tokens;
- preempted output tokens;
- adaptive selector counts.

### `benchmarks/run_tail_blend_generalization_sweep.py`

This script runs a repeated dataset/mode/N sweep over the prepared prompt sets.
It is a convenience wrapper around `benchmark_taper_plus.py`.

### `benchmarks/build_tail_blend_prompt_sets.py`

This script builds the JSONL prompt files used by the generalization sweep.
The generated prompt files are written under `benchmark_outputs/`, which is
ignored by Git.

### `requirements/tailblend.txt`

Extra Python packages used for prompt construction and result processing.
Install vLLM itself from this checkout first, then install this file.

## Legacy BoN-TAPER Paths

This repository still contains earlier BoN-TAPER experiment code. These paths
are kept for comparison and research history, but they are not the main
TailBlend implementation path.

Legacy files:

- `vllm/v1/core/sched/bon_taper_controller.py`
- `vllm/v1/core/sched/bon_taper_len_predictor.py`
- `docs/design/taper_bon_meeting_outline_ko.md`

Legacy policy names that still exist in `scheduler.py`,
`vllm/envs.py`, and `benchmarks/benchmark_taper_plus.py`:

- `bon_taper`
- `bon_taper_plus`
- `bon_taper_len_predictor`

Do not start from these files when implementing or reviewing TailBlend. Start
from `tail_blend_controller.py` and the `tail_blend` policy instead. The
BoN-TAPER paths regulate BoN branch admission/planning more directly, while the
main TailBlend result is about pressure-time preemption victim selection under
otherwise eager vLLM-BoN execution.

The legacy paths should be removed only on a dedicated cleanup branch after
checking that benchmark modes, scheduler policy validation, and old experiment
scripts no longer need them.

## Runtime Configuration

Minimal TailBlend run:

```bash
VLLM_ENABLE_TAPER_PLUS=1
VLLM_TAPER_PLUS_POLICY=tail_blend
VLLM_TAIL_BLEND_PREEMPTION=1
VLLM_TAIL_BLEND_PREEMPTION_MODE=full
VLLM_TAIL_BLEND_PILOT_K=0
VLLM_TAIL_BLEND_RESERVATION_Q=0
```

The important setting is `VLLM_TAIL_BLEND_PREEMPTION=1`. Without it,
`tail_blend` does not replace the default preemption victim.

## TailBlend Score Modes

`VLLM_TAIL_BLEND_PREEMPTION_MODE` supports:

- `remaining`: ablation using only predicted remaining output length.
- `remaining_recompute`: remaining length plus recompute protection.
- `remaining_parent`: remaining length plus parent-completion protection.
- `full`: main TailBlend score.
- `full_v2`: experimental KV-freed score.
- `full_slo`: experimental SLO-feasibility score.
- `prefill_aware`: recompute/prefix-heavy victim scoring.
- `prefill_aware_gated`: uses prefill-aware scoring only when a gate fires.
- `adaptive`: exploratory runtime selector.

The proposal currently treats `full` as the main policy. The adaptive and
prefill-aware modes are useful for failure-mode analysis but should be reported
as exploratory unless they are revalidated.

## What TailBlend Does Not Change

TailBlend does not modify:

- CUDA kernels;
- PagedAttention block layout;
- KV-cache allocation internals;
- model forward kernels;
- tokenizer behavior;
- sampling semantics;
- the user-facing meaning of `n=N`.

The current implementation is scheduler-level systems work. It changes which
already-running request is preempted under KV pressure.

## State Model

TailBlend relies on BoN child branches having a shared logical parent id. In
this implementation, that parent id is represented by `request.external_req_id`.

For each parent, the controller tracks:

- finished sibling output lengths;
- last progress time;
- live siblings currently in `scheduler.running`;
- queued siblings in `scheduler.waiting` and `scheduler.skipped_waiting`.

The online length estimator uses finished sibling lengths when available and
falls back to the request's output-token budget when there is not enough parent
specific evidence.

## Safe Extension Points

For someone continuing the project, these are the lowest-risk places to modify:

- change the main score in `_preemption_victim_key(...)`;
- add a new ablation mode in `_preemption_victim_key(...)` and the scheduler's
  allowed mode list;
- modify `_predict_remaining_tokens(...)` to test a different online predictor;
- add counters in `Scheduler` and expose them through
  `benchmark_taper_plus.py`;
- tune adaptive pressure thresholds in `_select_adaptive_preemption_mode(...)`;
- add benchmark sweeps in `run_tail_blend_generalization_sweep.py`.

Higher-risk changes:

- changing `kv_cache_manager.allocate_slots(...)`;
- changing PagedAttention block allocation/free behavior;
- changing request lifecycle or request ids;
- changing `n=N` sampling semantics;
- changing scheduler list ordering globally.

Those higher-risk changes can affect all vLLM serving modes, not only
TailBlend.

## Suggested Read Order

1. `docs/design/tailblend_full_research_proposal.md`
2. `docs/design/tailblend_reproducibility.md`
3. `vllm/envs.py`
4. `vllm/v1/core/sched/scheduler.py`
5. `vllm/v1/core/sched/tail_blend_controller.py`
6. `benchmarks/benchmark_taper_plus.py`
7. `benchmarks/run_tail_blend_generalization_sweep.py`

## Handoff Checklist

Before running new experiments:

- confirm the checkout is installed with `pip install -e .`;
- install `requirements/tailblend.txt`;
- build or verify prompt JSONLs under `benchmark_outputs/`;
- run a smoke test with a small request count;
- check that `comparison.json`, `*_summary.json`, and preemption counters are
  produced;
- keep `benchmark_outputs/` out of Git unless intentionally publishing a small
  curated artifact.

Before reporting new results:

- compare `irp_eager` against `tail_blend`;
- record model, GPU, request rate, seed, `N`, `max_output_tokens`, and
  `max_model_len`;
- report both goodput and SLO attainment;
- include preemption/recompute counters to explain why a policy wins or loses;
- separate main TailBlend results from adaptive/prefill-aware exploration.
