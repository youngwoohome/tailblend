# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TailBlend preemption and optional BoN lifetime hints.

This controller intentionally keeps stock eager admission semantics: all BoN
children are allowed to start as normal. It only influences which already
running decode branches receive the next decode step when the scheduler cannot
serve every runnable branch in one step. Once sibling branches have finished,
their lengths provide an online order-statistics signal for the remaining live
siblings.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil
from typing import TYPE_CHECKING

from vllm.v1.engine import FinishReason
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler


@dataclass
class _TailBlendGroupState:
    finished_lengths: list[int] = field(default_factory=list)
    last_progress_time_s: float = 0.0


@dataclass(frozen=True)
class _ParentSLOState:
    doomed: float
    urgent_feasible: float
    work_norm: float


@dataclass(frozen=True)
class _AdaptivePressureState:
    prefill_pressure: float
    output_tail_pressure: float
    kv_pressure: float
    backlog_ratio: float
    prompt_share: float
    recompute_risk: float
    tail_signal: float
    tail_skew: float


@dataclass(frozen=True)
class _TailBlendAdmissionPressure:
    active: bool
    running_util: float
    waiting_backlog: int
    uncovered_groups: int
    min_uncovered_slack_s: float
    avg_group_fanout: float
    free_block_ratio: float


class TailBlendController:
    """Running-only BoN tail-blend prioritizer.

    ``tail_blend`` does not defer waiting requests by default. It records completed
    sibling lengths and, under scheduling pressure, prioritizes logical parents
    that are cheap to complete, then prioritizes the shortest predicted live
    siblings inside the selected parent:

        parent_remaining = sum(tail_blend_remaining for live siblings)

    The predictor is only used for live branches. Unlike BoN-TAPER's deferred
    siblings, every candidate here has already started, so the live branch's
    current output length is a valid lower bound.
    """

    # Matches the offline experiment default. Larger values trust finished
    # siblings more slowly.
    _SHRINK_K = 4.0
    _QUEUED_PRIOR_FRACTION = 0.75

    def __init__(self, scheduler: "Scheduler") -> None:
        self.scheduler = scheduler
        self._groups: dict[str, _TailBlendGroupState] = defaultdict(
            _TailBlendGroupState
        )
        self._adaptive_choice_counts: dict[str, int] = defaultdict(int)
        self._last_adaptive_pressure_state: _AdaptivePressureState | None = None

    @property
    def adaptive_choice_counts(self) -> dict[str, int]:
        return dict(self._adaptive_choice_counts)

    def observe_finished_request(self, request: Request) -> None:
        if request.sampling_params is None or request.external_req_id is None:
            return
        if request.get_finished_reason() not in (
            FinishReason.STOP,
            FinishReason.LENGTH,
            FinishReason.REPETITION,
        ):
            return
        state = self._groups[request.external_req_id]
        state.finished_lengths.append(request.num_output_tokens)
        state.last_progress_time_s = max(
            state.last_progress_time_s,
            request.taper_last_token_time,
            time.time(),
        )

    def should_defer_waiting_request(self, request: Request) -> bool:
        """Optional BoN waiting admission hints.

        With ``VLLM_TAIL_BLEND_PILOT_K=0`` this controller keeps stock
        eager admission. When k is positive, it still stays eager while the
        system is not under pressure. Under pressure, each logical parent gets
        k child branches admitted first; remaining siblings are expanded only
        after finished sibling lengths provide a parent-level work estimate.

        With ``VLLM_TAIL_BLEND_RESERVATION_Q>0``, the controller also
        uses a planning-only predicted output length to avoid admitting more
        branches than the remaining KV cache can plausibly hold. It does not
        change the real ``max_tokens`` stopping condition.
        """
        scheduler = self.scheduler
        if (
            not scheduler.enable_taper_plus
            or scheduler.taper_plus_policy != "tail_blend"
            or request.external_req_id is None
        ):
            return False

        if self._should_defer_for_virtual_reservation(request):
            return True

        if self._should_defer_for_ttft_guard(request):
            return True

        pilot_k = scheduler.tail_blend_pilot_k
        if pilot_k <= 0 or not self._under_admission_pressure():
            return False

        group_id = scheduler._taper_plus_group_id(request)
        started_siblings = self._started_sibling_count(group_id)
        if started_siblings < pilot_k:
            return False

        state = self._groups.get(group_id)
        if state is None or not state.finished_lengths:
            return True

        best_group_id = self._best_expandable_group_id()
        return best_group_id is not None and group_id != best_group_id

    def plan_admitted_running_req_ids(self, token_budget: int) -> set[str] | None:
        """TailBlend TTFT guards do not regulate running decode width.

        Earlier TTFT-guard variants also selected a subset of already-running
        branches for decode. Long-prompt runs showed that this can preserve TTFT
        while destroying TPOT, so admission guardrails now only affect waiting
        requests and leave running decode scheduling on the stock eager path.
        """
        return None

    def select_preemption_victim(
        self,
        current_request: Request,
    ) -> Request | None:
        """Pick a KV-pressure victim using BoN parent completion hints.

        This is intentionally called only after ``allocate_slots`` fails, so it
        does not reduce eager admission or decode width. The default vLLM LIFO
        path remains active unless ``VLLM_TAIL_BLEND_PREEMPTION=1``.
        """
        scheduler = self.scheduler
        if (
            not scheduler.enable_taper_plus
            or scheduler.taper_plus_policy != "tail_blend"
            or not scheduler.tail_blend_preemption
        ):
            return None

        candidates = [
            request
            for request in scheduler.running
            if request.sampling_params is not None
            and request.external_req_id is not None
        ]
        if not candidates:
            return None

        now_s = time.time()
        mode = scheduler.tail_blend_preemption_mode
        effective_mode: str | None = mode
        if mode == "adaptive":
            effective_mode = self._select_adaptive_preemption_mode(
                current_request,
                candidates,
            )
            self._adaptive_choice_counts[effective_mode or "default"] += 1
            if effective_mode is None:
                return None

        if (
            mode == "prefill_aware_gated"
            and not self._should_use_prefill_aware_preemption(current_request)
        ):
            return None

        return max(
            candidates,
            key=lambda request: self._preemption_victim_key(
                request,
                current_request,
                now_s,
                effective_mode,
            ),
        )

    def _parent_rank_key(
        self,
        group_id: str,
        requests: list[Request],
        queued_siblings: int,
        now_s: float,
    ) -> tuple[bool, bool, float, float, int, float, str]:
        state = self._groups.get(group_id)
        finished_count = len(state.finished_lengths) if state is not None else 0
        predicted_remaining = [
            self._predict_remaining_tokens(request) for request in requests
        ]
        parent_remaining_sum = sum(predicted_remaining)
        parent_remaining_max = max(predicted_remaining, default=0.0)
        if queued_siblings > 0:
            max_tokens = max((request.max_tokens for request in requests), default=1)
            parent_remaining_sum += queued_siblings * max_tokens
            parent_remaining_max = max(parent_remaining_max, float(max_tokens))

        deadline_s = self.scheduler._get_taper_plus_group_deadline_s(requests)
        slack_s = deadline_s - now_s
        is_overdue = slack_s <= 0.0
        has_queued_siblings = queued_siblings > 0
        oldest_arrival = min(request.arrival_time for request in requests)

        return (
            has_queued_siblings,
            is_overdue,
            parent_remaining_sum,
            parent_remaining_max,
            -finished_count,
            oldest_arrival,
            group_id,
        )

    def _preemption_victim_key(
        self,
        request: Request,
        current_request: Request,
        now_s: float,
        effective_mode: str | None = None,
    ) -> tuple[float, float, float, str]:
        max_tokens = max(1, request.max_tokens)
        remaining = self._predict_remaining_tokens(request)
        remaining_norm = remaining / float(max_tokens)
        output_progress = min(1.0, request.num_output_tokens / float(max_tokens))
        recompute_cost = request.num_computed_tokens / float(
            max(1, request.num_prompt_tokens + max_tokens)
        )

        group_id = self.scheduler._taper_plus_group_id(request)
        parent_rescue = self._parent_rescue_score(group_id)
        near_finish = remaining <= max(32.0, 0.05 * float(max_tokens))
        overdue = now_s > self.scheduler._get_taper_plus_deadline_s(request)
        mode = effective_mode or self.scheduler.tail_blend_preemption_mode

        # Higher score means "better victim". Protect work that is expensive to
        # recompute, close to finishing, overdue, or likely to complete a parent.
        if mode == "remaining":
            score = remaining_norm
        elif mode == "remaining_recompute":
            score = (
                1.4 * remaining_norm
                - 0.9 * recompute_cost
                - (1.0 if near_finish else 0.0)
                - 0.2 * request.num_preemptions
            )
        elif mode == "remaining_parent":
            score = (
                1.4 * remaining_norm
                - 1.1 * parent_rescue
                - (1.0 if near_finish else 0.0)
                - 0.2 * request.num_preemptions
            )
        elif mode == "full_v2":
            score = self._kv_freed_preemption_score(
                request=request,
                remaining_norm=remaining_norm,
                output_progress=output_progress,
                recompute_cost=recompute_cost,
                parent_rescue=parent_rescue,
                near_finish=near_finish,
                overdue=overdue,
            )
        elif mode == "full_slo":
            score = self._slo_feasibility_preemption_score(
                request=request,
                group_id=group_id,
                now_s=now_s,
                remaining_norm=remaining_norm,
                output_progress=output_progress,
                recompute_cost=recompute_cost,
                parent_rescue=parent_rescue,
                near_finish=near_finish,
                overdue=overdue,
            )
        elif mode in ("prefill_aware", "prefill_aware_gated"):
            score = self._prefill_aware_preemption_score(
                request=request,
                output_progress=output_progress,
                recompute_cost=recompute_cost,
                parent_rescue=parent_rescue,
                near_finish=near_finish,
                overdue=overdue,
            )
        elif mode == "complicated":
            score = (
                1.6 * remaining_norm
                - 1.2 * output_progress
                - 1.0 * parent_rescue
                - 0.8 * recompute_cost
                - (1.0 if near_finish else 0.0)
                - (0.4 if overdue else 0.0)
                - 0.2 * request.num_preemptions
            )
        else:
            invested_progress = 0.5 * (output_progress + recompute_cost)
            completion_protection = max(
                parent_rescue,
                1.0 if near_finish else 0.0,
                1.0 if overdue else 0.0,
            )
            score = remaining_norm - invested_progress - completion_protection

        # If the failing request itself is clearly weak, allow preempting it;
        # on ties, free another branch so the current request can still run.
        current_tiebreak = -1.0 if request is current_request else 0.0
        return (score, current_tiebreak, request.arrival_time, request.request_id)

    def _select_adaptive_preemption_mode(
        self,
        current_request: Request,
        candidates: list[Request],
    ) -> str | None:
        pressure = self._adaptive_pressure_state(current_request, candidates)
        self._last_adaptive_pressure_state = pressure

        if (
            pressure.prefill_pressure >= 0.30
            and pressure.prefill_pressure >= pressure.output_tail_pressure + 0.10
        ):
            return "prefill_aware"

        if (
            pressure.output_tail_pressure >= 0.14
            and pressure.output_tail_pressure >= pressure.prefill_pressure + 0.05
        ):
            if self._full_candidate_has_high_prefix_waste(
                current_request,
                candidates,
            ):
                self._adaptive_choice_counts["full_prefix_guard"] += 1
                return None
            return "full"

        return None

    def _full_candidate_has_high_prefix_waste(
        self,
        current_request: Request,
        candidates: list[Request],
    ) -> bool:
        now_s = time.time()
        victim = max(
            candidates,
            key=lambda request: self._preemption_victim_key(
                request,
                current_request,
                now_s,
                "full",
            ),
        )
        prefix_threshold = max(128.0, 0.10 * float(max(1, victim.max_tokens)))
        return victim.num_output_tokens > prefix_threshold

    def _adaptive_pressure_state(
        self,
        current_request: Request,
        candidates: list[Request],
    ) -> _AdaptivePressureState:
        scheduler = self.scheduler
        try:
            total_blocks = max(1, scheduler.kv_cache_manager.block_pool.num_gpu_blocks)
            free_blocks = scheduler.kv_cache_manager.get_remaining_free_blocks()
            kv_util = 1.0 - min(1.0, free_blocks / float(total_blocks))
        except Exception:
            kv_util = 1.0
        kv_pressure = self._clamp01((kv_util - 0.85) / 0.15)

        waiting_backlog = len(scheduler.waiting) + len(scheduler.skipped_waiting)
        running_count = max(1, len(scheduler.running))
        backlog_ratio = waiting_backlog / float(running_count)
        backlog_pressure = self._clamp01((backlog_ratio - 1.5) / 4.0)

        requests = list(candidates)
        if all(
            request.request_id != current_request.request_id for request in requests
        ):
            requests.append(current_request)

        prompt_tokens = sum(max(0, request.num_prompt_tokens) for request in requests)
        total_tokens = sum(
            max(1, request.num_prompt_tokens + max(1, request.max_tokens))
            for request in requests
        )
        prompt_share = prompt_tokens / float(max(1, total_tokens))
        recompute_risk = sum(
            min(request.num_computed_tokens, request.num_prompt_tokens)
            for request in requests
        ) / float(max(1, prompt_tokens))

        prefill_pressure = (
            kv_pressure
            * backlog_pressure
            * prompt_share
            * self._clamp01(recompute_risk)
        )

        active_group_ids = {
            scheduler._taper_plus_group_id(request)
            for request in requests
            if request.external_req_id is not None
        }
        groups_with_tail_signal = sum(
            1
            for group_id in active_group_ids
            if self._groups.get(group_id) is not None
            and bool(self._groups[group_id].finished_lengths)
        )
        tail_signal = (
            groups_with_tail_signal / float(len(active_group_ids))
            if active_group_ids
            else 0.0
        )

        remaining_values = [
            self._predict_remaining_tokens(request)
            for request in requests
            if request.sampling_params is not None
        ]
        p50_remaining = self._percentile_float(remaining_values, 0.50)
        p90_remaining = self._percentile_float(remaining_values, 0.90)
        tail_skew = 0.0
        if p50_remaining > 1.0:
            tail_skew = self._clamp01((p90_remaining / p50_remaining - 1.0) / 1.5)

        decode_candidates = sum(
            1
            for request in scheduler.running
            if scheduler._is_taper_plus_decode_candidate(request)
        )
        decode_pressure = self._clamp01(
            (decode_candidates / float(running_count) - 0.75) / 0.25
        )
        output_share = 1.0 - prompt_share
        output_tail_pressure = (
            kv_pressure
            * output_share
            * tail_signal
            * max(0.25, tail_skew)
            * (0.5 + 0.5 * decode_pressure)
        )

        return _AdaptivePressureState(
            prefill_pressure=prefill_pressure,
            output_tail_pressure=output_tail_pressure,
            kv_pressure=kv_pressure,
            backlog_ratio=backlog_ratio,
            prompt_share=prompt_share,
            recompute_risk=self._clamp01(recompute_risk),
            tail_signal=tail_signal,
            tail_skew=tail_skew,
        )

    def _should_use_prefill_aware_preemption(
        self,
        current_request: Request,
    ) -> bool:
        total_tokens = max(
            1,
            current_request.num_prompt_tokens + max(1, current_request.max_tokens),
        )
        prompt_share = current_request.num_prompt_tokens / float(total_tokens)
        prompt_heavy = prompt_share >= 0.50 or current_request.num_prompt_tokens >= max(
            1024,
            current_request.max_tokens,
        )
        if not prompt_heavy:
            return False

        waiting_backlog = len(self.scheduler.waiting) + len(
            self.scheduler.skipped_waiting
        )
        running_count = max(1, len(self.scheduler.running))
        return waiting_backlog >= 3.5 * running_count

    def _prefill_aware_preemption_score(
        self,
        *,
        request: Request,
        output_progress: float,
        recompute_cost: float,
        parent_rescue: float,
        near_finish: bool,
        overdue: bool,
    ) -> float:
        """Score victims for prompt/prefill-heavy workloads.

        ``full`` is output-tail oriented: it uses predicted remaining length to
        protect branches that can help complete a BoN parent soon. For
        prefill-heavy prompts, the larger loss is often recomputing an already
        materialized long prompt and generated prefix. This module therefore
        ignores predicted tail length and chooses cheap-to-lose victims:
        branches with little computed work, little generated output, and low
        parent rescue value.
        """
        total_tokens = max(1, request.num_prompt_tokens + max(1, request.max_tokens))
        prompt_share = min(1.0, request.num_prompt_tokens / float(total_tokens))
        computed_norm = min(1.0, request.num_computed_tokens / float(total_tokens))
        allocated_blocks = self._allocated_block_count(request)
        capacity_blocks = max(
            1,
            ceil(total_tokens / max(1, self.scheduler.block_size)),
        )
        freed_ratio = min(1.0, allocated_blocks / float(capacity_blocks))

        cheap_to_recompute = 1.0 - computed_norm
        no_token_bonus = 0.25 if request.num_output_tokens == 0 else 0.0
        prompt_recompute_damage = recompute_cost * (0.45 + 0.85 * prompt_share)
        generated_prefix_damage = min(
            1.0,
            request.num_output_tokens / max(64.0, 0.10 * request.max_tokens),
        )

        return (
            1.40 * cheap_to_recompute
            + 0.30 * freed_ratio
            + no_token_bonus
            - 1.30 * prompt_recompute_damage
            - 1.80 * generated_prefix_damage
            - 0.95 * parent_rescue
            - (0.95 if near_finish else 0.0)
            - (0.35 if overdue else 0.0)
            - 0.20 * request.num_preemptions
        )

    def _kv_freed_preemption_score(
        self,
        *,
        request: Request,
        remaining_norm: float,
        output_progress: float,
        recompute_cost: float,
        parent_rescue: float,
        near_finish: bool,
        overdue: bool,
    ) -> float:
        """Score victims by pressure relief per expected parent-level damage.

        This is an ablation module for ``full_v2``. It keeps the same
        pressure-only intervention point as ``full`` but adds the amount of KV
        that would actually be released by preempting this branch. The score is
        still conservative around near-finish, overdue, and parent-rescue
        branches, where preemption is most likely to hurt user-level goodput.
        """
        capacity_blocks = max(
            1,
            ceil(
                (request.num_prompt_tokens + max(1, request.max_tokens))
                / max(1, self.scheduler.block_size)
            ),
        )
        allocated_blocks = self._allocated_block_count(request)
        freed_ratio = min(1.0, allocated_blocks / float(capacity_blocks))
        future_blocks = self._predicted_future_blocks(request)
        future_ratio = min(1.0, future_blocks / float(capacity_blocks))

        pressure_relief = (
            1.35 * remaining_norm + 0.55 * freed_ratio + 0.35 * future_ratio
        )
        parent_damage = (
            1.15 * output_progress
            + 0.95 * recompute_cost
            + 1.25 * parent_rescue
            + (1.10 if near_finish else 0.0)
            + (0.45 if overdue else 0.0)
            + 0.20 * request.num_preemptions
        )
        return pressure_relief - parent_damage

    def _slo_feasibility_preemption_score(
        self,
        *,
        request: Request,
        group_id: str,
        now_s: float,
        remaining_norm: float,
        output_progress: float,
        recompute_cost: float,
        parent_rescue: float,
        near_finish: bool,
        overdue: bool,
    ) -> float:
        """Score victims by parent-level SLO feasibility.

        ``full_slo`` keeps the same pressure-only intervention point as
        ``full``. Its only difference is that it first asks whether the logical
        BoN parent can still plausibly satisfy TTFT/TPOT. Branches from already
        missed parents become better victims; branches from urgent but still
        feasible parents are protected.
        """
        parent_state = self._parent_slo_state(group_id, now_s)
        doomed = parent_state.doomed
        feasible = 1.0 - doomed
        rescue_value = parent_rescue * feasible
        near_finish_value = (1.0 if near_finish else 0.0) * feasible
        recompute_value = recompute_cost * (0.45 + 0.55 * feasible)
        output_value = output_progress * (0.50 + 0.50 * feasible)
        branch_overdue_protection = (
            0.30 if overdue and parent_state.doomed < 0.5 else 0.0
        )

        return (
            1.45 * remaining_norm
            + 1.30 * doomed
            + 0.35 * parent_state.work_norm
            - 1.20 * rescue_value
            - 1.00 * near_finish_value
            - 0.90 * recompute_value
            - 1.05 * output_value
            - 0.85 * parent_state.urgent_feasible
            - branch_overdue_protection
            - 0.20 * request.num_preemptions
        )

    def _parent_slo_state(
        self,
        group_id: str,
        now_s: float,
    ) -> _ParentSLOState:
        scheduler = self.scheduler
        state = self._groups.get(group_id)
        finished_count = len(state.finished_lengths) if state is not None else 0
        progress_times: list[float] = []
        if state is not None and state.last_progress_time_s > 0.0:
            progress_times.append(state.last_progress_time_s)

        running_requests = [
            request
            for request in scheduler.running
            if scheduler._taper_plus_group_id(request) == group_id
            and request.sampling_params is not None
        ]
        queued_requests = [
            request
            for request in scheduler.waiting
            if scheduler._taper_plus_group_id(request) == group_id
            and request.sampling_params is not None
        ]
        queued_requests.extend(
            request
            for request in scheduler.skipped_waiting
            if scheduler._taper_plus_group_id(request) == group_id
            and request.sampling_params is not None
        )

        for request in running_requests:
            if request.num_output_tokens > 0:
                progress_times.append(request.taper_last_token_time)

        all_requests = running_requests + queued_requests
        max_tokens = max((request.max_tokens for request in all_requests), default=1)
        total_count = max(1, finished_count + len(all_requests))
        predicted_work = sum(
            self._predict_remaining_tokens(request) for request in all_requests
        )
        work_norm = min(1.0, predicted_work / float(total_count * max_tokens))

        if progress_times:
            deadline_s = max(progress_times) + scheduler.taper_plus_tpot_slo_s
            slo_window_s = max(1e-3, scheduler.taper_plus_tpot_slo_s)
        else:
            first_arrival = min(
                (request.arrival_time for request in all_requests),
                default=now_s,
            )
            deadline_s = first_arrival + scheduler.taper_plus_ttft_slo_s
            slo_window_s = max(1e-3, scheduler.taper_plus_ttft_slo_s)

        slack_s = deadline_s - now_s
        doomed = min(1.0, max(0.0, -slack_s / (0.5 * slo_window_s)))
        urgent_feasible = 0.0
        if slack_s >= 0.0:
            urgent_feasible = min(1.0, max(0.0, 1.0 - slack_s / slo_window_s))

        return _ParentSLOState(
            doomed=doomed,
            urgent_feasible=urgent_feasible,
            work_norm=work_norm,
        )

    def _allocated_block_count(self, request: Request) -> int:
        try:
            block_ids = self.scheduler.kv_cache_manager.get_block_ids(
                request.request_id
            )
        except Exception:
            return max(
                1,
                ceil(request.num_computed_tokens / max(1, self.scheduler.block_size)),
            )
        return max(1, sum(len(group) for group in block_ids))

    def _predicted_future_blocks(self, request: Request) -> int:
        predicted_remaining = ceil(self._predict_remaining_tokens(request))
        predicted_output_tokens = request.num_output_tokens + predicted_remaining
        try:
            kv_cache_manager = self.scheduler.kv_cache_manager
            return kv_cache_manager.estimate_future_footprint_for_output_len(
                request,
                predicted_output_tokens,
            )
        except Exception:
            remaining_tokens = max(
                0,
                min(request.max_tokens, predicted_output_tokens)
                - request.num_output_tokens,
            )
            return ceil(remaining_tokens / max(1, self.scheduler.block_size))

    def _parent_rescue_score(self, group_id: str) -> float:
        state = self._groups.get(group_id)
        finished_count = len(state.finished_lengths) if state is not None else 0
        live_requests = [
            request
            for request in self.scheduler.running
            if self.scheduler._taper_plus_group_id(request) == group_id
        ]
        queued_count = 0
        for request in self.scheduler.waiting:
            if self.scheduler._taper_plus_group_id(request) == group_id:
                queued_count += 1
        for request in self.scheduler.skipped_waiting:
            if self.scheduler._taper_plus_group_id(request) == group_id:
                queued_count += 1

        near_finish_live = 0
        for request in live_requests:
            threshold = max(32.0, 0.05 * float(max(1, request.max_tokens)))
            if self._predict_remaining_tokens(request) <= threshold:
                near_finish_live += 1

        total_count = max(1, finished_count + len(live_requests) + queued_count)
        return (finished_count + near_finish_live) / float(total_count)

    def _under_admission_pressure(self) -> bool:
        scheduler = self.scheduler
        memory_threshold = scheduler._get_taper_plus_memory_threshold()
        free_blocks = scheduler.kv_cache_manager.get_remaining_free_blocks()
        waiting_backlog = len(scheduler.waiting) + len(scheduler.skipped_waiting)
        running_pressure = (
            len(scheduler.running) >= (scheduler.max_num_running_reqs * 3) // 4
        )
        return (
            free_blocks < memory_threshold * 4
            or waiting_backlog > scheduler.max_num_running_reqs
            or running_pressure
        )

    def _should_defer_for_ttft_guard(self, request: Request) -> bool:
        scheduler = self.scheduler
        mode = scheduler.tail_blend_admission_mode
        if mode in ("ttft_fair_guarded", "ttft_cost_guarded"):
            return self._should_defer_for_ttft_fair_guard(request)
        if mode != "ttft_guarded":
            return False

        now_s = time.time()
        pressure = self._tail_blend_admission_pressure(now_s)
        if not pressure.active:
            return False
        if self._should_preserve_prompt_locality_for_ttft_guard(
            request,
            running_util=pressure.running_util,
            waiting_backlog=pressure.waiting_backlog,
            free_block_ratio=pressure.free_block_ratio,
        ):
            return False

        group_id = scheduler._taper_plus_group_id(request)
        protected_width = max(1, scheduler.tail_blend_pilot_k)
        started_siblings = self._started_sibling_count(group_id)
        running_requests = [
            running_request
            for running_request in scheduler.running
            if scheduler._taper_plus_group_id(running_request) == group_id
        ]

        if started_siblings < protected_width:
            return False

        if not self._group_has_first_token(group_id, running_requests):
            return True

        best_group_id = self._best_expandable_group_id()
        if best_group_id is None:
            return True
        return group_id != best_group_id

    def _should_defer_for_ttft_fair_guard(self, request: Request) -> bool:
        scheduler = self.scheduler
        if request.external_req_id is None:
            return False

        group_id = scheduler._taper_plus_group_id(request)
        protected_width = max(1, scheduler.tail_blend_pilot_k)
        if self._started_sibling_count(group_id) < protected_width:
            return False

        waiting_backlog = len(scheduler.waiting) + len(scheduler.skipped_waiting)
        if waiting_backlog <= 0:
            return False
        if self._should_preserve_prompt_locality_for_ttft_guard(
            request,
            waiting_backlog=waiting_backlog,
        ):
            return False

        max_running = max(1, scheduler.max_num_running_reqs)
        running_util = len(scheduler.running) / float(max_running)
        return running_util >= 0.50 or self._has_zero_start_waiting_group(
            exclude_group_id=group_id
        )

    def _should_preserve_prompt_locality_for_ttft_guard(
        self,
        request: Request,
        *,
        running_util: float | None = None,
        waiting_backlog: int | None = None,
        free_block_ratio: float | None = None,
    ) -> bool:
        if request.external_req_id is None:
            return False

        prompt_tokens = max(0, request.num_prompt_tokens)
        max_tokens = max(1, request.max_tokens)
        total_tokens = max(1, prompt_tokens + max_tokens)
        prompt_share = prompt_tokens / float(total_tokens)
        prompt_heavy = prompt_tokens >= 1024 and (
            prompt_share >= 0.40 or prompt_tokens >= max_tokens
        )
        if not prompt_heavy:
            return False

        scheduler = self.scheduler
        if waiting_backlog is None:
            waiting_backlog = len(scheduler.waiting) + len(scheduler.skipped_waiting)
        if waiting_backlog <= 0:
            return False

        if running_util is None:
            max_running = max(1, scheduler.max_num_running_reqs)
            running_util = len(scheduler.running) / float(max_running)
        if free_block_ratio is None:
            try:
                total_blocks = max(
                    1,
                    scheduler.kv_cache_manager.block_pool.num_gpu_blocks,
                )
                free_blocks = scheduler.kv_cache_manager.get_remaining_free_blocks()
                free_block_ratio = min(1.0, free_blocks / float(total_blocks))
            except Exception:
                free_block_ratio = 1.0

        return running_util >= 0.75 or free_block_ratio <= 0.25

    def _has_zero_start_waiting_group(self, exclude_group_id: str) -> bool:
        scheduler = self.scheduler
        started_group_ids = {
            scheduler._taper_plus_group_id(request)
            for request in scheduler.running
            if request.external_req_id is not None
        }
        started_group_ids.update(
            group_id
            for group_id, state in self._groups.items()
            if state.finished_lengths
        )
        for queue in (scheduler.waiting, scheduler.skipped_waiting):
            for request in queue:
                if request.external_req_id is None:
                    continue
                group_id = scheduler._taper_plus_group_id(request)
                if group_id != exclude_group_id and group_id not in started_group_ids:
                    return True
        return False

    def _tail_blend_admission_pressure(
        self,
        now_s: float,
    ) -> _TailBlendAdmissionPressure:
        scheduler = self.scheduler
        mode = scheduler.tail_blend_admission_mode
        if mode != "ttft_guarded" or scheduler.taper_plus_ttft_slo_s <= 0.0:
            return _TailBlendAdmissionPressure(
                active=False,
                running_util=0.0,
                waiting_backlog=0,
                uncovered_groups=0,
                min_uncovered_slack_s=float("inf"),
                avg_group_fanout=0.0,
                free_block_ratio=1.0,
            )

        max_running = max(1, scheduler.max_num_running_reqs)
        running_util = len(scheduler.running) / float(max_running)
        waiting_backlog = len(scheduler.waiting) + len(scheduler.skipped_waiting)

        group_arrivals: dict[str, float] = {}
        group_has_first: set[str] = set()
        running_group_counts: dict[str, int] = defaultdict(int)
        decode_candidates = 0

        def observe_request(request: Request, *, running: bool) -> None:
            if request.external_req_id is None:
                return
            group_id = scheduler._taper_plus_group_id(request)
            group_arrivals[group_id] = min(
                group_arrivals.get(group_id, request.arrival_time),
                request.arrival_time,
            )
            state = self._groups.get(group_id)
            if request.num_output_tokens > 0 or (
                state is not None and state.finished_lengths
            ):
                group_has_first.add(group_id)
            if running:
                running_group_counts[group_id] += 1

        for running_request in scheduler.running:
            observe_request(running_request, running=True)
            if scheduler._is_taper_plus_decode_candidate(running_request):
                decode_candidates += 1
        for waiting_request in scheduler.waiting:
            observe_request(waiting_request, running=False)
        for waiting_request in scheduler.skipped_waiting:
            observe_request(waiting_request, running=False)

        uncovered_slacks = [
            arrival_s + scheduler.taper_plus_ttft_slo_s - now_s
            for group_id, arrival_s in group_arrivals.items()
            if group_id not in group_has_first
        ]
        uncovered_groups = len(uncovered_slacks)
        min_uncovered_slack_s = (
            min(uncovered_slacks) if uncovered_slacks else float("inf")
        )
        active_running_groups = max(1, len(running_group_counts))
        avg_group_fanout = decode_candidates / float(active_running_groups)

        try:
            total_blocks = max(1, scheduler.kv_cache_manager.block_pool.num_gpu_blocks)
            free_blocks = scheduler.kv_cache_manager.get_remaining_free_blocks()
            free_block_ratio = min(1.0, free_blocks / float(total_blocks))
        except Exception:
            free_block_ratio = 1.0

        active = False
        if uncovered_groups > 0:
            ttft_slo = scheduler.taper_plus_ttft_slo_s
            backlog_threshold = max(8, max_running // 8)
            active = (
                min_uncovered_slack_s <= 0.50 * ttft_slo
                or (running_util >= 0.90 and waiting_backlog > 0)
                or (
                    running_util >= 0.75
                    and avg_group_fanout >= 2.0
                    and waiting_backlog > backlog_threshold
                )
                or (free_block_ratio <= 0.15 and waiting_backlog > 0)
            )

        return _TailBlendAdmissionPressure(
            active=active,
            running_util=running_util,
            waiting_backlog=waiting_backlog,
            uncovered_groups=uncovered_groups,
            min_uncovered_slack_s=min_uncovered_slack_s,
            avg_group_fanout=avg_group_fanout,
            free_block_ratio=free_block_ratio,
        )

    def _group_has_first_token(
        self,
        group_id: str,
        requests: list[Request],
    ) -> bool:
        state = self._groups.get(group_id)
        return bool(state is not None and state.finished_lengths) or any(
            request.num_output_tokens > 0 for request in requests
        )

    def _ttft_guard_group_key(
        self,
        group_id: str,
        requests: list[Request],
        now_s: float,
    ) -> tuple[bool, float, float, float, str]:
        has_first = self._group_has_first_token(group_id, requests)
        if has_first:
            deadline_s = self.scheduler._get_taper_plus_group_deadline_s(requests)
        else:
            deadline_s = (
                min(request.arrival_time for request in requests)
                + self.scheduler.taper_plus_ttft_slo_s
            )
        slack_s = deadline_s - now_s
        predicted_work = sum(
            self._predict_remaining_tokens(request) for request in requests
        )
        oldest_arrival = min(request.arrival_time for request in requests)
        return (
            has_first,
            slack_s,
            predicted_work,
            oldest_arrival,
            group_id,
        )

    def _ttft_guard_baseline_request_key(
        self,
        request: Request,
        now_s: float,
    ) -> tuple[float, bool, float, float, str]:
        branch_overdue_ms = self.scheduler._get_taper_plus_branch_overdue_ms(
            request,
            now_s,
        )
        return (
            -branch_overdue_ms,
            request.num_output_tokens > 0,
            self._predict_remaining_tokens(request),
            request.arrival_time,
            request.request_id,
        )

    def _should_defer_for_virtual_reservation(self, request: Request) -> bool:
        scheduler = self.scheduler
        reservation_q = scheduler.tail_blend_reservation_q
        if reservation_q <= 0.0:
            return False
        if not self._under_admission_pressure():
            return False

        free_blocks = scheduler.kv_cache_manager.get_remaining_free_blocks()
        if free_blocks <= 0:
            return True

        active_future_blocks = sum(
            self._estimate_reserved_future_blocks(running_request)
            for running_request in scheduler.running
            if running_request.sampling_params is not None
        )
        request_future_blocks = self._estimate_reserved_future_blocks(request)

        # Reserve against predicted future growth, not against the conservative
        # max_tokens bound. If the predicted aggregate still fits, preserve
        # stock FCFS admission. If it does not fit, do not stop admission
        # entirely; let the smallest predicted-footprint waiting branch go first.
        if active_future_blocks + request_future_blocks <= free_blocks:
            return False

        best_request_id = self._best_waiting_request_id_by_reservation()
        return best_request_id is not None and request.request_id != best_request_id

    def _best_waiting_request_id_by_reservation(self) -> str | None:
        candidates = [
            request
            for request in self.scheduler.waiting
            if request.external_req_id is not None
            and request.sampling_params is not None
        ]
        candidates.extend(
            request
            for request in self.scheduler.skipped_waiting
            if request.external_req_id is not None
            and request.sampling_params is not None
        )
        if not candidates:
            return None

        best_request = min(
            candidates,
            key=lambda request: (
                self._estimate_reserved_future_blocks(request),
                self._predict_output_tokens_for_reservation(request),
                request.arrival_time,
                request.request_id,
            ),
        )
        return best_request.request_id

    def _started_sibling_count(self, group_id: str) -> int:
        scheduler = self.scheduler
        running_count = sum(
            1
            for request in scheduler.running
            if scheduler._taper_plus_group_id(request) == group_id
        )
        finished_count = len(self._groups[group_id].finished_lengths)
        return running_count + finished_count

    def _best_expandable_group_id(self) -> str | None:
        scheduler = self.scheduler
        group_to_queued: dict[str, list[Request]] = defaultdict(list)
        for request in scheduler.waiting:
            if request.external_req_id is not None:
                group_to_queued[scheduler._taper_plus_group_id(request)].append(request)
        for request in scheduler.skipped_waiting:
            if request.external_req_id is not None:
                group_to_queued[scheduler._taper_plus_group_id(request)].append(request)

        expandable_group_ids = [
            group_id
            for group_id in group_to_queued
            if self._started_sibling_count(group_id) >= scheduler.tail_blend_pilot_k
            and self._has_finished_group_sibling(group_id)
        ]
        if not expandable_group_ids:
            return None

        now_s = time.time()
        return min(
            expandable_group_ids,
            key=lambda group_id: self._expand_rank_key(
                group_id, group_to_queued[group_id], now_s
            ),
        )

    def _expand_rank_key(
        self,
        group_id: str,
        queued_requests: list[Request],
        now_s: float,
    ) -> tuple[float, float, float, int, float, str]:
        scheduler = self.scheduler
        running_requests = [
            request
            for request in scheduler.running
            if scheduler._taper_plus_group_id(request) == group_id
        ]
        live_remaining = sum(
            self._predict_remaining_tokens(request) for request in running_requests
        )
        max_tokens = max(
            [request.max_tokens for request in queued_requests]
            + [request.max_tokens for request in running_requests],
            default=1,
        )
        queued_remaining = len(queued_requests) * self._predict_queued_tokens(
            group_id, max_tokens
        )
        predicted_parent_work = live_remaining + queued_remaining

        if running_requests:
            deadline_s = scheduler._get_taper_plus_group_deadline_s(running_requests)
            oldest_arrival = min(request.arrival_time for request in running_requests)
        else:
            oldest_arrival = min(request.arrival_time for request in queued_requests)
            deadline_s = oldest_arrival + scheduler.taper_plus_ttft_slo_s

        slack_s = max(1e-3, deadline_s - now_s)
        urgency_cost = predicted_parent_work / slack_s
        finished_count = len(self._groups[group_id].finished_lengths)
        return (
            urgency_cost,
            predicted_parent_work,
            -slack_s,
            -finished_count,
            oldest_arrival,
            group_id,
        )

    def _predict_queued_tokens(self, group_id: str, max_tokens: int) -> float:
        state = self._groups.get(group_id)
        if state is None or not state.finished_lengths:
            return float(max_tokens)

        num_finished = len(state.finished_lengths)
        sibling_mean = sum(state.finished_lengths) / max(1, num_finished)
        weight = num_finished / (num_finished + self._SHRINK_K)
        prior_len = self._QUEUED_PRIOR_FRACTION * float(max(1, max_tokens))
        predicted_len = (1.0 - weight) * prior_len + weight * sibling_mean
        return max(1.0, min(float(max_tokens), predicted_len))

    def _estimate_reserved_future_blocks(self, request: Request) -> int:
        target_output_tokens = int(
            ceil(self._predict_output_tokens_for_reservation(request))
        )
        return self.scheduler.kv_cache_manager.estimate_future_footprint_for_output_len(
            request,
            target_output_tokens,
        )

    def _predict_output_tokens_for_reservation(self, request: Request) -> float:
        reservation_q = self.scheduler.tail_blend_reservation_q
        if reservation_q <= 0.0:
            return float(request.max_tokens)

        max_tokens = max(1, request.max_tokens)
        prior_len = reservation_q * float(max_tokens)
        state = self._groups.get(request.external_req_id)
        running_lengths = [
            running_request.num_output_tokens
            for running_request in self.scheduler.running
            if running_request.external_req_id == request.external_req_id
        ]
        if state is None or not state.finished_lengths:
            sample_q_len = self._percentile(running_lengths, reservation_q)
            predicted_len = max(prior_len, sample_q_len)
        else:
            samples = state.finished_lengths + running_lengths
            sample_q_len = self._percentile(samples, reservation_q)
            num_finished = len(state.finished_lengths)
            weight = num_finished / (num_finished + self._SHRINK_K)
            predicted_len = (1.0 - weight) * prior_len + weight * sample_q_len

        predicted_len = max(float(request.num_output_tokens), predicted_len)
        return min(float(max_tokens), predicted_len)

    @staticmethod
    def _percentile(values: list[int], q: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        q = max(0.0, min(1.0, q))
        index = q * (len(values) - 1)
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        frac = index - lower
        return values[lower] * (1.0 - frac) + values[upper] * frac

    @staticmethod
    def _percentile_float(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        q = max(0.0, min(1.0, q))
        index = q * (len(values) - 1)
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        frac = index - lower
        return values[lower] * (1.0 - frac) + values[upper] * frac

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _predict_remaining_tokens(self, request: Request) -> float:
        state = self._groups.get(request.external_req_id)
        if state is None or not state.finished_lengths:
            return float(max(0, request.max_tokens - request.num_output_tokens))

        finished_lengths = state.finished_lengths
        num_finished = len(finished_lengths)

        sibling_mean = sum(finished_lengths) / max(1, num_finished)
        weight = num_finished / (num_finished + self._SHRINK_K)

        lower_bound = float(request.num_output_tokens)
        prior_len = float(max(1, request.max_tokens))
        tail_mean = 0.5 * (lower_bound + prior_len)
        predicted_final_len = (1.0 - weight) * tail_mean + weight * sibling_mean
        predicted_final_len = max(lower_bound, predicted_final_len)

        return max(0.0, predicted_final_len - float(request.num_output_tokens))

    def _has_finished_sibling(self, request: Request) -> bool:
        state = self._groups.get(request.external_req_id)
        return bool(state is not None and state.finished_lengths)

    def _has_finished_group_sibling(self, group_id: str) -> bool:
        state = self._groups.get(group_id)
        return bool(state is not None and state.finished_lengths)
