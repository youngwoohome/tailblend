# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BoN-aware TAPER controllers.

This module keeps the paper-reproduction TAPER path in ``scheduler.py` intact
and defines a second controller for the BoN research setting. The abstraction is
parent-level: one branch per BoN parent is mandatory progress, while sibling
branches are opportunistic exploration admitted by marginal utility per
predicted marginal step cost.
"""

import time
from collections import defaultdict
from typing import TYPE_CHECKING

from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler


class BoNTaperController:
    """Parent-level BoN width controller.

    ``bon_taper`` uses only parent deadlines, slack, and utility. ``bon_taper_plus``
    extends the same objective with branch lifetime and KV footprint signals.
    """

    def __init__(self, scheduler: "Scheduler") -> None:
        self.scheduler = scheduler
        self.len_predictor_protected_width = 4

    def should_defer_waiting_request(self, request: Request) -> bool:
        """Throttle new BoN parents before KV pressure cascades.

        Under heavy BoN pressure, spreading capacity thinly across every parent
        improves first-token latency but makes almost every parent miss TPOT.
        Instead, limit how many logical parents are active at once; once a
        parent is admitted, its siblings can join and complete as a cohort.
        """
        scheduler = self.scheduler
        if (
            not scheduler.enable_taper_plus
            or scheduler.taper_plus_policy
            not in ("bon_taper", "bon_taper_plus", "bon_taper_len_predictor")
            or not request.external_req_id
        ):
            return False

        memory_threshold = scheduler._get_taper_plus_memory_threshold()
        free_blocks = scheduler.kv_cache_manager.get_remaining_free_blocks()
        waiting_backlog = len(scheduler.waiting) + len(scheduler.skipped_waiting)
        if (
            free_blocks >= memory_threshold * 4
            and waiting_backlog <= scheduler.max_num_running_reqs
        ):
            return False

        group_id = scheduler._taper_plus_group_id(request)
        active_parent_ids = {
            scheduler._taper_plus_group_id(running_request)
            for running_request in scheduler.running
            if running_request.external_req_id
        }
        if group_id in active_parent_ids:
            return False

        sibling_count = sum(
            1
            for queued_request in scheduler.waiting
            if scheduler._taper_plus_group_id(queued_request) == group_id
        )
        sibling_count += sum(
            1
            for queued_request in scheduler.skipped_waiting
            if scheduler._taper_plus_group_id(queued_request) == group_id
        )
        sibling_count = max(1, sibling_count)

        blocks_per_sequence = max(
            1,
            (scheduler.max_model_len + scheduler.block_size - 1)
            // scheduler.block_size,
        )
        sequence_capacity = max(
            1,
            scheduler.kv_cache_manager.block_pool.num_gpu_blocks
            // blocks_per_sequence,
        )
        parent_cap = max(1, min(8, (sequence_capacity * 4) // sibling_count))
        return len(active_parent_ids) >= parent_cap

    def plan_admitted_req_ids(self, token_budget: int) -> set[str] | None:
        scheduler = self.scheduler
        if not scheduler.enable_taper_plus or token_budget <= 0:
            return None
        if scheduler.taper_plus_policy == "bon_taper":
            return None
        if scheduler.taper_plus_policy == "bon_taper_len_predictor":
            scheduler.bon_taper_len_predictor.refresh_running_requests()

        use_memory_lifetime = scheduler.taper_plus_policy in (
            "bon_taper_plus",
            "bon_taper_len_predictor",
        )
        hard_short_lifetime_gate = scheduler.taper_plus_policy == "bon_taper_plus"
        group_to_candidates: dict[str, list[Request]] = defaultdict(list)
        for request in scheduler.running:
            if scheduler._is_taper_plus_decode_candidate(request):
                group_to_candidates[
                    scheduler._taper_plus_group_id(request)
                ].append(request)

        regulated_groups = {
            group_id
            for group_id, requests in group_to_candidates.items()
            if len(requests) > 1
        }
        if not regulated_groups:
            return None

        memory_threshold = scheduler._get_taper_plus_memory_threshold()
        free_blocks = scheduler.kv_cache_manager.get_remaining_free_blocks()
        if free_blocks >= memory_threshold:
            return None

        admitted_req_ids: set[str] = set()
        deferred_by_group: dict[str, list[Request]] = defaultdict(list)
        admitted_width_by_group: dict[str, int] = defaultdict(int)
        active_decode_width = 0
        active_context_tokens = 0

        for request in scheduler.running:
            group_id = scheduler._taper_plus_group_id(request)
            is_decode_candidate = scheduler._is_taper_plus_decode_candidate(request)

            if group_id not in regulated_groups or not is_decode_candidate:
                admitted_req_ids.add(request.request_id)
                if is_decode_candidate:
                    active_decode_width += 1
                    active_context_tokens += request.num_computed_tokens
                continue

            if admitted_width_by_group[group_id] == 0:
                # Mandatory BoN parent progress: each active logical parent
                # receives one branch in the protected composition.
                admitted_req_ids.add(request.request_id)
                admitted_width_by_group[group_id] = 1
                active_decode_width += 1
                active_context_tokens += request.num_computed_tokens
            else:
                deferred_by_group[group_id].append(request)

        if not deferred_by_group:
            return admitted_req_ids

        if scheduler.taper_plus_step_target_ms <= 0.0:
            return admitted_req_ids

        now_s = time.time()
        parent_slack_ms = {
            group_id: max(
                0.0,
                scheduler._get_taper_plus_group_deadline_s(requests) - now_s,
            )
            * 1000.0
            for group_id, requests in group_to_candidates.items()
        }
        if not parent_slack_ms:
            return admitted_req_ids

        baseline_step_ms = scheduler._predict_taper_plus_step_ms(
            active_decode_width, active_context_tokens
        )
        min_slack_ms = min(parent_slack_ms.values())

        available_extra_slots = max(0, token_budget - active_decode_width)
        if available_extra_slots == 0:
            return admitted_req_ids

        virtual_free_blocks = scheduler.kv_cache_manager.get_remaining_free_blocks()
        eps = 1e-6
        global_budget_ms = baseline_step_ms + scheduler.taper_plus_rho * max(
            0.0, min_slack_ms - baseline_step_ms
        )

        while available_extra_slots > 0:
            best_group_id: str | None = None
            best_request: Request | None = None
            best_score = 0.0
            best_predicted_step_ms = 0.0
            best_future_footprint = 0

            for group_id, deferred in deferred_by_group.items():
                if not deferred:
                    continue

                for request in deferred:
                    predicted_step_ms = scheduler._predict_taper_plus_step_ms(
                        active_decode_width + 1,
                        active_context_tokens + request.num_computed_tokens,
                    )
                    group_budget_ms = baseline_step_ms + scheduler.taper_plus_rho * max(
                        0.0,
                        parent_slack_ms.get(group_id, min_slack_ms)
                        - baseline_step_ms,
                    )
                    candidate_budget_ms = max(global_budget_ms, group_budget_ms)
                    overdue_ms = 0.0
                    if scheduler.taper_plus_branch_overdue_boost:
                        overdue_ms = scheduler._get_taper_plus_branch_overdue_ms(
                            request, now_s
                        )
                    if overdue_ms > 0.0:
                        # A parent-level deadline can be refreshed by one
                        # admitted branch while sibling branches starve. Let
                        # overdue siblings pass feasibility and compete by
                        # utility instead of being hidden by global min slack.
                        candidate_budget_ms = max(
                            candidate_budget_ms, predicted_step_ms
                        )
                    if predicted_step_ms > candidate_budget_ms:
                        continue

                    future_footprint = 0
                    if use_memory_lifetime:
                        if (
                            scheduler.taper_plus_policy
                            == "bon_taper_len_predictor"
                            and admitted_width_by_group[group_id]
                            >= self.len_predictor_protected_width
                        ):
                            predicted_total_output = (
                                scheduler.bon_taper_len_predictor
                                .predict_total_output_tokens(request)
                            )
                            if predicted_total_output is None:
                                future_footprint = (
                                    scheduler.kv_cache_manager
                                    .estimate_future_footprint(request)
                                )
                            else:
                                future_footprint = (
                                    scheduler.kv_cache_manager
                                    .estimate_future_footprint_for_output_len(
                                        request,
                                        predicted_total_output,
                                    )
                                )
                        else:
                            future_footprint = (
                                scheduler.kv_cache_manager
                                .estimate_future_footprint(request)
                            )
                        under_memory_pressure = (
                            virtual_free_blocks < memory_threshold
                        )
                        over_future_budget = virtual_free_blocks < future_footprint
                        if (
                            hard_short_lifetime_gate
                            and (under_memory_pressure or over_future_budget)
                            and not request.is_short_lived()
                        ):
                            continue

                    marginal_utility = self._marginal_parent_utility(
                        admitted_width_by_group[group_id]
                    )
                    if overdue_ms > 0.0:
                        marginal_utility *= 1.0 + overdue_ms / max(
                            1.0, scheduler.taper_plus_tpot_slo_s * 1000.0
                        )

                    if use_memory_lifetime:
                        marginal_utility *= 1.0 + max(0.0, request.prob_short)
                        if future_footprint > 0:
                            memory_cost = future_footprint / max(
                                1, virtual_free_blocks
                            )
                            marginal_utility /= 1.0 + memory_cost

                    marginal_step_ms = max(0.0, predicted_step_ms - baseline_step_ms)
                    score = marginal_utility / (eps + marginal_step_ms)
                    if self._is_better_candidate(
                        score=score,
                        best_score=best_score,
                        request=request,
                        best_request=best_request,
                        use_memory_lifetime=use_memory_lifetime,
                    ):
                        best_group_id = group_id
                        best_request = request
                        best_score = score
                        best_predicted_step_ms = predicted_step_ms
                        best_future_footprint = future_footprint

            if best_request is None or best_group_id is None or best_score <= 0.0:
                break

            admitted_req_ids.add(best_request.request_id)
            active_decode_width += 1
            active_context_tokens += best_request.num_computed_tokens
            baseline_step_ms = best_predicted_step_ms
            admitted_width_by_group[best_group_id] += 1
            deferred_by_group[best_group_id].remove(best_request)
            available_extra_slots -= 1

            if use_memory_lifetime:
                virtual_free_blocks = max(
                    0, virtual_free_blocks - best_future_footprint
                )

        return admitted_req_ids

    def _marginal_parent_utility(self, admitted_width: int) -> float:
        scheduler = self.scheduler
        if scheduler.taper_plus_utility == "concave":
            return (admitted_width + 1) ** 0.5 - admitted_width**0.5
        return 1.0

    @staticmethod
    def _is_better_candidate(
        *,
        score: float,
        best_score: float,
        request: Request,
        best_request: Request | None,
        use_memory_lifetime: bool,
    ) -> bool:
        if best_request is None or score > best_score:
            return True
        if score < best_score:
            return False
        if use_memory_lifetime and request.prob_short != best_request.prob_short:
            return request.prob_short > best_request.prob_short
        return (request.arrival_time, request.request_id) < (
            best_request.arrival_time,
            best_request.request_id,
        )
