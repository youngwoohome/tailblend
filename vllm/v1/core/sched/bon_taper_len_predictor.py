# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-overhead live-branch length hints for BoN-TAPER.

The scheduler must not pay a logits/logprob tax to make a scheduling decision.
This predictor only uses values already tracked by the scheduler: current
output length, max_tokens, live sibling lengths, and finished sibling lengths.
It provides a planning-only output length estimate; generation ``max_tokens``
is not changed.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass
class _GroupState:
    finished_lengths: list[int] = field(default_factory=list)


class BoNTaperLenPredictor:
    """Maintains cheap, scheduler-visible length hints.

    This is intentionally a heuristic, not a learned predictor. The aim is to
    estimate the future KV footprint of opportunistic BoN siblings without
    assuming every branch will consume ``max_tokens``.
    """

    def __init__(self, scheduler: "Scheduler") -> None:
        self.scheduler = scheduler
        self._groups: dict[str, _GroupState] = defaultdict(_GroupState)
        self._request_to_group: dict[str, str] = {}

    def prepare_request(self, request: Request) -> None:
        """Record the parent group; do not request any extra model outputs."""
        if (
            not self.scheduler.enable_taper_plus
            or self.scheduler.taper_plus_policy != "bon_taper_len_predictor"
        ):
            return
        group_id = self.scheduler._taper_plus_group_id(request)
        self._request_to_group[request.request_id] = group_id

    def observe_request_output(
        self,
        request: Request,
        new_token_ids: list[int],
        logprobs: object | None = None,
    ) -> None:
        if (
            self.scheduler.taper_plus_policy != "bon_taper_len_predictor"
            or not new_token_ids
        ):
            return
        self.refresh_request(request)

    def observe_finished_request(self, request: Request) -> None:
        group_id = self._request_to_group.pop(
            request.request_id,
            self.scheduler._taper_plus_group_id(request),
        )
        if request.num_output_tokens > 0:
            self._groups[group_id].finished_lengths.append(request.num_output_tokens)

    def refresh_running_requests(self) -> None:
        if self.scheduler.taper_plus_policy != "bon_taper_len_predictor":
            return

        running_by_group: dict[str, list[Request]] = defaultdict(list)
        for request in self.scheduler.running:
            if self.scheduler._is_taper_plus_decode_candidate(request):
                group_id = self.scheduler._taper_plus_group_id(request)
                running_by_group[group_id].append(request)
                self._request_to_group[request.request_id] = group_id

        for group_id, requests in running_by_group.items():
            live_lengths = [request.num_output_tokens for request in requests]
            for request in requests:
                request.prob_short = max(
                    request.prob_short,
                    self.predict_short_probability(
                        request,
                        live_lengths=live_lengths,
                        finished_lengths=self._groups[
                            group_id
                        ].finished_lengths,
                    ),
                )

        active_groups = set(running_by_group)
        for group_id in list(self._groups):
            if (
                group_id not in active_groups
                and not any(value == group_id for value in self._request_to_group.values())
            ):
                self._groups.pop(group_id, None)

    def refresh_request(self, request: Request) -> None:
        group_id = self.scheduler._taper_plus_group_id(request)
        self._request_to_group[request.request_id] = group_id
        live_lengths = [
            live.num_output_tokens
            for live in self.scheduler.running
            if self.scheduler._taper_plus_group_id(live) == group_id
            and self.scheduler._is_taper_plus_decode_candidate(live)
        ]
        request.prob_short = max(
            request.prob_short,
            self.predict_short_probability(
                request,
                live_lengths=live_lengths,
                finished_lengths=self._groups[group_id].finished_lengths,
            ),
        )

    def predict_short_probability(
        self,
        request: Request,
        *,
        live_lengths: list[int],
        finished_lengths: list[int],
    ) -> float:
        output_len = request.num_output_tokens
        max_tokens = max(1, request.max_tokens)
        progress = min(1.0, output_len / max_tokens)

        if output_len <= 0:
            return progress

        if not finished_lengths:
            # With no completed sibling, the only free signal is progress.
            return progress

        finished_mean = statistics.fmean(finished_lengths)
        finished_std = (
            statistics.pstdev(finished_lengths)
            if len(finished_lengths) > 1
            else finished_mean * 0.25
        )
        sorted_finished = sorted(finished_lengths)
        tail_count = max(1, math.ceil(len(sorted_finished) * 0.5))
        tail_mean = statistics.fmean(sorted_finished[-tail_count:])

        # Be conservative: for scheduling, underpredicting remaining lifetime is
        # worse than overpredicting it because it can flood the decode batch with
        # long-tail branches. A high quantile proxy is a safer completion target.
        predicted_total = min(
            max_tokens,
            max(tail_mean, finished_mean + finished_std),
        )
        closeness = _sigmoid((output_len - predicted_total + 128.0) / 128.0)

        live_rank = 0.5
        if live_lengths:
            live_mean = statistics.fmean(live_lengths)
            live_rank = _sigmoid((output_len - live_mean) / 128.0)

        completed_ratio = len(finished_lengths) / max(
            1, len(finished_lengths) + len(live_lengths)
        )

        # Blend sibling-relative completion with hard max-token progress. The
        # result is only a ranking hint; the controller still enforces step and
        # memory budgets.
        return _clamp01(
            0.45 * closeness
            + 0.30 * progress
            + 0.15 * live_rank
            + 0.10 * completed_ratio
        )

    def predict_total_output_tokens(self, request: Request) -> int | None:
        """Return a conservative planning length for this branch.

        The returned value is an output-token count, not total prompt+output
        tokens. ``None`` means the caller should fall back to ``max_tokens``.
        """
        group_id = self._request_to_group.get(
            request.request_id,
            self.scheduler._taper_plus_group_id(request),
        )
        finished_lengths = self._groups[group_id].finished_lengths
        if not finished_lengths:
            return None

        current_output = request.num_output_tokens
        max_tokens = max(1, request.max_tokens)
        if current_output >= max_tokens:
            return max_tokens

        predicted = self._predict_from_finished_lengths(
            finished_lengths,
            max_tokens=max_tokens,
        )
        # Do not let a short sibling predict that a live branch is already done.
        # The margin is a planning safety buffer, not a generation cutoff.
        min_target = current_output + min(256, max_tokens - current_output)
        return min(max_tokens, max(predicted, min_target))

    @staticmethod
    def _predict_from_finished_lengths(
        finished_lengths: list[int],
        *,
        max_tokens: int,
    ) -> int:
        sorted_lengths = sorted(finished_lengths)
        count = len(sorted_lengths)
        if count == 1:
            base = sorted_lengths[0]
            margin = max(256, math.ceil(base * 0.25))
            return min(max_tokens, base + margin)

        # Use an upper-tail sibling statistic because underestimating future KV
        # footprint is worse than losing some potential concurrency.
        quantile_index = min(count - 1, math.ceil(0.75 * count) - 1)
        p75 = sorted_lengths[quantile_index]
        mean = statistics.fmean(sorted_lengths)
        std = statistics.pstdev(sorted_lengths)
        margin = max(128, math.ceil(0.25 * std))
        return min(max_tokens, math.ceil(max(p75, mean + 0.5 * std) + margin))
