# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backward-compatible import shim for the renamed TailBlend controller."""

from vllm.v1.core.sched.tail_blend_controller import TailBlendController

EagerTailBlendController = TailBlendController
