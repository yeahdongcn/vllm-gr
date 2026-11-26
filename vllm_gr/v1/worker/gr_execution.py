# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Device routing and input ownership for native GR dispatches."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any

from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

if TYPE_CHECKING:
    from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import GPUBeamStageRunner
    from vllm_gr.v1.worker.cuda.gpu_stage_hooks import CudaStageHooksType
    from vllm_gr.v1.worker.npu.npu_beam_stage_runner import NPUEagerBeamStageRunner


def _copy_record(value: Any) -> Any:
    # These native dataclasses store their fields (including GR metadata) in
    # __dict__. Preserve the complete shallow state without copy's reducer.
    if type(value) in (SchedulerOutput, NewRequestData, CachedRequestData):
        local = object.__new__(type(value))
        local.__dict__ = value.__dict__.copy()
        return local
    return copy(value)


def worker_input(scheduled: Any) -> Any:
    """Detach mutable fields retained by native Worker request state.

    Tensors, immutable GR metadata and ordinary cached-step data stay shared.
    Native resumed requests retain their replacement block lists as well.
    """
    local = _copy_record(scheduled)
    local.num_scheduled_tokens = dict(scheduled.num_scheduled_tokens)
    local.finished_req_ids = set(scheduled.finished_req_ids)
    local.scheduled_new_reqs = []
    for request in scheduled.scheduled_new_reqs:
        owned = _copy_record(request)
        owned.block_ids = tuple(list(group) for group in request.block_ids)
        owned.prompt_token_ids = request.prompt_token_ids.copy()
        prefill = getattr(request, "prefill_token_ids", None)
        if prefill is not None:
            owned.prefill_token_ids = prefill.copy()
        # Native SamplingParams retains flat scalar containers and optional
        # nested parameters. Detach both; remaining fields are immutable scalars.
        # clone() is shallow when skip_clone=True. A shared memo preserves
        # aliases between fields and any values in its cached-property dict.
        params = request.sampling_params
        owned.sampling_params = copy(params)
        if params is not None:
            memo = {id(params): owned.sampling_params}
            for name in (
                "stop",
                "stop_token_ids",
                "logprob_token_ids",
                "_all_stop_token_ids",
                "logit_bias",
                "allowed_token_ids",
                "bad_words",
            ):
                value = getattr(params, name)
                if value is not None:
                    key = id(value)
                    if key not in memo:
                        memo[key] = copy(value)
                    setattr(owned.sampling_params, name, memo[key])
            for name in (
                "structured_outputs",
                "extra_args",
                "_bad_words_token_ids",
                "repetition_detection",
            ):
                value = getattr(params, name)
                if value is not None:
                    setattr(owned.sampling_params, name, deepcopy(value, memo))
            owned.sampling_params.__dict__ = deepcopy(params.__dict__, memo)
        local.scheduled_new_reqs.append(owned)
    cached = scheduled.scheduled_cached_reqs
    if cached.resumed_req_ids:
        local.scheduled_cached_reqs = _copy_record(cached)
        local.scheduled_cached_reqs.new_block_ids = [
            tuple(list(group) for group in blocks)
            if req_id in cached.resumed_req_ids and blocks is not None
            else blocks
            for req_id, blocks in zip(cached.req_ids, cached.new_block_ids)
        ]
    return local


def execute_gr_stage(
    worker: Any,
    scheduled: Any,
    original: Callable[..., Any],
    *args: Any,
    hooks_type: CudaStageHooksType | None = None,
    **kwargs: Any,
) -> Any:
    local = worker_input(scheduled)
    device_type = str(worker.device).split(":")[0]
    stage_runner: GPUBeamStageRunner | NPUEagerBeamStageRunner
    if device_type in ("cuda", "musa"):
        from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import (
            get_stage_runner,
        )

        if hooks_type is None:
            raise ValueError("GPU GR execution requires explicit stage hooks")
        stage_runner = get_stage_runner(
            worker,
            local,
            hooks_type=hooks_type,
        )
    elif device_type == "npu":
        from vllm_gr.v1.worker.npu.npu_beam_stage_runner import get_npu_stage_runner

        stage_runner = get_npu_stage_runner(worker, local)
    else:
        raise ValueError("beam_search has no execution adapter for this device")
    try:
        return stage_runner.execute(local, original, *args, **kwargs)
    except Exception as error:
        # Native BatchQueue consumes sample first. Stale logits must never hide
        # an execute failure from the corresponding sample Future.
        stage_runner.execution_error = error
        raise


def execute_gr_retirement(worker: Any, scheduled: Any) -> Any:
    local = worker_input(scheduled)
    device_type = str(worker.device).split(":")[0]
    if device_type in ("cuda", "musa"):
        from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import (
            execute_gr_retirement as retire,
        )
    elif device_type == "npu":
        from vllm_gr.v1.worker.npu.npu_beam_stage_runner import (
            execute_npu_gr_retirement,
        )

        return execute_npu_gr_retirement(worker, local)
    else:
        raise ValueError("beam_search has no retirement adapter for this device")

    return retire(worker, local)


def _dispatch_retirement_frame(self: Any, scheduler_output: Any) -> Any:
    """Handle a GR retirement control frame; returns the retire output.

    Returns None when ``scheduler_output`` is not a retirement frame (the
    caller continues with the normal dispatch).
    """
    from vllm_gr.v1.beam.dispatch import GR_RETIRE_OUTPUT_ATTR

    if getattr(scheduler_output, GR_RETIRE_OUTPUT_ATTR, None) is None:
        return None
    return execute_gr_retirement(self, scheduler_output)


def _dispatch_stage_frame(
    self: Any,
    scheduler_output: Any,
    orig_execute_model: Callable[..., Any],
    *args: Any,
    hooks_type: CudaStageHooksType | None = None,
    **kwargs: Any,
) -> Any:
    """Handle a persistent GR stage frame (beam_search execution).

    Returns ``_NOT_A_STAGE_FRAME`` when ``scheduler_output`` is not a stage
    frame; otherwise the stage runner's execute() result.  On an execution
    failure the error is recorded on the stage runner before re-raising —
    the native BatchQueue associates the execute and sample Futures, so
    sampling must also fail or valid-looking stale logits could hide a
    failed execute Future from that native path.
    """
    from vllm_gr.v1.beam.dispatch import GR_STAGE_OUTPUT_ATTR

    if not getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, None):
        return _NOT_A_STAGE_FRAME
    return execute_gr_stage(
        self, scheduler_output, orig_execute_model, *args, hooks_type=hooks_type, **kwargs
    )


# Sentinel distinguishing "not a stage frame" from a legitimate None result.
_NOT_A_STAGE_FRAME = object()
