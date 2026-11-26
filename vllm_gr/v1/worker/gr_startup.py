# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Prepare the persistent stage execution view for shared Beam Forward capture."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger

from vllm_gr.config.execution import validate_gr_execution_runtime
from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchParams,
    BeamOutputOptions,
)
from vllm_gr.v1.beam.dispatch import (
    GR_STAGE_OUTPUT_ATTR,
    GRDeviceStateRef,
    GRExecutionCapacity,
    GRStageKind,
    GRStageMetadata,
)
from vllm_gr.v1.engine.beam_admission import validate_gr_device_config

if TYPE_CHECKING:
    from vllm_gr.v1.worker.cuda.gpu_stage_hooks import CudaStageHooksType

logger = init_logger(__name__)


def capture_async_forward(
    worker: Any,
    *,
    hooks_type: CudaStageHooksType | None = None,
    allow_eager_fallback: bool = False,
) -> bool:
    with torch.inference_mode():
        if not _capture_enabled(worker):
            return False
        device_type = str(worker.device).split(":")[0]
        if device_type == "npu":
            from vllm.config import set_current_vllm_config

            from vllm_gr.v1.worker.npu.npu_model_runner_patch import (
                capture_beam_decode_graph,
            )

            # Ascend's native capture can restore its config context before
            # GR starts capturing. Bind this worker's config for mode selection
            # and the dummy forwards so they use the same Decode backend.
            with set_current_vllm_config(worker.vllm_config):
                captured: bool = capture_beam_decode_graph(worker)
            return captured
        if device_type != "cuda":
            return False
        if hooks_type is None:
            raise ValueError("CUDA GR startup capture requires explicit stage hooks")
        from vllm_gr.v1.worker.cuda.gpu_stage_hooks import validate_stage_hooks_type
        from vllm_gr.v1.worker.cuda.v2.gpu_beam_stage_runner_v2 import CudaStageHooksV2

        if hooks_type is CudaStageHooksV2:
            from vllm_gr.v1.worker.cuda.v2.gpu_model_runner_v2_resources import (
                capture_multibatch_forward,
            )

            return capture_multibatch_forward(worker, allow_eager_fallback=allow_eager_fallback)
        # Measure (or load) the device-local prefix cost model before any Beam graph
        # is captured: the cache write is a YAML file and the probe needs the GPU to
        # itself. Failures degrade to "no calibration", never to a failed start-up.
        from vllm_gr.v1.attention.backends.beam_prefix_calibrate import (
            ensure_prefix_calibration,
        )
        from vllm_gr.v1.worker.gr_context import _get_gr_config

        cfg = _get_gr_config(worker)
        if getattr(cfg, "beam_graph_widths", None) is not None:
            logger.warning(
                "beam.graph_widths is NPU-only startup pre-capture; CUDA captures "
                "dynamic Decode (batch, width) shapes lazily on first use."
            )

        max_reqs = worker.vllm_config.scheduler_config.max_num_seqs
        batches = tuple(b for b in cfg.deployment.beam_batch_sizes if b <= max_reqs)
        if not batches:
            return False
        if hasattr(worker, "_gr_decode_capture_failures"):
            owner = getattr(worker, "_gr_gpu_stage_runner", None)
            if owner is None or not owner.closed:
                raise RuntimeError(
                    "Asynchronous scheduling startup encountered an incomplete capture state"
                )
            validate_stage_hooks_type(owner.hooks, hooks_type)
            graph = getattr(worker, "_beam_decode_graph", None)
            for batch in batches:
                if batch not in worker._gr_decode_capture_failures and (
                    graph is None or not graph.select(batch, cfg.beam_max_width).ready
                ):
                    raise RuntimeError("Published Decode graph is not ready")
            return len(worker._gr_decode_capture_failures) < len(batches)
        if getattr(worker, "_gr_gpu_stage_runner", None) is not None:
            raise RuntimeError(
                "Asynchronous scheduling startup encountered an incomplete capture state"
            )
        ensure_prefix_calibration(worker)
        failures = {}
        for batch in batches:
            error = _capture_batch(
                worker,
                batch,
                hooks_type=hooks_type,
                allow_eager_fallback=allow_eager_fallback,
            )
            if error is not None:
                failures[batch] = error
        worker._gr_decode_capture_failures = failures
        worker._gr_async_decode_graph_capture_failed = len(failures) == len(batches)
        worker._gr_async_decode_graph_capture_error = next(iter(failures.values()), None)
        return len(failures) < len(batches)


def _capture_enabled(worker: Any) -> bool:
    from vllm_gr.config.models import ConstraintMode, configured_constraint_mode
    from vllm_gr.v1.worker.gr_context import _get_gr_config

    cfg = _get_gr_config(worker)
    validate_gr_execution_runtime(worker.vllm_config)
    mode = configured_constraint_mode(cfg)
    if not cfg.beam_worker_decision or cfg.beam_max_width is None or mode is None:
        raise ValueError(
            "Beam V1 requires worker decisions, fixed width and either an enabled "
            "canonical table or gr.constraint_mode=none"
        )
    if mode is ConstraintMode.NONE and str(worker.device).split(":")[0] not in (
        "cuda",
        "musa",
        "npu",
    ):
        raise ValueError("unconstrained Beam V1 requires CUDA, MUSA, or NPU")
    if mode is ConstraintMode.NONE and str(worker.device).split(":")[0] == "npu":
        validate_gr_device_config(worker.vllm_config)
    # A one-token session completes in Prefill and never replays a Decode graph.
    return (
        not worker.model_config.enforce_eager
        and cfg.beam_graph_enabled
        and cfg.beam_max_decode_steps > 1
    )


def _capture_batch(
    worker: Any,
    batch_size: int,
    *,
    hooks_type: CudaStageHooksType,
    allow_eager_fallback: bool,
) -> str | None:
    """Capture the explicit batch selected by the V1 or V2 entry point."""
    from vllm_gr.v1.worker.cuda.beam_decode_graph import BeamDecodeCaptureUnsupported
    from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import (
        get_stage_runner,
    )
    from vllm_gr.v1.worker.gr_context import (
        _get_gr_config,
    )

    cfg = _get_gr_config(worker)
    width = cfg.beam_max_width
    GRExecutionCapacity(width)
    sessions = getattr(worker, "_beam_sessions", {})
    if sessions:
        raise RuntimeError("Asynchronous scheduling startup capture requires drained Beam sessions")
    worker._beam_sessions = sessions
    name = "__async_scheduling_startup_capture__"
    request_ids = (
        (name,) if batch_size == 1 else tuple(f"{name}_{index}" for index in range(batch_size))
    )
    m = GRStageMetadata(
        session_id=name,
        dispatch_id=0,
        stage_index=0,
        kind=GRStageKind.PREFILL,
        decode_step=None,
        is_last_stage=False,
        input_device_state=None,
        output_device_state=GRDeviceStateRef(name, 0),
        # The capture includes Decode step 0 after the first Prefill token.
        beam_params=BeamBatchParams(width, 2, 0.0, (), True),
        output_options=BeamOutputOptions(width, True, True, True),
        num_prompt_tokens=batch_size,
        request_ids=request_ids,
        num_prompt_tokens_by_request=(1,) * batch_size,
    )
    scheduled = SimpleNamespace(num_scheduled_tokens=dict.fromkeys(request_ids, 1))
    setattr(scheduled, GR_STAGE_OUTPUT_ATTR, {name: m})
    owner = get_stage_runner(worker, scheduled, hooks_type=hooks_type, for_startup_capture=True)
    owner.previous = m
    # The null block is a safe capture prefix. Real requests replace its
    # contents and actual length while the device table address stays fixed.
    # Both runner surfaces consume the same published KV metadata, without
    # creating a fake entry in native request state.
    graph = None
    fallback_error = None
    try:
        try:
            decode = replace(
                m,
                kind=GRStageKind.DECODE,
                stage_index=1,
                decode_step=0,
                is_last_stage=True,
                input_device_state=GRDeviceStateRef(name, 0),
                output_device_state=GRDeviceStateRef(name, 1),
                kv_blocks=((0,),) * batch_size,
            )
            owner.prepared = owner.prepare_cpu(decode, 0)
            owner._bind_attention(decode)
            # Keep the startup fallback graph at model capacity. Measured
            # length buckets are captured lazily against real request buffers.
            if getattr(getattr(owner, "metadata", None), "prefix_length_bucket", 0):
                assert owner.inputs is not None
                owner.metadata.prefix_length_bucket = 0
                owner.metadata.prefix_max_seq_len = (
                    (owner.inputs.max_model_len + owner.inputs.block_size - 1)
                    // owner.inputs.block_size
                    * owner.inputs.block_size
                )
            owner.input_ids.zero_()
            owner.positions.zero_()
            owner.execution_mask.zero_()
            owner.beam.session.context.decode_step_buf.fill_(-1)
            try:
                graph = owner.capture_forward()
            except BeamDecodeCaptureUnsupported as error:
                if not allow_eager_fallback:
                    raise
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Decode capture did not leave capture mode") from error
                # Failed capture ran on a side stream. Drain it before ending
                # the dummy session; a device error here aborts startup.
                torch.accelerator.synchronize(worker.device)
                fallback_error = error
        finally:
            done = torch.Event()
            done.record()
            owner.release(owner.binding, consumer_done=done)
        if not owner.closed:
            raise RuntimeError("Decode startup resources were not released")
        torch.cuda.current_stream(worker.device).synchronize()
        if graph is not None and not graph.ready:
            raise RuntimeError("Decode capture completed without a ready graph")
    except Exception:
        if graph is not None:
            graph.clear()
        raise

    if fallback_error is not None:
        logger.warning(
            "GR Decode capture unsupported; cleanup and device checks succeeded. "
            "Using eager Decode for B=%d: %s",
            batch_size,
            fallback_error,
        )
        return str(fallback_error)
    if graph is None:
        raise RuntimeError("Decode capture completed without a graph candidate")
    # Publishing is the final step: no serving path can see this graph before
    # the startup owner has been released and its GPU work has completed.
    owner.graph = graph
    return None
