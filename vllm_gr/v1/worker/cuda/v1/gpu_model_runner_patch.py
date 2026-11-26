# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""GPU ModelRunner wrappers for grouped Beam execution."""

from __future__ import annotations

import sys
from functools import wraps

import torch
from vllm.logger import init_logger

from vllm_gr.v1.beam.dispatch import GR_STAGE_OUTPUT_ATTR
from vllm_gr.v1.beam.execution_batch import (
    BeamBatchExecutionView,
    BeamExecutionContractError,
)
from vllm_gr.v1.worker.cuda.runner_installer import V1_RUNNER
from vllm_gr.v1.worker.cuda.v1.gpu_prefill_inputs import try_prepare_gr_prefill_inputs
from vllm_gr.v1.worker.gr_context import (
    _beam_full_graph_ready,
    _beam_graph_enabled_for_run,
    _create_beam_context,
    _get_gr_config,
)

# Shared context helpers and installation contracts have independent owners.
from vllm_gr.v1.worker.model_runner_contracts import (
    _WORKER_DETERMINE_PATCH_MARKER,
    _WORKER_MODEL_FORWARD_PATCH_MARKER,
    _WORKER_PREPARE_PATCH_MARKER,
    _WORKER_SAMPLE_TOKENS_PATCH_MARKER,
    is_worker_runner_patched,
)
from vllm_gr.v1.worker.prefill_graph_common import scheduled_requests_are_fresh
from vllm_gr.v1.worker.prefill_graph_runner import (
    SUPPORTED_NUM_REQS,
    get_prefill_graph_runner,
    is_prefill_graph_enabled,
    is_prefill_phase_batch,
)

logger = init_logger(__name__)


def make_patched_prepare_inputs_gpu(_orig_prepare_inputs):
    """Preserve native logits indices, spec metadata and maximum sampling width."""
    if getattr(_orig_prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, False):
        return _orig_prepare_inputs

    # vLLM-MUSA execute_model passes use_cached_decode_logits_indices to _prepare_inputs.
    @wraps(_orig_prepare_inputs)
    def patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens, **native_options):
        # Publish the scheduler output so the _model_forward prefill-graph
        # interception can classify the batch and extract token counts.
        self._current_scheduler_output = scheduler_output

        logits_indices = try_prepare_gr_prefill_inputs(self, scheduler_output, num_scheduled_tokens)
        if logits_indices is None:
            logits_indices, spec_decode_metadata, max_num_sampled_tokens = _orig_prepare_inputs(
                self, scheduler_output, num_scheduled_tokens, **native_options
            )
        else:
            spec_decode_metadata = None
            # The single-request Prefill fast path excludes speculative tokens.
            max_num_sampled_tokens = 1

        # GPU attention metadata is built after _prepare_inputs returns.
        if getattr(self, "_current_beam_info", None) is not None:
            self._current_beam_info["req_ids"] = self.input_batch.req_ids

        return logits_indices, spec_decode_metadata, max_num_sampled_tokens

    setattr(patched_prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, True)
    return patched_prepare_inputs


def make_patched_sample_tokens_gpu(_orig_sample_tokens):
    """Attach a CUDA worker decision produced during async sampling."""
    if getattr(_orig_sample_tokens, _WORKER_SAMPLE_TOKENS_PATCH_MARKER, False):
        return _orig_sample_tokens

    @wraps(_orig_sample_tokens)
    def patched_sample_tokens(self, grammar_output, *args, **kwargs):
        stage_runner = getattr(self, "_gr_gpu_stage_runner", None)
        if stage_runner is not None and stage_runner.pending is not None:
            return stage_runner.sample(grammar_output)
        return _orig_sample_tokens(self, grammar_output, *args, **kwargs)

    setattr(patched_sample_tokens, _WORKER_SAMPLE_TOKENS_PATCH_MARKER, True)
    return patched_sample_tokens


def make_patched_determine_batch_execution_and_padding_gpu(_orig_determine):
    """Override cudagraph_mode to FULL for beam decode steps.

    In standard vllm, cascade attention sets ``use_cascade_attn=True``,
    which causes ``_determine_batch_execution_and_padding`` to call
    ``dispatch_cudagraph(disable_full=True)`` — preventing FULL CUDA Graph
    capture.  For beam search decode, we override the result to force
    ``CUDAGraphMode.FULL`` with ``uniform_decode=True``, the same explicit
    override the vllm-GR ACL Graph integration applies for beam decode.

    This patch is the single entry point that enables CUDA Graph capture
    for beam decode: all downstream logic (CUDAGraphWrapper lazy capture
    and replay) flows from the ``cudagraph_runtime_mode`` value set here.
    """
    if getattr(_orig_determine, _WORKER_DETERMINE_PATCH_MARKER, False):
        return _orig_determine

    @wraps(_orig_determine)
    def patched_determine(self, *args, **kwargs):
        from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

        result = _orig_determine(self, *args, **kwargs)

        beam_info = getattr(self, "_current_beam_info", None)
        if beam_info is not None:
            beam_data = beam_info.get("beam_data", {})
            beam_keys = {
                req_id
                for req_id, mdata in beam_data.items()
                if mdata.get("is_beam_decode", False) and "decode_step" in mdata
            }
            if beam_keys:
                # Beam decode FULL mode requires an active BeamSearchSession
                # — the metadata builder and Triton kernels read slot/pool/
                # suffix-buf pointers from it. Missing session state must not
                # allow the FULL path to dereference absent pool bindings.
                if beam_info.get("beam_session") is None:
                    if result[0] == CUDAGraphMode.FULL:
                        result = (
                            CUDAGraphMode.NONE,
                            BatchDescriptor(num_tokens=result[1].num_tokens),
                        ) + result[2:]
                    return result

                # FULL is granted only for a PURE beam-decode batch: the
                # captured beam graph contains no standard-attention path, so
                # a mixed batch replaying it would compute garbage for the
                # non-beam requests.  ``req_ids`` (filled by the patched
                # _prepare_inputs) enumerates the whole batch.
                batch_req_ids = set(beam_info.get("req_ids") or [])
                pure_beam_batch = batch_req_ids == beam_keys
                if not pure_beam_batch:
                    # Mixed batch: never touch the dispatch result beyond
                    # making sure a native FULL graph is not replayed for beam
                    # tokens (it contains no beam kernels).
                    if result[0] == CUDAGraphMode.FULL:
                        result = (
                            CUDAGraphMode.NONE,
                            BatchDescriptor(num_tokens=result[1].num_tokens),
                        ) + result[2:]
                    return result

                # The startup beam graph is captured only for the default
                # no-LoRA descriptor and ``_beam_full_graph_ready`` is keyed
                # by width alone.  Granting FULL here would rewrite the
                # descriptor without has_lora/num_active_loras and silently
                # route an active-LoRA beam batch to the LoRA-free graph
                # (and a native FULL graph lacks beam kernels anyway), so
                # downgrade to eager until LoRA-aware beam graphs are
                # captured per full descriptor.
                if getattr(result[1], "has_lora", False):
                    if result[0] == CUDAGraphMode.FULL:
                        result = (
                            CUDAGraphMode.NONE,
                            BatchDescriptor(num_tokens=result[1].num_tokens),
                        ) + result[2:]
                    return result

                # Beam graphs are opt-in via GRConfig (beam_max_width) and
                # can be disabled for A/B comparison (beam_graph_enabled /
                # CLI --no-beam-graph).  Disabled or unconfigured -> fully
                # eager, nothing below changes the dispatch result.
                if not _beam_graph_enabled_for_run(self):
                    return result

                # Exact-width gate: compare the UNPADDED scheduled token count
                # against the captured graph's width.  ``result[1].num_tokens``
                # is the dispatcher-padded size, so a width of e.g. 100 with a
                # graph captured at 128 would otherwise replay a graph whose
                # tail 28 rows are garbage (only accidentally masked by
                # cu_seqlens / logits_indices).  For a pure beam-decode batch
                # the unpadded count is exactly the beam width.
                num_tokens_arg = kwargs.get("num_tokens")
                if num_tokens_arg is None and args:
                    num_tokens_arg = args[0]
                real_num_tokens = (
                    int(num_tokens_arg) if num_tokens_arg is not None else result[1].num_tokens
                )

                # The beam graph is captured once at startup; the runtime is
                # replay-only.  Grant FULL iff a graph for this exact width
                # (== real_num_tokens for a pure beam batch) exists in the
                # captured set; otherwise fall back to eager — and if the
                # native dispatch already returned FULL, downgrade to NONE so
                # the beam tokens never replay a native (beam-kernel-free) graph.
                if not _beam_full_graph_ready(self, real_num_tokens):
                    if result[0] == CUDAGraphMode.FULL:
                        result = (
                            CUDAGraphMode.NONE,
                            BatchDescriptor(num_tokens=real_num_tokens),
                        ) + result[2:]
                    return result

                new_descriptor = BatchDescriptor(num_tokens=real_num_tokens, uniform=True)
                logger.debug(
                    "Beam decode: cudagraph_mode %s -> FULL, batch_descriptor %s -> %s",
                    result[0],
                    result[1],
                    new_descriptor,
                )
                # Count only real runtime dispatches.  Startup beam capture
                # calls ``_dummy_beam_forward`` directly and therefore never
                # reaches this branch.  The counter gives integration tests
                # and profiling harnesses an authoritative, worker-local
                # assertion that a request selected the captured FULL graph;
                # it is deliberately outside the graph and has no CUDA work.
                self._beam_graph_full_dispatch_count = (
                    int(getattr(self, "_beam_graph_full_dispatch_count", 0)) + 1
                )
                result = (CUDAGraphMode.FULL, new_descriptor) + result[2:]

        return result

    setattr(patched_determine, _WORKER_DETERMINE_PATCH_MARKER, True)
    return patched_determine


# ---------------------------------------------------------------------------
# Beam decode CUDA Graph capture helpers
# ---------------------------------------------------------------------------

_BEAM_DUMMY_REQ_ID = "vllm-gr-beam-graph-dummy"


def _capture_create_dummy_context(
    runner, gr_cfg, width: int, steps_upper: int, prefix_len: int
) -> tuple:
    """Create the BeamAttentionContext, dummy session, and beam_info for capture.

    Returns:
        ``(context, session, beam_info, batch_descriptor)`` — all needed by
        ``_dummy_beam_forward`` and the capture orchestration loop.
    """
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor

    from vllm_gr.v1.beam.session import BeamSearchSession

    context = _create_beam_context(runner, gr_cfg, max_batch=1)
    slot = context.registry.allocate(_BEAM_DUMMY_REQ_ID, beam_width=width, prefix_len=prefix_len)
    session = BeamSearchSession(session_id=_BEAM_DUMMY_REQ_ID, slot=slot, context=context)
    session.begin_session(width, prefix_len)

    runner._beam_context = context
    runner._beam_sessions = {_BEAM_DUMMY_REQ_ID: session}

    beam_data = {
        _BEAM_DUMMY_REQ_ID: {
            "prefix_len": prefix_len,
            "cache_len": prefix_len,
            "max_decode_steps": steps_upper,
            "is_beam_decode": True,
            "beam_width": width,
            "beam_decode_steps": 1,
            "parent_beam_ids": [0] * width,
            "decode_step": 0,
        }
    }
    beam_info = {
        "beam_data": beam_data,
        "req_ids": [_BEAM_DUMMY_REQ_ID],
        "beam_session": session,
    }

    batch_descriptor = BatchDescriptor(num_tokens=width, uniform=True)
    return context, session, beam_info, batch_descriptor


def _dummy_beam_forward(
    runner,
    context,
    beam_info,
    batch_descriptor,
    width: int,
    prefix_len: int,
    capture: bool,
) -> None:
    """Execute one dummy forward for beam-decode CUDA Graph capture (or warmup).

    This function mutates several runner buffers in a **sequence-sensitive**
    order.  The ordering constraints are:

    1. ``decode_step_buf.fill_`` — device scalar, must precede model forward.
    2. ``input_ids`` / ``positions`` — runner persistent GPU buffers, must
       precede the model call that reads them.
    3. ``optimistic_seq_lens_cpu`` → ``seq_lens.copy_`` — CPU write must
       complete before the H2D copy.
    4. ``query_start_loc`` — CPU write → H2D copy, same ordering.
    5. ``block_table.commit_block_table`` — must follow the seq_lens setup.
    6. ``slot_mappings`` fill — must precede ``_build_attention_metadata``.
    7. ``BEAM_DATA_VAR.set`` / ``reset`` — contextvars scope around the
       entire metadata + forward call.

    Args:
        runner: The GPU ModelRunner.
        context: The BeamAttentionContext (for ``decode_step_buf``).
        beam_info: The beam context dict to publish via ``BEAM_DATA_VAR``.
        batch_descriptor: The ``BatchDescriptor`` for this capture width.
        width: Beam width (== num_tokens for uniform decode).
        prefix_len: Dummy prefix length (arbitrary under host-param upper bounds).
        capture: ``True`` for the FULL-mode capture forward; ``False`` for
            eager warmup forwards (NONE mode — primes Triton JIT + FA2 shapes).
    """
    from vllm.forward_context import set_forward_context
    from vllm.v1.cudagraph_dispatcher import CUDAGraphMode

    from vllm_gr.v1.beam import BEAM_DATA_VAR

    # 1. Refresh the per-step device buffer (same as runtime path).
    context.decode_step_buf.fill_(0)

    # 2. Dummy inputs in the runner's persistent buffers (fixed addresses).
    input_ids = runner.input_ids.gpu[:width]
    input_ids.fill_(1)
    positions = runner.positions[:width]
    positions.copy_(
        torch.arange(prefix_len, prefix_len + width, dtype=positions.dtype, device=runner.device)
    )

    # 3. Dummy batch state, mirroring _dummy_run's bookkeeping:
    #    1 req with `width` scheduled tokens; seq_len = prefix_len + width.
    #    MUST: CPU write precedes the H2D copy below.
    runner.optimistic_seq_lens_cpu[:1] = prefix_len + width
    runner.optimistic_seq_lens_cpu[1:] = 0
    runner.seq_lens.copy_(runner.optimistic_seq_lens_cpu, non_blocking=True)

    # 4. MUST: CPU write precedes copy_to_gpu.
    runner.query_start_loc.np[0] = 0
    runner.query_start_loc.np[1] = width
    runner.query_start_loc.np[2:] = 0
    runner.query_start_loc.copy_to_gpu()

    # 5. Commit block table for 1 request.
    runner.input_batch.block_table.commit_block_table(1)

    # 6. Dummy runs have no real slot assignments — fill with -1 so
    #    concat_and_cache kernels skip the KV write (same as _dummy_run).
    slot_mappings_by_group, _ = runner._get_slot_mappings(
        num_tokens_padded=width,
        num_reqs_padded=1,
        num_tokens_unpadded=width,
    )
    if slot_mappings_by_group is not None:
        for sm in slot_mappings_by_group.values():
            sm.fill_(-1)

    # 7. Publish beam context and run the model forward.
    token = BEAM_DATA_VAR.set(beam_info)
    try:
        metadata, _ = runner._build_attention_metadata(
            num_tokens=width,
            num_reqs=1,
            max_query_len=width,
            num_tokens_padded=width,
            num_reqs_padded=1,
            for_cudagraph_capture=capture,
            slot_mappings=slot_mappings_by_group,
        )
        with set_forward_context(
            metadata,
            runner.vllm_config,
            num_tokens=width,
            cudagraph_runtime_mode=(CUDAGraphMode.FULL if capture else CUDAGraphMode.NONE),
            batch_descriptor=batch_descriptor,
        ):
            runner.model(input_ids=input_ids, positions=positions)
    finally:
        BEAM_DATA_VAR.reset(token)


def capture_beam_decode_graph(runner) -> bool:
    """Startup dummy capture of the beam-decode FULL CUDA graph.

    Runs once after ``capture_model()`` when the GRConfig beam graph is active
    (``beam_graph_enabled`` and ``beam_max_width`` set; CLI
    ``--beam-max-width``).  Mirrors
    vllm's own ``_warmup_and_capture`` pattern: N eager warmup forwards (mode
    NONE — primes the Triton JIT of cache_unshared_kv/copy_suffix_kv and FA2
    shape selection), then one FULL-mode forward which the CUDAGraphWrapper
    captures.  The captured graph's shape is (num_tokens=width, uniform=True);
    every runtime-varying quantity (decode_step, effective KV lengths, prefix
    extent, block-table contents) is device-side data or a fixed-address
    buffer value, so the single graph serves every decode step and every
    prompt length (Prefix FA host-side parameters use global upper bounds).

    The capture forward consumes dummy inputs and its output is discarded —
    exactly the native startup-capture contract, so no post-capture replay is
    needed.  Real requests replay only (``_beam_full_graph_ready`` gates FULL
    in patched_determine); uncaptured shapes fall back to eager.

    Returns True if a graph was captured.
    """
    from vllm_gr.v1.worker.cuda.beam_decode_graph import get_beam_decode_graph
    from vllm_gr.v1.worker.gr_context import (
        _get_gr_config,
    )

    gr_cfg = _get_gr_config(runner)
    width = gr_cfg.beam_max_width
    if (
        width is None
        or not gr_cfg.beam_graph_enabled
        or runner.model_config.enforce_eager
        or 1 not in gr_cfg.deployment.beam_batch_sizes
    ):
        return False

    # ── TP / DP ubatching guard ─────────────────────────────────
    # Beam capture runs on a side stream *after* native capture_model()
    # returns, bypassing vllm's graph_capture() context that provides
    # TP rank synchronisation.  With TP>1 the captured graphs would
    # diverge across ranks -> silent wrong results on replay.
    # With DP ubatching, the native _determine result[2:] (should_ubatch,
    # num_tokens_across_dp) is computed from the original token counts and
    # is NOT overridden by the beam patch -- replaying a beam graph with
    # stale ubatch metadata would produce incorrect micro-batch slices.
    # Fail fast rather than silently producing incorrect graphs.
    tp_size = runner.parallel_config.tensor_parallel_size
    if tp_size > 1:
        logger.warning(
            "vllm-gr: beam decode CUDA graph is not supported with "
            "tensor parallelism (TP=%d); beam decode will run eager",
            tp_size,
        )
        return False
    if runner.parallel_config.use_ubatching:
        logger.warning(
            "vllm-gr: beam decode CUDA graph is not supported with "
            "DP ubatching; beam decode will run eager",
        )
        return False
    # ─────────────────────────────────────────────────────────────

    # Re-entry guard: if a previous capture already produced a graph for this
    # width, do NOT rebuild the beam context (a new context moves every
    # buffer's data_ptr, so the existing graph would silently read/write the
    # OLD pool while the runtime refreshes the NEW one).
    if _beam_full_graph_ready(runner, width):
        return True

    steps_upper = int(gr_cfg.beam_max_decode_steps)
    # Dummy-capture constants:
    # - prefix_len: fake prefix length for the dummy session.  Under the
    #   Prefix FA Host-side parameter global-upper-bound design, the captured
    #   graph's shape does not depend on the prefix length (FA2 reads the true
    #   extent from device-side seqused_k), so 1024 is arbitrary.
    # The shared component runs two eager warmups before FULL capture.
    prefix_len = 1024

    # 1. Create the worker-level beam context + one dummy session.
    context, session, beam_info, batch_descriptor = _capture_create_dummy_context(
        runner, gr_cfg, width, steps_upper, prefix_len
    )

    # 2. Warmup + capture.
    captured = False
    try:
        graph = get_beam_decode_graph(runner)
        graph.capture(
            lambda capture: _dummy_beam_forward(
                runner, context, beam_info, batch_descriptor, width, prefix_len, capture
            ),
            persistent_install=True,
        )
        captured = graph.ready
    finally:
        # Drop the dummy session even if capture threw. The context and
        # pools stay — a captured graph (if any) references their addresses.
        session.end_session()
        runner._beam_sessions.pop(_BEAM_DUMMY_REQ_ID, None)
        runner._current_beam_info = None

    if captured:
        logger.info(
            "vllm-gr: startup dummy capture of beam decode graph succeeded "
            "(num_tokens=%d, uniform=True); runtime is replay-only",
            width,
        )
    else:
        logger.warning(
            "vllm-gr: beam dummy capture did NOT produce a graph for "
            "num_tokens=%d; beam decode will run eager",
            width,
        )
    return captured


def discover_worker_runners() -> tuple[tuple[str, type], ...]:
    """Discover the upstream GPU ModelRunner without importing NPU runtime."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError as exc:
        logger.warning(
            "Failed to import vllm.v1.worker.gpu_model_runner.GPUModelRunner; "
            "skipping vllm-gr GPU beam-search runner patches for this attempt. "
            "ImportError: %s",
            exc,
        )
        logger.debug("GPUModelRunner import traceback", exc_info=True)
        return ()
    return (("GPUModelRunner", GPUModelRunner),)


make_patched_prepare_inputs = make_patched_prepare_inputs_gpu
make_patched_sample_tokens = make_patched_sample_tokens_gpu
make_patched_determine_batch_execution_and_padding = (
    make_patched_determine_batch_execution_and_padding_gpu
)


def make_patched_model_forward_gpu(_orig_model_forward):
    """Wrap _model_forward to intercept and replay pure prefill CUDA graphs."""
    if getattr(_orig_model_forward, _WORKER_MODEL_FORWARD_PATCH_MARKER, False):
        return _orig_model_forward

    @wraps(_orig_model_forward)
    def patched_model_forward(self, input_ids=None, positions=None, *args, **kwargs):
        scheduler_output = getattr(self, "_current_scheduler_output", None)
        gr_stages = getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, None)
        batch_view = None
        if gr_stages:
            # Decode uses its independent startup graph. A Batch Session has
            # exactly one stage descriptor whose request_ids define the
            # immutable logical order for this Prefill dispatch.
            if len(gr_stages) != 1:
                return _orig_model_forward(
                    self, input_ids=input_ids, positions=positions, *args, **kwargs
                )
            try:
                batch_view = BeamBatchExecutionView.from_stage_metadata(
                    next(iter(gr_stages.values()))
                )
            except BeamExecutionContractError:
                logger.debug("Prefill CUDA graph fallback: invalid_batch_contract")
                return _orig_model_forward(
                    self, input_ids=input_ids, positions=positions, *args, **kwargs
                )
            if batch_view.kind != "prefill":
                return _orig_model_forward(
                    self, input_ids=input_ids, positions=positions, *args, **kwargs
                )
            if getattr(self.model_config, "enforce_eager", False):
                logger.debug("Prefill CUDA graph fallback: enforce_eager")
                return _orig_model_forward(
                    self, input_ids=input_ids, positions=positions, *args, **kwargs
                )
            if is_prefill_graph_enabled(self) and not hasattr(self, "_prefill_graph_runner"):
                logger.debug("Prefill CUDA graph fallback: runner_not_initialized")
                return _orig_model_forward(
                    self, input_ids=input_ids, positions=positions, *args, **kwargs
                )
        if scheduler_output is not None:
            if (
                is_prefill_graph_enabled(self)
                and is_prefill_phase_batch(self, scheduler_output)
                # Dense-Prefill graphs replay only for fresh prompts.
                and (
                    _get_gr_config(self).deployment.dense_prefill == "off"
                    or scheduled_requests_are_fresh(self, scheduler_output)
                )
            ):
                raw_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", [])
                # The scheduler already computed the batch's token total; only
                # the request count is derived here (the runner re-derives the
                # per-request counts it needs from the dict itself).
                if batch_view is not None:
                    if not isinstance(raw_scheduled_tokens, dict):
                        logger.debug(
                            "Prefill CUDA graph fallback: batch_session_missing_request_ids"
                        )
                        return _orig_model_forward(
                            self, input_ids=input_ids, positions=positions, *args, **kwargs
                        )
                    try:
                        batch_view.validate_request_order(tuple(raw_scheduled_tokens))
                    except BeamExecutionContractError:
                        logger.debug("Prefill CUDA graph fallback: batch_session_request_order")
                        return _orig_model_forward(
                            self, input_ids=input_ids, positions=positions, *args, **kwargs
                        )
                num_reqs = (
                    batch_view.batch_size if batch_view is not None else len(raw_scheduled_tokens)
                )
                total_tokens = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
                if not total_tokens and raw_scheduled_tokens:
                    total_tokens = (
                        sum(raw_scheduled_tokens.values())
                        if isinstance(raw_scheduled_tokens, dict)
                        else sum(raw_scheduled_tokens)
                    )

                if 1 <= num_reqs <= SUPPORTED_NUM_REQS[-1] and total_tokens > 0:
                    p_runner = get_prefill_graph_runner(self)
                    bucket = p_runner.match_bucket(total_tokens)
                    reason = "no_matching_bucket" if bucket is None else "graph_not_captured"
                    if gr_stages:
                        # Native preprocessing may pad these 1D tensors to a
                        # compilation bucket. Only their scheduled prefix is
                        # live: the Prefill runner copies that prefix, preserves
                        # real Attention lengths, and masks its own padded KV
                        # slots. Native padding must not determine GR's bucket.
                        if (
                            input_ids is None
                            or input_ids.ndim != 1
                            or input_ids.shape[0] < total_tokens
                            or positions is None
                            or tuple(positions.shape) != tuple(input_ids.shape)
                            or kwargs.get("inputs_embeds") is not None
                            or kwargs.get("intermediate_tensors") is not None
                        ):
                            reason = "unsupported_inputs"
                            bucket = None
                    if bucket is not None and p_runner.has_captured_graph(bucket, num_reqs):
                        # Extract slot mapping from active block table
                        slot_mapping = None
                        if (
                            hasattr(self, "input_batch")
                            and getattr(self.input_batch, "block_table", None) is not None
                        ):
                            try:
                                blk_table = self.input_batch.block_table[0]
                                if hasattr(blk_table, "slot_mapping") and hasattr(
                                    blk_table.slot_mapping, "gpu"
                                ):
                                    slot_mapping = blk_table.slot_mapping.gpu
                            except Exception as e:
                                logger.warning("Could not get slot_mapping: %r", e)
                        output_buf = p_runner.execute_model_prefill(
                            bucket=bucket,
                            input_ids=input_ids,
                            positions=positions,
                            slot_mapping=slot_mapping,
                            num_scheduled_tokens=raw_scheduled_tokens,
                        )

                        if output_buf is not None:
                            dispatches = getattr(self, "_gr_prefill_graph_dispatches", None)
                            if dispatches is None:
                                dispatches = self._gr_prefill_graph_dispatches = {}
                            key = f"replay:{num_reqs}"
                            dispatches[key] = dispatches.get(key, 0) + 1
                            logger.debug(
                                "Prefill CUDA graph dispatch: scheduled=%d requests=%d "
                                "bucket=%s outcome=replay",
                                total_tokens,
                                num_reqs,
                                bucket,
                            )
                            return output_buf
                        reason = getattr(p_runner, "last_replay_reason", "runner_rejected")
                    dispatches = getattr(self, "_gr_prefill_graph_dispatches", None)
                    if dispatches is None:
                        dispatches = self._gr_prefill_graph_dispatches = {}
                    key = f"fallback:{num_reqs}:{reason}"
                    dispatches[key] = dispatches.get(key, 0) + 1
                    logger.debug(
                        "Prefill CUDA graph dispatch: scheduled=%d requests=%d "
                        "bucket=%s outcome=%s",
                        total_tokens,
                        num_reqs,
                        bucket,
                        reason,
                    )
        return _orig_model_forward(self, input_ids=input_ids, positions=positions, *args, **kwargs)

    setattr(patched_model_forward, _WORKER_MODEL_FORWARD_PATCH_MARKER, True)
    return patched_model_forward


def capture_prefill_graphs(runner) -> None:
    """Startup capture of pure-prefill CUDA graphs keyed by bucket and request count.

    Chained after the beam decode-graph capture in ``capture_model``; runs
    only when the prefill graph env gate is enabled. Only explicitly
    recoverable capture failures fall back to eager prefill.
    """
    if not is_prefill_graph_enabled(runner):
        return
    if getattr(runner, "_prefill_graphs_captured", False):
        return
    try:
        # CUDA graph capture requires a non-default stream; capture_model()
        # already returned, so its capture stream context is gone. Mirror the
        # beam decode-graph capture: a side stream, ordered against the
        # default stream before and after.
        side = torch.cuda.Stream()
        side.wait_stream(torch.accelerator.current_stream())
        with torch.cuda.stream(side):
            p_runner = get_prefill_graph_runner(runner)
            p_runner.capture_graphs(runner)
        torch.accelerator.current_stream().wait_stream(side)
        torch.accelerator.synchronize()
        runner._prefill_graphs_captured = True
    except Exception as error:
        from vllm_gr.v1.worker.cudagraph_capture import is_recoverable_capture_error

        if not isinstance(error, RuntimeError) or not is_recoverable_capture_error(error):
            raise
        torch.accelerator.synchronize()
        logger.exception(
            "vllm-gr: prefill CUDA graph capture failed; prefill will fall back to eager"
        )


make_patched_model_forward = make_patched_model_forward_gpu


def is_patched(runner_class: type) -> bool:
    from vllm_gr.v1.worker.runner_lifecycle import has_runner_lifecycle

    return is_worker_runner_patched(runner_class) and has_runner_lifecycle(runner_class, V1_RUNNER)


def install() -> None:
    """Install CUDA Worker utilities and the V1 native runner wrappers."""
    from vllm_gr.v1.worker.cuda.gpu_postprocess_workspace import (
        make_load_model_with_beam_postprocess_workspace,
    )
    from vllm_gr.v1.worker.cuda.runner_installer import install_worker_hooks
    from vllm_gr.v1.worker.cuda.v1.gpu_beam_stage_runner_v1 import CudaStageHooksV1
    from vllm_gr.v1.worker.model_runner_patch import install_v1_worker_hooks

    install_worker_hooks()
    install_v1_worker_hooks(
        sys.modules[__name__],
        cuda_hooks_type=CudaStageHooksV1,
        workspace_load_patcher=make_load_model_with_beam_postprocess_workspace,
        implementation_id=V1_RUNNER,
    )
