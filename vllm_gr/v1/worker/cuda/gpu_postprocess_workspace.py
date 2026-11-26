# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Fixed-address CUDA Beam workspaces, prewarming, and model-load accounting."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger

from vllm_gr.v1.beam.batch_contracts import SUPPORTED_BEAM_BATCH_SIZES

if TYPE_CHECKING:
    from vllm_gr.ops.cuda.beam_state import BeamStateScratch

logger = init_logger(__name__)

_BEAM_POSTPROCESS_LOAD_PATCH_MARKER = "_vllm_gr_beam_postprocess_load_patched"


def _validate_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class BeamPostprocessWorkspaceKey:
    """Shape/dtype identity for one fixed-address postprocess workspace.

    The CUDA kernels have exact contiguous shape contracts.  In particular, a
    top-left ``[:width, :top_k]`` slice of a larger two-dimensional bucket is
    not contiguous, so a request may only use a workspace whose width/top-k
    exactly match its operator geometry.  ``logits_dtype`` is part of the key
    even though the operator outputs are int32/fp32; this prevents an fp16
    warmup from silently standing in for a bf16 deployment.
    """

    beam_width: int
    top_k: int
    logits_dtype: torch.dtype


@dataclass(slots=True)
class BeamPostprocessWorkspace:
    """Runner-owned fixed storage for one CUDA Beam postprocess geometry.

    This object is scratch/output storage only.  It deliberately owns no
    decode-step counter, request identity, or continuation state; the existing
    ``BeamDecisionBuffers`` remain the source of cumulative cross-step state.
    Keeping this distinction explicit prevents an allocation optimization from
    growing into a second Beam lifecycle protocol.

    The first dimension is always ``max_batch``.  Runtime code takes leading
    batch slices, while tests can construct ``max_batch=2`` even though the
    current Worker-level Beam context accepts one active session.
    """

    key: BeamPostprocessWorkspaceKey
    max_batch: int
    max_decode_steps: int
    device: torch.device

    # Constrained Top-K outputs: [Bmax, W, K] / [Bmax, W].
    topk_token_ids: torch.Tensor
    topk_scores: torch.Tensor
    topk_valid_mask: torch.Tensor
    topk_valid_counts: torch.Tensor

    # Device inputs for constrained Top-K.
    prefix_token_ids: torch.Tensor
    temperatures: torch.Tensor

    # BeamSearchGroup outputs: [Bmax, W] and fixed sequence extent.
    selected_token_ids: torch.Tensor
    selected_parent_ids: torch.Tensor
    selected_scores: torch.Tensor
    selected_flat_indices: torch.Tensor
    next_sequence: torch.Tensor

    # Caller-owned hierarchical selection ping-pong buffers.
    selection_scratch: Any

    # Optional persistent-runner scratch, allocated once when the runner binds.
    # Ordinary postprocess callers retain their existing allocations and ABI.
    state_scratch: BeamStateScratch | None = None
    execution_owner: Any = None

    # Optional FlashInfer Top-K input shared by every batch adapter of this
    # worker. It is allocated with the workspace, during model load, so KV-cache
    # sizing already counts it. None when ``gr.topk_select`` is not ``flash``,
    # the device cannot run it, or no batch geometry qualifies.
    flash_logits: torch.Tensor | None = None

    @classmethod
    def allocate(
        cls,
        *,
        max_batch: int,
        beam_width: int,
        top_k: int,
        max_decode_steps: int,
        device: torch.device | str,
        logits_dtype: torch.dtype,
        flash_logits_rows: int = 0,
        flash_logits_vocab_size: int = 0,
    ) -> BeamPostprocessWorkspace:
        """Allocate every tensor once; callers retain the returned object."""

        _validate_positive_int("max_batch", max_batch)
        _validate_positive_int("beam_width", beam_width)
        _validate_positive_int("top_k", top_k)
        _validate_positive_int("max_decode_steps", max_decode_steps)

        resolved_device = torch.device(device)
        from vllm_gr.ops.cuda.beam_selection import allocate_beam_selection_scratch

        key = BeamPostprocessWorkspaceKey(beam_width, top_k, logits_dtype)
        flash_logits = None
        if flash_logits_rows > 0 and flash_logits_vocab_size > 0:
            from vllm_gr.ops.cuda.unconstrained_topk_mcap import allocate_flash_logits

            flash_logits = allocate_flash_logits(
                flash_logits_rows, flash_logits_vocab_size, logits_dtype, resolved_device
            )
        candidate_shape = (max_batch, beam_width, top_k)
        beam_shape = (max_batch, beam_width)
        sequence_shape = (max_batch, beam_width, max_decode_steps)

        def device_empty(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(shape, dtype=dtype, device=resolved_device)

        # Group selects W winners from W*K candidates.  Its scratch is the
        # largest selection scratch needed by the worker-specific terminal
        # reorder as well, so no second final-step allocation is necessary.
        selection_scratch = allocate_beam_selection_scratch(
            max_batch,
            beam_width,
            top_k,
            beam_width,
            resolved_device,
        )

        return cls(
            key=key,
            max_batch=max_batch,
            max_decode_steps=max_decode_steps,
            device=resolved_device,
            topk_token_ids=device_empty(candidate_shape, torch.int32),
            topk_scores=device_empty(candidate_shape, torch.float32),
            topk_valid_mask=device_empty(candidate_shape, torch.bool),
            topk_valid_counts=device_empty(beam_shape, torch.int32),
            prefix_token_ids=device_empty((max_batch, beam_width, 3), torch.int32),
            temperatures=device_empty(beam_shape, torch.float32),
            selected_token_ids=device_empty(beam_shape, torch.int32),
            selected_parent_ids=device_empty(beam_shape, torch.int32),
            selected_scores=device_empty(beam_shape, torch.float32),
            selected_flat_indices=device_empty(beam_shape, torch.int32),
            next_sequence=device_empty(sequence_shape, torch.int32),
            selection_scratch=selection_scratch,
            flash_logits=flash_logits,
        )

    def device_tensors(self) -> tuple[torch.Tensor, ...]:
        scratch = self.selection_scratch
        tensors: tuple[torch.Tensor, ...] = (
            self.topk_token_ids,
            self.topk_scores,
            self.topk_valid_mask,
            self.topk_valid_counts,
            self.prefix_token_ids,
            self.temperatures,
            self.selected_token_ids,
            self.selected_parent_ids,
            self.selected_scores,
            self.selected_flat_indices,
            self.next_sequence,
            scratch.local_scores,
            scratch.local_flat_indices,
            scratch.merge_scores,
            scratch.merge_flat_indices,
        )
        if self.state_scratch is not None:
            tensors += self.state_scratch.device_tensors()
        if self.flash_logits is not None:
            tensors += (self.flash_logits,)
        return tensors

    @property
    def device_memory_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.device_tensors())


class BeamPostprocessWorkspacePool:
    """Prewarmed workspace buckets; request-time creation is intentionally absent."""

    def __init__(self) -> None:
        self._buckets: dict[BeamPostprocessWorkspaceKey, BeamPostprocessWorkspace] = {}

    def add(self, workspace: BeamPostprocessWorkspace) -> None:
        if workspace.key in self._buckets:
            raise ValueError(f"duplicate Beam postprocess workspace bucket {workspace.key}")
        self._buckets[workspace.key] = workspace

    def require(self, key: BeamPostprocessWorkspaceKey) -> BeamPostprocessWorkspace:
        try:
            return self._buckets[key]
        except KeyError as exc:
            raise KeyError(
                f"Beam postprocess workspace {key} was not prewarmed; "
                "the hot path is not allowed to allocate a bucket"
            ) from exc

    @property
    def keys(self) -> tuple[BeamPostprocessWorkspaceKey, ...]:
        return tuple(self._buckets)

    @property
    def device_memory_bytes(self) -> int:
        """Return the device storage owned by all prewarmed buckets."""

        return sum(workspace.device_memory_bytes for workspace in self._buckets.values())


def flash_bytes(workspace: Any) -> int:
    logits = getattr(workspace, "flash_logits", None)
    return 0 if logits is None else logits.numel() * logits.element_size()


def initialize_beam_postprocess_workspaces(
    runner,
    *,
    max_batch: int | None = None,
):
    """Prewarm the configured GPU Beam postprocess bucket on ``runner``.

    V1 reserves the largest supported logical batch so every admitted size
    reuses the same fixed-address state and selection storage.
    """

    device = torch.device(getattr(runner, "device", "cpu"))
    if device.type == "cpu":
        return None
    provider_has_selector = False
    if device.type != "cuda":
        from vllm_gr.platforms.interface import FullVocabularyProvider
        from vllm_gr.platforms.registry import get_active_backend

        provider_has_selector = isinstance(get_active_backend(), FullVocabularyProvider)
        if not provider_has_selector:
            return None

    from vllm_gr.config import resolve_candidate_width
    from vllm_gr.config.models import ConstraintMode, configured_constraint_mode
    from vllm_gr.v1.worker.gr_context import (
        _get_gr_config,
    )

    gr_cfg = _get_gr_config(runner)
    width = gr_cfg.beam_max_width
    if not bool(gr_cfg.beam_worker_decision) or width is None:
        return None
    candidate_width = resolve_candidate_width(gr_cfg)

    context = getattr(runner, "_beam_context", None)
    # The Beam context is created later, on the first admitted batch, with
    # max_batch=max(SUPPORTED_BEAM_BATCH_SIZES); the workspace built here must
    # match it or the stage runner rejects the session as incompatible.
    default_max_batch = max(SUPPORTED_BEAM_BATCH_SIZES)
    resolved_max_batch = (
        int(max_batch)
        if max_batch is not None
        else int(getattr(context, "max_batch", default_max_batch))
    )
    dtype = getattr(getattr(runner, "model_config", None), "dtype", None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(
            f"CUDA Beam postprocess workspace requires model_config.dtype, got {dtype!r}"
        )

    # Flash Top-K serves only the unconstrained selector. Reserve its input here,
    # once per worker, rather than lazily in the first request.
    # A forced positive head_slice slices every stage, so Flash never runs.
    flash_rows = flash_vocab_size = 0
    deployment = gr_cfg.deployment
    if (
        not provider_has_selector
        and deployment.topk_select == "flash"
        and configured_constraint_mode(gr_cfg) is ConstraintMode.NONE
        and not (isinstance(deployment.head_slice, int) and deployment.head_slice > 0)
    ):
        from vllm_gr.ops.cuda.unconstrained_topk_mcap import flash_logits_rows

        flash_rows = flash_logits_rows(int(width), resolved_max_batch)
        flash_vocab_size = int(runner.model_config.get_vocab_size()) if flash_rows else 0

    key = BeamPostprocessWorkspaceKey(int(width), int(candidate_width), dtype)
    existing = getattr(runner, "_beam_postprocess_workspaces", None)
    if existing is not None:
        workspace = existing.require(key)
        if workspace.max_batch != resolved_max_batch or workspace.max_decode_steps != int(
            gr_cfg.beam_max_decode_steps
        ):
            raise RuntimeError(
                "prewarmed Beam postprocess workspace geometry changed after "
                "initialization; rebuilding fixed-address buffers is forbidden"
            )
        return existing

    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=resolved_max_batch,
        beam_width=int(width),
        top_k=int(candidate_width),
        max_decode_steps=int(gr_cfg.beam_max_decode_steps),
        device=device,
        logits_dtype=dtype,
        flash_logits_rows=flash_rows,
        flash_logits_vocab_size=flash_vocab_size,
    )
    pool = BeamPostprocessWorkspacePool()
    pool.add(workspace)
    runner._beam_postprocess_workspaces = pool
    logger.info(
        "vllm-gr: prewarmed CUDA Beam postprocess workspace "
        "(batch=%d, width=%d, top_k=%d, dtype=%s, device_bytes=%d, flash_logits_bytes=%d)",
        resolved_max_batch,
        width,
        candidate_width,
        dtype,
        workspace.device_memory_bytes,
        flash_bytes(workspace),
    )
    return pool


def make_load_model_with_beam_postprocess_workspace(original):
    """Initialize fixed Beam postprocess storage after model weights load."""

    if getattr(original, _BEAM_POSTPROCESS_LOAD_PATCH_MARKER, False):
        return original

    @wraps(original)
    def load_model_with_beam_postprocess_workspace(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        workspace_pool = initialize_beam_postprocess_workspaces(self)
        if workspace_pool is not None:
            # The pool is allocated after native load_model has measured the
            # model weights, but before Worker memory profiling.  Treat its
            # device storage as model-owned memory so KV-cache capacity is not
            # computed from bytes that are already permanently resident.
            self.model_memory_usage = int(getattr(self, "model_memory_usage", 0)) + int(
                workspace_pool.device_memory_bytes
            )
        return result

    setattr(
        load_model_with_beam_postprocess_workspace,
        _BEAM_POSTPROCESS_LOAD_PATCH_MARKER,
        True,
    )
    return load_model_with_beam_postprocess_workspace
