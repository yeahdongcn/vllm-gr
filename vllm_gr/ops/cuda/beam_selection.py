# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Deterministic hierarchical CUDA Top-K for OneRec Beam decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from vllm_gr.config.models import MAX_BEAM_WIDTH, MAX_CANDIDATE_WIDTH
from vllm_gr.ops.cuda.topk_sort_key import unpack_score

# Triton JIT functions intentionally use its untyped DSL values.
# mypy: disable-error-code="misc,no-untyped-def"


LOCAL_TILE_SIZE = 256
MAX_REQUESTS = 48


@triton.jit
def _pack_score_flat(score, flat_index, valid):
    """Pack (score DESC, flat_index ASC) into ascending signed-i64 order."""

    min_i32: tl.constexpr = -2147483648
    score = tl.where(score == score, score, -float("inf"))
    score = tl.where(valid, score, -float("inf"))
    score = tl.where(score == 0.0, 0.0, score)
    score_bits = score.to(tl.int32, bitcast=True)
    sign = score_bits >> 31
    sort_key = tl.where(sign == 0, score_bits ^ -1, score_bits ^ min_i32)
    key64 = sort_key.to(tl.int64) & 0x00000000FFFFFFFF
    # Triton 3.6 type-checks both ``where`` branches before applying a
    # trailing cast.  Promote the live index first so the unsigned-32 sentinel
    # is represented in i64 instead of overflowing the inferred i32 branch.
    tie_key = tl.where(valid, flat_index.to(tl.int64), 0x00000000FFFFFFFF)
    return (key64 << 32) | tie_key


@triton.jit(
    do_not_specialize=[
        "beam_width",
        "candidate_width",
        "select_width",
        "scratch_stride",
    ]
)
def _local_beam_topk_kernel(
    cumulative_logprobs,
    top_logprobs,
    candidate_valid_mask,
    local_scores,
    local_flat_indices,
    beam_width,
    candidate_width,
    select_width,
    scratch_stride,
    HAS_VALID_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
    MASK_INVALID_INDICES: tl.constexpr = False,
    SELECT_BLOCK: tl.constexpr = 0,
    TOPK: tl.constexpr = None,
):
    request = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    candidate_count = beam_width * candidate_width
    flat = tile * BLOCK + offsets
    lane_valid = flat < candidate_count
    parent = flat // candidate_width
    candidate = flat - parent * candidate_width
    row = request * beam_width + parent
    candidate_valid = lane_valid
    if HAS_VALID_MASK:
        candidate_valid = candidate_valid & tl.load(
            candidate_valid_mask + row * candidate_width + candidate,
            mask=lane_valid,
            other=False,
        )
    cumulative = tl.load(
        cumulative_logprobs + row,
        mask=candidate_valid,
        other=-float("inf"),
    )
    candidate_score = tl.load(
        top_logprobs + row * candidate_width + candidate,
        mask=candidate_valid,
        other=-float("inf"),
    )
    score = cumulative + candidate_score
    packed = _pack_score_flat(score, flat, candidate_valid if MASK_INVALID_INDICES else lane_valid)
    if SELECT_BLOCK > 0 and SELECT_BLOCK < BLOCK:
        # Only the best ``select_width`` lanes survive, so a bitonic Top-K of
        # the packed keys is cheaper than sorting the whole list.  Packed keys
        # are unique for live candidates and the retained set/order is
        # identical to the full sort.
        retained = tl.arange(0, SELECT_BLOCK)
        if SELECT_BLOCK == 1:
            # Triton Top-K reduces K=1 to a scalar that its final reshape
            # cannot handle.  The smallest packed key preserves the same order.
            ranked = tl.min(packed, axis=0, keep_dims=True)
        else:
            ranked = ~TOPK(~packed, SELECT_BLOCK)
        output_mask = retained < select_width
        output = request * scratch_stride + tile * select_width + retained
        tl.store(local_scores + output, unpack_score(ranked), mask=output_mask)
        tl.store(
            local_flat_indices + output,
            (ranked & 0x00000000FFFFFFFF).to(tl.int32),
            mask=output_mask,
        )
    else:
        sorted_pairs = tl.sort(packed, descending=False)
        output_mask = offsets < select_width
        output = request * scratch_stride + tile * select_width + offsets
        tl.store(local_scores + output, unpack_score(sorted_pairs), mask=output_mask)
        tl.store(
            local_flat_indices + output,
            (sorted_pairs & 0x00000000FFFFFFFF).to(tl.int32),
            mask=output_mask,
        )


@triton.jit(
    do_not_specialize=[
        "list_count",
        "select_width",
        "scratch_stride",
    ]
)
def _merge_beam_topk_kernel(
    input_scores,
    input_flat_indices,
    output_scores,
    output_flat_indices,
    list_count,
    select_width,
    scratch_stride,
    BLOCK: tl.constexpr,
    FAN_IN: tl.constexpr = 2,
):
    request = tl.program_id(0)
    output_list = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    source_list = output_list * FAN_IN + offsets // select_width
    source_rank = offsets % select_width
    valid = (offsets < FAN_IN * select_width) & (source_list < list_count)
    source = request * scratch_stride + source_list * select_width + source_rank
    score = tl.load(input_scores + source, mask=valid, other=-float("inf"))
    flat = tl.load(input_flat_indices + source, mask=valid, other=-1)
    valid = valid & (flat >= 0)
    packed = _pack_score_flat(score, flat, valid)
    sorted_pairs = tl.sort(packed, descending=False)
    selected_scores = unpack_score(sorted_pairs)
    selected_flat = (sorted_pairs & 0x00000000FFFFFFFF).to(tl.int32)
    output_mask = offsets < select_width
    output = request * scratch_stride + output_list * select_width + offsets
    tl.store(output_scores + output, selected_scores, mask=output_mask)
    tl.store(output_flat_indices + output, selected_flat, mask=output_mask)


@triton.jit(
    do_not_specialize=[
        "beam_width",
        "candidate_width",
        "scratch_stride",
    ]
)
def _group_beam_outputs_kernel(
    selected_scores,
    selected_flat_indices,
    top_tokens,
    out_token_ids,
    out_parent_indices,
    out_scores,
    out_grouped_flat_indices,
    beam_width,
    candidate_width,
    scratch_stride,
    WRITE_FLAT_INDICES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    rank = tl.arange(0, BLOCK)
    candidate_count = beam_width * candidate_width
    selected_offset = request * scratch_stride + rank
    flat = tl.load(selected_flat_indices + selected_offset, mask=rank < beam_width, other=-1)
    valid = (rank < beam_width) & (flat >= 0) & (flat < candidate_count)
    parent = flat // candidate_width

    group_key = (parent.to(tl.int64) << 32) | rank.to(tl.int64)
    group_key = tl.where(valid, group_key, 0x7FFFFFFFFFFFFFFF)
    grouped = tl.sort(group_key, descending=False)
    grouped_rank = (grouped & 0x00000000FFFFFFFF).to(tl.int64)
    grouped_valid = (rank < beam_width) & (grouped_rank < beam_width)
    safe_rank = tl.where(grouped_valid, grouped_rank, 0)
    grouped_flat = tl.load(
        selected_flat_indices + request * scratch_stride + safe_rank,
        mask=grouped_valid,
        other=-1,
    )
    grouped_valid = grouped_valid & (grouped_flat >= 0)
    grouped_parent = grouped_flat // candidate_width
    grouped_score = tl.load(
        selected_scores + request * scratch_stride + safe_rank,
        mask=grouped_valid,
        other=-float("inf"),
    )
    token = tl.load(
        top_tokens + request * candidate_count + grouped_flat,
        mask=grouped_valid,
        other=0,
    )

    output = request * beam_width + rank
    tl.store(out_token_ids + output, token, mask=rank < beam_width)
    tl.store(
        out_parent_indices + output,
        tl.where(grouped_valid, request * beam_width + grouped_parent, -1),
        mask=rank < beam_width,
    )
    tl.store(out_scores + output, grouped_score, mask=rank < beam_width)
    if WRITE_FLAT_INDICES:
        tl.store(
            out_grouped_flat_indices + output,
            grouped_flat,
            mask=rank < beam_width,
        )


@triton.jit(do_not_specialize=["beam_width", "candidate_width"])
def _bootstrap_beam_outputs_kernel(
    cumulative_logprobs,
    top_tokens,
    top_logprobs,
    candidate_valid_mask,
    out_token_ids,
    out_parent_indices,
    out_scores,
    out_flat_indices,
    beam_width,
    candidate_width,
    HAS_VALID_MASK: tl.constexpr,
    COMPACT_INPUT: tl.constexpr,
    WRITE_FLAT_INDICES: tl.constexpr,
    BLOCK: tl.constexpr,
    REUSE_ORDERED_CANDIDATES: tl.constexpr = False,
    GATHER: tl.constexpr = None,
):
    """Expand the first live parent of every request from one beam to W."""

    request = tl.program_id(0)
    rank = tl.arange(0, BLOCK)
    # Canonical prefill produces one RecConstrainedTopK row per request.  The
    # legacy standalone ABI supplies B*W rows and only row 0 of every request
    # is live.  Accept both layouts so serving does not allocate/copy a dummy
    # W*W tensor merely to satisfy the bootstrap step.
    row = request if COMPACT_INPUT else request * beam_width
    # Select across all K candidates, including unordered rows and mask holes.
    candidate = row * candidate_width + rank
    lane_valid = rank < candidate_width
    candidate_valid = lane_valid
    if HAS_VALID_MASK:
        candidate_valid = candidate_valid & tl.load(
            candidate_valid_mask + candidate,
            mask=lane_valid,
            other=False,
        )
    score = tl.load(cumulative_logprobs + row) + tl.load(
        top_logprobs + candidate,
        mask=lane_valid,
        other=-float("inf"),
    )
    packed = _pack_score_flat(score, rank, candidate_valid)
    if REUSE_ORDERED_CANDIDATES:
        # Compare the same packed keys as the fallback sort. Unordered
        # candidates or valid lanes following a stop/masked lane still sort.
        previous = GATHER(packed, tl.maximum(rank - 1, 0), axis=0)
        if tl.sum(((rank > 0) & (packed < previous)).to(tl.int32), axis=0) != 0:
            selected = tl.sort(packed, descending=False)
        else:
            selected = packed
    else:
        selected = tl.sort(packed, descending=False)
    selected_flat = (selected & 0x00000000FFFFFFFF).to(tl.int32)
    selected_valid = (selected_flat >= 0) & (selected_flat < candidate_width)
    token = tl.load(
        top_tokens + row * candidate_width + selected_flat,
        mask=selected_valid,
        other=0,
    )
    output = request * beam_width + rank
    tl.store(out_token_ids + output, token, mask=rank < beam_width)
    tl.store(
        out_parent_indices + output,
        tl.where(selected_valid, request * beam_width, -1),
        mask=rank < beam_width,
    )
    tl.store(out_scores + output, unpack_score(selected), mask=rank < beam_width)
    if WRITE_FLAT_INDICES:
        tl.store(
            out_flat_indices + output,
            tl.where(selected_valid, selected_flat, -1),
            mask=rank < beam_width,
        )


@triton.jit(do_not_specialize=["beam_width", "result_width", "max_steps"])
def _grouped_final_sequence_kernel(
    grouped_sequence,
    group_positions,
    out_sequence,
    beam_width,
    result_width,
    max_steps,
    BLOCK_STEPS: tl.constexpr,
):
    request = tl.program_id(0)
    rank = tl.program_id(1)
    step = tl.arange(0, BLOCK_STEPS)
    group_position = tl.load(group_positions + request * result_width + rank)
    source = (request * beam_width + group_position) * max_steps + step
    output = (request * result_width + rank) * max_steps + step
    mask = step < max_steps
    tl.store(
        out_sequence + output,
        tl.load(grouped_sequence + source, mask=mask, other=0),
        mask=mask,
    )


@triton.jit(do_not_specialize=["beam_width"])
def _parent_count_prefix_kernel(
    grouped_parent_indices,
    out_parent_count_prefix,
    beam_width,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    parent_id = tl.program_id(1)
    rank = tl.arange(0, BLOCK)
    global_parent = tl.load(
        grouped_parent_indices + request * beam_width + rank,
        mask=rank < beam_width,
        other=-1,
    )
    local_parent = global_parent - request * beam_width
    count = tl.sum(
        (rank < beam_width) & (local_parent >= 0) & (local_parent <= parent_id),
        axis=0,
    )
    tl.store(
        out_parent_count_prefix + request * beam_width + parent_id,
        request * beam_width + count,
    )


@triton.jit(
    do_not_specialize=[
        "beam_width",
        "candidate_width",
        "result_width",
        "scratch_stride",
    ]
)
def _final_beam_outputs_kernel(
    selected_scores,
    selected_flat_indices,
    top_tokens,
    out_token_ids,
    out_parent_indices,
    out_scores,
    beam_width,
    candidate_width,
    result_width,
    scratch_stride,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    rank = tl.arange(0, BLOCK)
    selected_offset = request * scratch_stride + rank
    flat = tl.load(selected_flat_indices + selected_offset, mask=rank < result_width, other=-1)
    valid = (rank < result_width) & (flat >= 0) & (flat < beam_width * candidate_width)
    parent = flat // candidate_width
    token = tl.load(
        top_tokens + request * beam_width * candidate_width + flat,
        mask=valid,
        other=0,
    )
    score = tl.load(selected_scores + selected_offset, mask=valid, other=-float("inf"))
    output = request * result_width + rank
    tl.store(out_token_ids + output, token, mask=rank < result_width)
    tl.store(
        out_parent_indices + output,
        tl.where(valid, request * beam_width + parent, -1),
        mask=rank < result_width,
    )
    tl.store(out_scores + output, score, mask=rank < result_width)


@triton.jit(
    do_not_specialize=[
        "beam_width",
        "output_width",
        "max_steps",
        "current_step",
    ]
)
def _write_beam_sequence_kernel(
    sequence,
    out_token_ids,
    out_parent_indices,
    out_sequence,
    beam_width,
    output_width,
    max_steps,
    current_step,
    BLOCK_STEPS: tl.constexpr,
):
    request = tl.program_id(0)
    child = tl.program_id(1)
    step = tl.arange(0, BLOCK_STEPS)
    output_row = request * output_width + child
    global_parent = tl.load(out_parent_indices + output_row)
    local_parent = global_parent - request * beam_width
    valid_parent = (local_parent >= 0) & (local_parent < beam_width)
    history_mask = (step < max_steps) & (step < current_step) & valid_parent
    history = tl.load(
        sequence + (request * beam_width + local_parent) * max_steps + step,
        mask=history_mask,
        other=0,
    )
    token = tl.load(out_token_ids + output_row)
    value = tl.where(step < current_step, history, tl.where(step == current_step, token, 0))
    output = (request * output_width + child) * max_steps + step
    tl.store(out_sequence + output, value, mask=step < max_steps)


@dataclass(frozen=True, slots=True)
class BeamSelectionScratch:
    """Caller-owned ping-pong storage for hierarchical Beam Top-K."""

    local_scores: torch.Tensor
    local_flat_indices: torch.Tensor
    merge_scores: torch.Tensor
    merge_flat_indices: torch.Tensor


def _beam_selection_tiling(
    beam_width: int,
    candidate_width: int,
    selection_width: int,
) -> tuple[int, int]:
    """Return ``(list_count, local_block)`` for the hierarchical Top-K.

    ``local_block`` is the width of one local list, so it must be at least
    ``selection_width``: ``_local_beam_topk_kernel`` writes
    ``selection_width`` slots into each stride-``selection_width`` list, and a
    narrower list would leave the remaining slots untouched.  ``LOCAL_TILE_SIZE``
    is therefore only the *minimum* list width — a selection wider than it (a
    beam width above 256) widens the list instead of silently truncating.
    Both call sites (allocation and launch) use this one definition so the
    prewarmed scratch and the kernel grid can never disagree.
    """
    candidate_count = beam_width * candidate_width
    tile_capacity = triton.next_power_of_2(max(LOCAL_TILE_SIZE, selection_width))
    # ``max(..., selection_width)`` keeps the invariant when there are fewer
    # candidates than winners (a step-0 bootstrap with K < W): the lanes past
    # ``candidate_count`` still get written, as ``-inf``/``-1``, instead of
    # leaving uninitialized scratch for the merge stage to read.
    local_block = triton.next_power_of_2(max(min(candidate_count, tile_capacity), selection_width))
    list_count = (candidate_count + local_block - 1) // local_block
    return list_count, local_block


def beam_selection_scratch_slots(
    beam_width: int,
    candidate_width: int,
    selection_width: int,
) -> int:
    # The tiling uses integer bit arithmetic (``next_power_of_2``), so a float
    # width would otherwise fail deep inside it with an opaque operand error.
    for name, value in (
        ("beam_width", beam_width),
        ("candidate_width", candidate_width),
        ("selection_width", selection_width),
    ):
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an int")
    if min(beam_width, candidate_width, selection_width) <= 0:
        raise ValueError("beam, candidate, and selection widths must be positive")
    list_count, _ = _beam_selection_tiling(beam_width, candidate_width, selection_width)
    return list_count * selection_width


def allocate_beam_selection_scratch(
    batch_size: int,
    beam_width: int,
    candidate_width: int,
    selection_width: int,
    device: torch.device,
) -> BeamSelectionScratch:
    """Allocate convenience scratch; graph/runtime code should retain it."""

    slots = beam_selection_scratch_slots(beam_width, candidate_width, selection_width)
    return BeamSelectionScratch(
        local_scores=torch.empty((batch_size, slots), dtype=torch.float32, device=device),
        local_flat_indices=torch.empty((batch_size, slots), dtype=torch.int32, device=device),
        merge_scores=torch.empty((batch_size, slots), dtype=torch.float32, device=device),
        merge_flat_indices=torch.empty((batch_size, slots), dtype=torch.int32, device=device),
    )


def _require_cuda_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_beam_inputs(
    cumulative_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    candidate_valid_mask: torch.Tensor | None,
) -> tuple[int, int, int, int]:
    if isinstance(current_step, bool) or not isinstance(current_step, int):
        raise TypeError("current_step must be an int")
    if cumulative_logprobs.device.type != "cuda":
        raise ValueError("CUDA Beam selection requires CUDA tensors")
    device = cumulative_logprobs.device
    if cumulative_logprobs.dtype != torch.float32:
        raise TypeError("cumulative_logprobs must use torch.float32")
    if cumulative_logprobs.ndim != 2 or cumulative_logprobs.shape[1] != 1:
        raise ValueError("cumulative_logprobs must have shape [B, 1] or [B*W, 1]")
    if not cumulative_logprobs.is_contiguous():
        raise ValueError("cumulative_logprobs must be contiguous")
    if sequence.dtype != torch.int32 or sequence.ndim != 3 or not sequence.is_contiguous():
        raise ValueError("sequence must be contiguous int32 [B, W, max_steps]")
    if sequence.device != device:
        raise ValueError("sequence must be on the cumulative_logprobs device")
    batch_size, beam_width, max_steps = sequence.shape
    if not 1 <= batch_size <= MAX_REQUESTS:
        raise ValueError(f"batch size must be in [1, {MAX_REQUESTS}]")
    if not 1 <= beam_width <= MAX_BEAM_WIDTH:
        raise ValueError(f"beam width must be in [1, {MAX_BEAM_WIDTH}]")
    if current_step < 0 or current_step >= max_steps:
        raise ValueError("current_step is outside the fixed sequence extent")
    compact_bootstrap = current_step == 0 and cumulative_logprobs.shape[0] == batch_size
    expected_rows = batch_size if compact_bootstrap else batch_size * beam_width
    if cumulative_logprobs.shape[0] != expected_rows:
        raise ValueError(
            "cumulative_logprobs rows must equal B for compact step-0 bootstrap or B*W otherwise"
        )
    if top_tokens.dtype != torch.int32 or top_tokens.ndim != 2:
        raise TypeError("top_tokens must be a 2-D torch.int32 tensor")
    if top_logprobs.dtype != torch.float32 or top_logprobs.ndim != 2:
        raise TypeError("top_logprobs must be a 2-D torch.float32 tensor")
    if top_tokens.device != device or top_logprobs.device != device:
        raise ValueError("candidate tensors must be on the cumulative_logprobs device")
    if not top_tokens.is_contiguous() or not top_logprobs.is_contiguous():
        raise ValueError("candidate tensors must be contiguous")
    if top_tokens.shape != top_logprobs.shape:
        raise ValueError("top_tokens and top_logprobs must have equal shapes")
    candidate_width = top_tokens.shape[1]
    if top_tokens.shape[0] != expected_rows:
        raise ValueError(
            "candidate rows must equal B for compact step-0 bootstrap or B*W otherwise"
        )
    if not 1 <= candidate_width <= MAX_CANDIDATE_WIDTH:
        raise ValueError(f"candidate width must be in [1, {MAX_CANDIDATE_WIDTH}]")
    if candidate_valid_mask is not None:
        _require_cuda_tensor(
            "candidate_valid_mask",
            candidate_valid_mask,
            device=device,
            dtype=torch.bool,
            shape=tuple(top_tokens.shape),
        )
    return batch_size, beam_width, candidate_width, max_steps


def _validate_scratch(
    scratch: BeamSelectionScratch,
    *,
    device: torch.device,
    batch_size: int,
    required_slots: int,
) -> int:
    tensors = (
        ("local_scores", scratch.local_scores, torch.float32),
        ("local_flat_indices", scratch.local_flat_indices, torch.int32),
        ("merge_scores", scratch.merge_scores, torch.float32),
        ("merge_flat_indices", scratch.merge_flat_indices, torch.int32),
    )
    stride = scratch.local_scores.shape[1] if scratch.local_scores.ndim == 2 else -1
    for name, tensor, dtype in tensors:
        if tensor.device != device or tensor.dtype != dtype:
            raise ValueError(f"{name} must be {dtype} on {device}")
        if tensor.ndim != 2 or tensor.shape[0] < batch_size or tensor.shape[1] < required_slots:
            raise ValueError(f"{name} must be [B, >= {required_slots}], got {tuple(tensor.shape)}")
        if tensor.shape[1] != stride or not tensor.is_contiguous():
            raise ValueError("all Beam scratch tensors must share one contiguous stride")
    return int(stride)


def _hierarchical_topk_out(
    cumulative_logprobs: torch.Tensor,
    top_logprobs: torch.Tensor,
    candidate_valid_mask: torch.Tensor | None,
    batch_size: int,
    beam_width: int,
    candidate_width: int,
    selection_width: int,
    scratch: BeamSelectionScratch,
    *,
    mask_invalid_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    list_count, local_block = _beam_selection_tiling(beam_width, candidate_width, selection_width)
    required_slots = list_count * selection_width
    scratch_stride = _validate_scratch(
        scratch,
        device=cumulative_logprobs.device,
        batch_size=batch_size,
        required_slots=required_slots,
    )
    validity = candidate_valid_mask if candidate_valid_mask is not None else top_logprobs
    select_block = triton.next_power_of_2(selection_width)
    _local_beam_topk_kernel[(batch_size, list_count)](
        cumulative_logprobs,
        top_logprobs,
        validity,
        scratch.local_scores,
        scratch.local_flat_indices,
        beam_width,
        candidate_width,
        selection_width,
        scratch_stride,
        HAS_VALID_MASK=candidate_valid_mask is not None,
        BLOCK=local_block,
        MASK_INVALID_INDICES=mask_invalid_indices,
        # When a list is wider than the selection, keep only the winners
        # (bitonic Top-K) instead of sorting every lane.  0 keeps the plain
        # sort for the square W == K case.
        SELECT_BLOCK=select_block if select_block < local_block and hasattr(tl, "topk") else 0,
        TOPK=getattr(tl, "topk", None),
        # A wider selection (a beam width above 256) makes one local list span
        # more than 256 lanes; add a tier so the per-thread register load of
        # the in-block sort does not explode.  Widths <= 256 are unchanged.
        num_warps=16 if local_block >= 1024 else (8 if local_block >= 128 else 4),
    )
    input_scores = scratch.local_scores
    input_indices = scratch.local_flat_indices
    output_scores = scratch.merge_scores
    output_indices = scratch.merge_flat_indices
    # Merge several lists per level while keeping one CTA's candidate set
    # bounded at 2048 lanes: 8-way for a 256-wide selection, 4-way at 512,
    # binary at 1024.  Fewer levels means fewer launches and fewer full passes
    # over the survivor set.
    fan_in = max(2, min(8, 2048 // select_block))
    merge_block = triton.next_power_of_2(fan_in * selection_width)
    while list_count > 1:
        next_count = (list_count + fan_in - 1) // fan_in
        _merge_beam_topk_kernel[(batch_size, next_count)](
            input_scores,
            input_indices,
            output_scores,
            output_indices,
            list_count,
            selection_width,
            scratch_stride,
            BLOCK=merge_block,
            FAN_IN=fan_in,
            # Same tier rationale as the local list: one merge spans
            # ``fan_in * selection_width`` lanes, i.e. 1024 at a beam width of
            # 512 and 2048 at 1024.  Widths <= 256 are unchanged.
            num_warps=16 if merge_block >= 1024 else (8 if merge_block >= 256 else 4),
        )
        input_scores, output_scores = output_scores, input_scores
        input_indices, output_indices = output_indices, input_indices
        list_count = next_count
    return input_scores, input_indices, scratch_stride


def bootstrap_beam_outputs_out(
    cumulative_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
    candidate_valid_mask: torch.Tensor | None,
    out_token_ids: torch.Tensor,
    out_parent_indices: torch.Tensor,
    out_scores: torch.Tensor,
    out_grouped_flat_indices: torch.Tensor | None,
    *,
    batch_size: int,
    beam_width: int,
    candidate_width: int,
    compact_input: bool,
    reuse_ordered_candidates: bool = False,
    num_warps: int = 4,
) -> None:
    """Expand the first parent of each request into caller-owned Beam outputs."""
    gather = getattr(tl, "gather", None)
    _bootstrap_beam_outputs_kernel[(batch_size,)](
        cumulative_logprobs,
        top_tokens,
        top_logprobs,
        candidate_valid_mask if candidate_valid_mask is not None else top_logprobs,
        out_token_ids,
        out_parent_indices,
        out_scores,
        out_grouped_flat_indices if out_grouped_flat_indices is not None else out_parent_indices,
        beam_width,
        candidate_width,
        HAS_VALID_MASK=candidate_valid_mask is not None,
        COMPACT_INPUT=compact_input,
        WRITE_FLAT_INDICES=out_grouped_flat_indices is not None,
        BLOCK=triton.next_power_of_2(max(beam_width, candidate_width)),
        REUSE_ORDERED_CANDIDATES=reuse_ordered_candidates and gather is not None,
        GATHER=gather,
        num_warps=num_warps,
    )


def beam_search_group_cuda_out(
    cumulative_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    top_k: int,
    out_token_ids: torch.Tensor,
    out_parent_indices: torch.Tensor,
    out_scores: torch.Tensor,
    out_parent_count_prefix: torch.Tensor,
    out_sequence: torch.Tensor,
    local_scores: torch.Tensor,
    local_flat_indices: torch.Tensor,
    merge_scores: torch.Tensor,
    merge_flat_indices: torch.Tensor,
    candidate_valid_mask: torch.Tensor | None = None,
    out_grouped_flat_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write W survivors; the historical CUDA ``top_k`` argument denotes K."""

    # Preserve the CUDA keyword: historically top_k denotes K, not survivors W.
    candidate_width = top_k
    batch_size, beam_width, tensor_candidate_width, max_steps = _validate_beam_inputs(
        cumulative_logprobs,
        top_tokens,
        top_logprobs,
        sequence,
        current_step,
        candidate_valid_mask,
    )
    if current_step not in (0, 1, 2):
        raise ValueError("CUDA BeamSearchGroup supports current_step 0, 1, or 2")
    # ``candidate_width`` is the per-row candidate count K (columns per beam
    # row); it is independent of the beam width W (rows per request).  Unlike
    # the torch entry point, this operator emits exactly W survivors, so there
    # is no separate selection width to pass.
    if isinstance(candidate_width, bool) or not isinstance(candidate_width, int):
        raise TypeError("candidate_width must be an int")
    if candidate_width != tensor_candidate_width:
        raise ValueError("CUDA BeamSearchGroup requires candidate_width == top_tokens.shape[1]")
    device = cumulative_logprobs.device
    output_specs = (
        ("out_token_ids", out_token_ids, torch.int32, (batch_size, beam_width)),
        ("out_parent_indices", out_parent_indices, torch.int32, (batch_size, beam_width)),
        ("out_scores", out_scores, torch.float32, (batch_size, beam_width)),
        (
            "out_parent_count_prefix",
            out_parent_count_prefix,
            torch.int32,
            (batch_size, beam_width),
        ),
        (
            "out_sequence",
            out_sequence,
            torch.int32,
            (batch_size, beam_width, max_steps),
        ),
    )
    for name, tensor, dtype, shape in output_specs:
        _require_cuda_tensor(name, tensor, device=device, dtype=dtype, shape=shape)
    if out_grouped_flat_indices is not None:
        _require_cuda_tensor(
            "out_grouped_flat_indices",
            out_grouped_flat_indices,
            device=device,
            dtype=torch.int32,
            shape=(batch_size, beam_width),
        )
    if out_sequence.data_ptr() == sequence.data_ptr():
        raise ValueError("out_sequence must not alias sequence")

    scratch = BeamSelectionScratch(
        local_scores,
        local_flat_indices,
        merge_scores,
        merge_flat_indices,
    )
    block = triton.next_power_of_2(beam_width)
    if current_step == 0:
        # Bootstrap launches no hierarchical Top-K kernel; retain a one-slot
        # scratch ABI for the caller-owned/out variant without allocating W-by-W
        # temporary storage.
        required_slots = 1
        _validate_scratch(
            scratch,
            device=device,
            batch_size=batch_size,
            required_slots=required_slots,
        )
        bootstrap_beam_outputs_out(
            cumulative_logprobs,
            top_tokens,
            top_logprobs,
            candidate_valid_mask,
            out_token_ids,
            out_parent_indices,
            out_scores,
            out_grouped_flat_indices,
            batch_size=batch_size,
            beam_width=beam_width,
            candidate_width=candidate_width,
            compact_input=cumulative_logprobs.shape[0] == batch_size,
            num_warps=8 if block >= 128 else 4,
        )
        _parent_count_prefix_kernel[(batch_size, beam_width)](
            out_parent_indices,
            out_parent_count_prefix,
            beam_width,
            BLOCK=block,
            num_warps=8 if block >= 128 else 4,
        )
        step_block = triton.next_power_of_2(max_steps)
        _write_beam_sequence_kernel[(batch_size, beam_width)](
            sequence,
            out_token_ids,
            out_parent_indices,
            out_sequence,
            beam_width,
            beam_width,
            max_steps,
            current_step,
            BLOCK_STEPS=step_block,
            num_warps=4,
        )
        return (
            out_token_ids,
            out_parent_indices,
            out_scores,
            out_parent_count_prefix,
            out_sequence,
        )

    selected_scores, selected_indices, scratch_stride = _hierarchical_topk_out(
        cumulative_logprobs,
        top_logprobs,
        candidate_valid_mask,
        batch_size,
        beam_width,
        candidate_width,
        beam_width,
        scratch,
    )
    _group_beam_outputs_kernel[(batch_size,)](
        selected_scores,
        selected_indices,
        top_tokens,
        out_token_ids,
        out_parent_indices,
        out_scores,
        (out_grouped_flat_indices if out_grouped_flat_indices is not None else out_parent_indices),
        beam_width,
        candidate_width,
        scratch_stride,
        WRITE_FLAT_INDICES=out_grouped_flat_indices is not None,
        BLOCK=block,
        num_warps=8 if block >= 128 else 4,
    )
    _parent_count_prefix_kernel[(batch_size, beam_width)](
        out_parent_indices,
        out_parent_count_prefix,
        beam_width,
        BLOCK=block,
        num_warps=8 if block >= 128 else 4,
    )
    step_block = triton.next_power_of_2(max_steps)
    _write_beam_sequence_kernel[(batch_size, beam_width)](
        sequence,
        out_token_ids,
        out_parent_indices,
        out_sequence,
        beam_width,
        beam_width,
        max_steps,
        current_step,
        BLOCK_STEPS=step_block,
        num_warps=4,
    )
    return (
        out_token_ids,
        out_parent_indices,
        out_scores,
        out_parent_count_prefix,
        out_sequence,
    )


def beam_search_group_cuda(
    cumulative_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    top_k: int,
    candidate_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate W survivors; the historical CUDA ``top_k`` argument denotes K."""

    # Preserve the CUDA keyword: historically top_k denotes K, not survivors W.
    candidate_width = top_k
    batch_size, beam_width, tensor_candidate_width, max_steps = _validate_beam_inputs(
        cumulative_logprobs,
        top_tokens,
        top_logprobs,
        sequence,
        current_step,
        candidate_valid_mask,
    )
    if current_step not in (0, 1, 2):
        raise ValueError("CUDA BeamSearchGroup supports current_step 0, 1, or 2")
    # ``candidate_width`` is the per-row candidate count K (columns per beam
    # row); it is independent of the beam width W (rows per request).  Unlike
    # the torch entry point, this operator emits exactly W survivors, so there
    # is no separate selection width to pass.
    if isinstance(candidate_width, bool) or not isinstance(candidate_width, int):
        raise TypeError("candidate_width must be an int")
    if candidate_width != tensor_candidate_width:
        raise ValueError("CUDA BeamSearchGroup requires candidate_width == top_tokens.shape[1]")
    device = cumulative_logprobs.device
    scratch = allocate_beam_selection_scratch(
        batch_size,
        1 if current_step == 0 else beam_width,
        1 if current_step == 0 else candidate_width,
        1 if current_step == 0 else beam_width,
        device,
    )
    outputs = (
        torch.empty((batch_size, beam_width), dtype=torch.int32, device=device),
        torch.empty((batch_size, beam_width), dtype=torch.int32, device=device),
        torch.empty((batch_size, beam_width), dtype=torch.float32, device=device),
        torch.empty((batch_size, beam_width), dtype=torch.int32, device=device),
        torch.empty(
            (batch_size, beam_width, max_steps),
            dtype=torch.int32,
            device=device,
        ),
    )
    return beam_search_group_cuda_out(
        cumulative_logprobs,
        top_tokens,
        top_logprobs,
        sequence,
        current_step,
        candidate_width,
        *outputs,
        scratch.local_scores,
        scratch.local_flat_indices,
        scratch.merge_scores,
        scratch.merge_flat_indices,
        candidate_valid_mask,
    )


def beam_search_rec_final_select_cuda_out(
    cumulative_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    result_width: int,
    out_token_ids: torch.Tensor,
    out_parent_indices: torch.Tensor,
    out_scores: torch.Tensor,
    out_sequence: torch.Tensor,
    local_scores: torch.Tensor,
    local_flat_indices: torch.Tensor,
    merge_scores: torch.Tensor,
    merge_flat_indices: torch.Tensor,
    candidate_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write score-ordered final Beam results into caller-owned CUDA tensors."""

    batch_size, beam_width, candidate_width, max_steps = _validate_beam_inputs(
        cumulative_logprobs,
        top_tokens,
        top_logprobs,
        sequence,
        current_step,
        candidate_valid_mask,
    )
    if current_step not in (0, 1, 2):
        raise ValueError("CUDA FinalSelect supports current_step 0, 1, or 2")
    if isinstance(result_width, bool) or not isinstance(result_width, int):
        raise TypeError("result_width must be an int")
    if not 1 <= result_width <= beam_width:
        raise ValueError("CUDA FinalSelect requires result_width in [1, beam_width]")
    device = cumulative_logprobs.device
    output_specs = (
        ("out_token_ids", out_token_ids, torch.int32, (batch_size * result_width, 1)),
        (
            "out_parent_indices",
            out_parent_indices,
            torch.int32,
            (batch_size * result_width, 1),
        ),
        ("out_scores", out_scores, torch.float32, (batch_size * result_width, 1)),
        (
            "out_sequence",
            out_sequence,
            torch.int32,
            (batch_size, result_width, max_steps),
        ),
    )
    for name, tensor, dtype, shape in output_specs:
        _require_cuda_tensor(name, tensor, device=device, dtype=dtype, shape=shape)
    if out_sequence.data_ptr() == sequence.data_ptr():
        raise ValueError("out_sequence must not alias sequence")

    scratch = BeamSelectionScratch(
        local_scores,
        local_flat_indices,
        merge_scores,
        merge_flat_indices,
    )
    selected_scores, selected_indices, scratch_stride = _hierarchical_topk_out(
        cumulative_logprobs,
        top_logprobs,
        candidate_valid_mask,
        batch_size,
        beam_width,
        candidate_width,
        result_width,
        scratch,
    )
    block = triton.next_power_of_2(result_width)
    _final_beam_outputs_kernel[(batch_size,)](
        selected_scores,
        selected_indices,
        top_tokens,
        out_token_ids,
        out_parent_indices,
        out_scores,
        beam_width,
        candidate_width,
        result_width,
        scratch_stride,
        BLOCK=block,
        num_warps=8 if block >= 128 else 4,
    )
    step_block = triton.next_power_of_2(max_steps)
    _write_beam_sequence_kernel[(batch_size, result_width)](
        sequence,
        out_token_ids,
        out_parent_indices,
        out_sequence,
        beam_width,
        result_width,
        max_steps,
        current_step,
        BLOCK_STEPS=step_block,
        num_warps=4,
    )
    return out_token_ids, out_parent_indices, out_scores, out_sequence


def beam_search_rec_final_select_cuda(
    cumulative_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    result_width: int,
    candidate_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocating convenience wrapper for CUDA OneRec FinalSelect."""

    batch_size, beam_width, candidate_width, max_steps = _validate_beam_inputs(
        cumulative_logprobs,
        top_tokens,
        top_logprobs,
        sequence,
        current_step,
        candidate_valid_mask,
    )
    if current_step not in (0, 1, 2):
        raise ValueError("CUDA FinalSelect supports current_step 0, 1, or 2")
    if isinstance(result_width, bool) or not isinstance(result_width, int):
        raise TypeError("result_width must be an int")
    if not 1 <= result_width <= beam_width:
        raise ValueError("CUDA FinalSelect requires result_width in [1, beam_width]")
    device = cumulative_logprobs.device
    scratch = allocate_beam_selection_scratch(
        batch_size, beam_width, candidate_width, result_width, device
    )
    outputs = (
        torch.empty((batch_size * result_width, 1), dtype=torch.int32, device=device),
        torch.empty((batch_size * result_width, 1), dtype=torch.int32, device=device),
        torch.empty((batch_size * result_width, 1), dtype=torch.float32, device=device),
        torch.empty(
            (batch_size, result_width, max_steps),
            dtype=torch.int32,
            device=device,
        ),
    )
    return beam_search_rec_final_select_cuda_out(
        cumulative_logprobs,
        top_tokens,
        top_logprobs,
        sequence,
        current_step,
        result_width,
        *outputs,
        scratch.local_scores,
        scratch.local_flat_indices,
        scratch.merge_scores,
        scratch.merge_flat_indices,
        candidate_valid_mask,
    )


__all__ = [
    "BeamSelectionScratch",
    "LOCAL_TILE_SIZE",
    "MAX_BEAM_WIDTH",
    "MAX_CANDIDATE_WIDTH",
    "MAX_REQUESTS",
    "allocate_beam_selection_scratch",
    "beam_search_group_cuda",
    "beam_search_group_cuda_out",
    "beam_search_rec_final_select_cuda",
    "beam_search_rec_final_select_cuda_out",
    "beam_selection_scratch_slots",
]
