# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA constrained Top-K with table lookup and gather fused in Triton."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm_gr.config.models import MAX_CONSTRAINT_CANDIDATE_WIDTH
from vllm_gr.constraints.scoring import INVALID_LOGIT

# Triton JIT functions intentionally use its untyped DSL values.
# mypy: disable-error-code="misc,no-untyped-def"


# Constraint-table kernels retain their independent output capacity.
MAX_TOP_K = MAX_CONSTRAINT_CANDIDATE_WIDTH
MAX_SORT_BLOCK = 65536


def validate_cuda_sort_capacity(
    first_token_count: int,
    max_prefix1_degree: int,
    max_prefix2_degree: int,
) -> None:
    """Validate the table-wide sort extents required by the CUDA kernel.

    Unchecked Worker launches rely on this invariant being established once
    when CUDA tables are loaded, rather than repeating structural table
    validation on every sampling step.
    """

    for name, candidate_count in (
        ("first_token_ids", first_token_count),
        ("max_prefix1_degree", max_prefix1_degree),
        ("max_prefix2_degree", max_prefix2_degree),
    ):
        if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
            raise TypeError(f"{name} candidate count must be an int")
        if candidate_count < 0:
            raise ValueError(f"{name} candidate count must be non-negative")
        block = triton.next_power_of_2(max(candidate_count, 1))
        if block > MAX_SORT_BLOCK:
            raise ValueError(
                f"{name} candidate degree {candidate_count} requires sort block "
                f"{block}, above supported {MAX_SORT_BLOCK}"
            )


@triton.jit(
    do_not_specialize=[
        "logits_vocab_size",
        "table_vocab_size",
        "vocab_offset",
        "pair_count",
        "first_count",
        "prefix1_value_count",
        "prefix2_value_count",
        "logits_row_stride",
        "logits_vocab_stride",
        "prefix_stride",
        "temperature_count",
        "top_k",
    ]
)
def _fused_topk_kernel(
    logits,
    prefixes,
    trie_ids,
    first_offsets,
    first_ids,
    prefix1_offsets,
    prefix1_values,
    pair_keys,
    prefix2_offsets,
    prefix2_values,
    temperatures,
    output_ids,
    output_logprobs,
    output_wire_logprobs,
    output_valid_mask,
    valid_counts,
    prefill_status,
    prefill_payload,
    logits_vocab_size,
    table_vocab_size,
    vocab_offset,
    pair_count,
    first_count,
    prefix1_value_count,
    prefix2_value_count,
    logits_row_stride,
    logits_vocab_stride,
    prefix_stride,
    temperature_count,
    top_k,
    WRITE_WIRE_LOGPROBS: tl.constexpr,
    PACK_PREFILL_RESULT: tl.constexpr,
    TRIE_COUNT: tl.constexpr,
    step: tl.constexpr,
    device_step,
    DEVICE_STEP: tl.constexpr,
    BLOCK: tl.constexpr,
    OUTPUT_BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    WINDOW_BLOCK: tl.constexpr,
):
    if DEVICE_STEP:
        step = tl.minimum(tl.maximum(tl.load(device_step), 0), 2)
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    begin = tl.full((), 0, tl.int64)
    degree = tl.cast(first_count, tl.int64)
    root = tl.full((), 0, tl.int64)
    if TRIE_COUNT > 1:
        # Checked entrypoints or Worker admission already validated root IDs.
        root = tl.load(trie_ids + row).to(tl.int64)
        begin = tl.load(first_offsets + root).to(tl.int64)
        end = tl.load(first_offsets + root + 1).to(tl.int64)
        degree = end - begin
    if step == 1:
        first = tl.load(prefixes + row * prefix_stride).to(tl.int64)
        prefix_valid = (first >= 0) & (first < table_vocab_size)
        safe_first = tl.where(prefix_valid, first, 0)
        if TRIE_COUNT > 1:
            safe_first = tl.where(prefix_valid, root * table_vocab_size + safe_first, 0)
        begin = tl.load(prefix1_offsets + safe_first).to(tl.int64)
        end = tl.load(prefix1_offsets + safe_first + 1).to(tl.int64)
        degree = tl.where(prefix_valid, end - begin, 0)
    elif step == 2:
        first = tl.load(prefixes + row * prefix_stride).to(tl.int64)
        second = tl.load(prefixes + row * prefix_stride + 1).to(tl.int64)
        prefix_valid = (
            (first >= 0) & (first < table_vocab_size) & (second >= 0) & (second < table_vocab_size)
        )
        query = first * table_vocab_size + second
        if TRIE_COUNT > 1:
            query += root * table_vocab_size * table_vocab_size
        low = 0
        high = pair_count
        for _ in tl.static_range(0, 32):
            active = low < high
            mid = (low + high) // 2
            safe_mid = tl.maximum(0, tl.minimum(mid, pair_count - 1))
            key = tl.load(pair_keys + safe_mid, mask=active & (pair_count > 0), other=0)
            take_right = active & (key < query)
            low = tl.where(take_right, mid + 1, low)
            high = tl.where(active & ~take_right, mid, high)
        found_pos = tl.maximum(0, tl.minimum(low, pair_count - 1))
        found_key = tl.load(
            pair_keys + found_pos,
            mask=(pair_count > 0) & (low < pair_count),
            other=-1,
        )
        found = prefix_valid & (pair_count > 0) & (low < pair_count) & (found_key == query)
        safe_pos = tl.where(found, found_pos, 0)
        begin = tl.load(prefix2_offsets + safe_pos, mask=found, other=0).to(tl.int64)
        end = tl.load(prefix2_offsets + safe_pos + 1, mask=found, other=0).to(tl.int64)
        degree = tl.where(found, end - begin, 0)

    candidate_mask = offsets < degree
    if step == 0:
        # Keep single-Trie addressing independent of device-driven prefix offsets.
        first_begin = begin if TRIE_COUNT > 1 else 0
        token = tl.load(first_ids + first_begin + offsets, mask=candidate_mask, other=0)
    elif step == 1:
        token = tl.load(prefix1_values + begin + offsets, mask=candidate_mask, other=0)
    else:
        token = tl.load(prefix2_values + begin + offsets, mask=candidate_mask, other=0)
    local_token = token.to(tl.int64) - vocab_offset
    token_in_window = (local_token >= 0) & (local_token < logits_vocab_size)
    candidate_mask = candidate_mask & token_in_window
    valid_degree = tl.sum(candidate_mask, axis=0)
    score = tl.load(
        logits + row * logits_row_stride + local_token * logits_vocab_stride,
        mask=candidate_mask,
        other=-float("inf"),
    ).to(tl.float32)
    temperature_index = tl.where(temperature_count == 1, 0, row)
    temperature = tl.load(
        temperatures + temperature_index,
        mask=temperature_count > 0,
        other=1.0,
    )
    temperature = tl.where(temperature == 0.0, 1.0, temperature)
    score = score / temperature
    score = tl.where(score == score, score, -float("inf"))
    row_max = tl.max(score, axis=0)
    normalizable = (valid_degree > 0) & (row_max != -float("inf"))

    # Full-window logsumexp over the complete logits window so candidate
    # scores are comparable across parent rows (the scoring contract).  The
    # kernel walks the window once per row with a numerically stable online
    # reduction: a running max rescales the running exp-sum, so every window
    # element is loaded and divided exactly once instead of once per pass.
    full_denominator = 0.0
    if valid_degree > 0:
        window_offsets = tl.arange(0, WINDOW_BLOCK)
        n_chunks = tl.cdiv(logits_vocab_size, WINDOW_BLOCK)
        window_max = tl.full((), -float("inf"), tl.float32)
        window_sum = tl.zeros((), tl.float32)
        for c in tl.range(0, n_chunks):
            idx = c * WINDOW_BLOCK + window_offsets
            mask = idx < logits_vocab_size
            v = tl.load(
                logits + row * logits_row_stride + idx * logits_vocab_stride,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            v = v / temperature
            # NaN logits intentionally stay NaN: they propagate through the
            # running exp-sum into a NaN logsumexp, surfacing the engine's
            # protocol error (beam_device_error) exactly as +inf logits do.
            chunk_max = tl.max(v, axis=0)
            new_max = tl.maximum(window_max, chunk_max)
            # A leading all -inf chunk must not inject NaN into the running
            # exp-sum: exp(-inf - -inf) is NaN, so mask it out (an all -inf
            # window is detected by the final ``window_max`` guard anyway).
            window_sum = window_sum * tl.where(
                window_max == -float("inf"),
                0.0,
                tl.exp(window_max - new_max),
            ) + tl.sum(
                tl.where(
                    mask & (new_max != -float("inf")),
                    tl.exp(v - new_max),
                    0.0,
                ),
                axis=0,
            )
            window_max = new_max
        full_denominator = tl.where(
            window_max != -float("inf"),
            window_max + tl.log(window_sum),
            0.0,
        )

    denominator = tl.where(
        normalizable,
        full_denominator,
        0.0,
    )

    # Pack the exact fp32 ordering and the table offset into one sortable i64.
    # Ascending signed-i64 order is (score DESC, candidate offset ASC).
    min_i32: tl.constexpr = -2147483648
    canonical_score = tl.where(score == 0.0, 0.0, score)
    score_bits = canonical_score.to(tl.int32, bitcast=True)
    sign = score_bits >> 31
    sort_key = tl.where(sign == 0, score_bits ^ -1, score_bits ^ min_i32)
    sort_key = tl.where(candidate_mask, sort_key, 0x7F800000)
    key64 = sort_key.to(tl.int64) & 0x00000000FFFFFFFF
    # Triton 3.6 rejects 0xFFFFFFFF when ``where`` first infers an i32 result;
    # promote the table offset before selecting the i64 sentinel.
    tie_key = tl.where(candidate_mask, offsets.to(tl.int64), 0x00000000FFFFFFFF)
    packed = (key64 << 32) | tie_key
    # Complement reverses signed-i64 order, retaining the exact score/tie key.
    # Select only the output bucket instead of sorting the entire candidate row.
    if OUTPUT_BLOCK == 1:
        sorted_pairs = tl.min(packed, axis=0).expand_dims(0)
    elif TOPK is not None:
        sorted_pairs = ~TOPK(~packed, OUTPUT_BLOCK)
    else:
        sorted_pairs = tl.sort(packed, descending=False)
    offsets = tl.arange(0, sorted_pairs.shape[0])
    selected_offsets = (sorted_pairs & 0x00000000FFFFFFFF).to(tl.int64)
    sorted_keys = ((sorted_pairs >> 32) & 0x00000000FFFFFFFF).to(tl.int32)
    sorted_sign = sorted_keys >> 31
    sorted_bits = tl.where(
        sorted_sign < 0,
        sorted_keys ^ -1,
        sorted_keys ^ min_i32,
    )
    selected_scores = sorted_bits.to(tl.float32, bitcast=True)
    output_mask = offsets < top_k
    valid_output = output_mask & (offsets < valid_degree)
    if step == 0:
        default_token = tl.load(first_ids, mask=first_count > 0, other=0)
        first_begin = begin if TRIE_COUNT > 1 else 0
        selected_tokens = tl.load(
            first_ids + first_begin + selected_offsets,
            mask=valid_output,
            other=default_token,
        )
    elif step == 1:
        default_token = tl.load(prefix1_values, mask=prefix1_value_count > 0, other=0)
        selected_tokens = tl.load(
            prefix1_values + begin + selected_offsets,
            mask=valid_output,
            other=default_token,
        )
    else:
        default_token = tl.load(prefix2_values, mask=prefix2_value_count > 0, other=0)
        selected_tokens = tl.load(
            prefix2_values + begin + selected_offsets,
            mask=valid_output,
            other=default_token,
        )
    output_offsets = row * top_k + offsets
    tl.store(output_ids + output_offsets, selected_tokens, mask=output_mask)
    selected_logprobs = selected_scores - denominator
    tl.store(
        output_logprobs + output_offsets,
        tl.where(valid_output, selected_logprobs, -1.0e20),
        mask=output_mask,
    )
    if WRITE_WIRE_LOGPROBS:
        tl.store(
            output_wire_logprobs + output_offsets,
            tl.where(valid_output, selected_logprobs, -float("inf")),
            mask=output_mask,
        )
    tl.store(output_valid_mask + output_offsets, valid_output, mask=output_mask)
    tl.store(valid_counts + row, valid_degree)
    if PACK_PREFILL_RESULT:
        active_count = tl.num_programs(0) * top_k
        payload_token = tl.where(valid_output, selected_tokens, 0).to(tl.int32)
        payload_score = tl.where(valid_output, selected_logprobs, -float("inf"))
        payload_score_bits = payload_score.to(tl.int32, bitcast=True)
        tl.store(prefill_payload + output_offsets, payload_token, mask=output_mask)
        tl.store(prefill_payload + active_count + output_offsets, 0, mask=output_mask)
        tl.store(
            prefill_payload + 2 * active_count + output_offsets,
            payload_score_bits,
            mask=output_mask,
        )
        tl.store(prefill_status + row, tl.where(valid_degree > 0, 0, 1))


def _require_cuda_contiguous(
    name: str,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype | tuple[torch.dtype, ...],
    ndim: int,
) -> None:
    allowed = dtype if isinstance(dtype, tuple) else (dtype,)
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype not in allowed:
        expected = "/".join(str(item).removeprefix("torch.") for item in allowed)
        raise TypeError(f"{name} must use {expected}, got {tensor.dtype}")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_inputs(
    logits: torch.Tensor,
    sequence_group: torch.Tensor,
    first_token_ids: torch.Tensor,
    prefix1_offsets: torch.Tensor,
    prefix1_values: torch.Tensor,
    prefix1_pair_keys: torch.Tensor,
    prefix2_value_offsets: torch.Tensor,
    prefix2_values: torch.Tensor,
    temperatures: torch.Tensor,
    current_step: int,
    top_k: int,
    max_prefix1_degree: int = 0,
    max_prefix2_degree: int = 0,
    vocab_offset: int = 0,
    table_vocab_size: int | None = None,
    trie_ids: torch.Tensor | None = None,
    first_token_offsets: torch.Tensor | None = None,
    max_first_degree: int = 0,
) -> tuple[int, int, torch.Tensor, int]:
    if logits.device.type not in ("cuda", "musa"):
        raise ValueError(f"CUDA constrained Top-K requires CUDA logits, got {logits.device}")
    if logits.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"logits must use fp16/bf16 on CUDA, got {logits.dtype}")
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D, got shape {tuple(logits.shape)}")
    rows, logits_vocab_size = logits.shape
    if rows <= 0 or logits_vocab_size <= 0:
        raise ValueError("logits rows and vocabulary width must be positive")
    if logits.stride(0) <= 0 or logits.stride(1) <= 0:
        raise ValueError(
            f"logits must have positive row and vocabulary strides, got {tuple(logits.stride())}"
        )
    if isinstance(vocab_offset, bool) or not isinstance(vocab_offset, int):
        raise TypeError("vocab_offset must be an int")
    if vocab_offset < 0:
        raise ValueError("vocab_offset must be non-negative")
    if table_vocab_size is None:
        table_vocab_size = logits_vocab_size
    if isinstance(table_vocab_size, bool) or not isinstance(table_vocab_size, int):
        raise TypeError("table_vocab_size must be an int or None")
    if table_vocab_size <= 0:
        raise ValueError("table_vocab_size must be positive")
    if vocab_offset + logits_vocab_size > table_vocab_size:
        raise ValueError(
            "logits vocabulary window must fit inside table_vocab_size: "
            f"offset={vocab_offset}, logits_vocab={logits_vocab_size}, "
            f"table_vocab_size={table_vocab_size}"
        )
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an int")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be in [1, {MAX_TOP_K}], got {top_k}")
    if isinstance(current_step, bool) or not isinstance(current_step, int):
        raise TypeError("current_step must be an int")
    if current_step not in (0, 1, 2):
        raise ValueError(f"unsupported current_step={current_step}")
    if sequence_group.device != logits.device:
        raise ValueError("sequence_group must be on the logits device")
    if sequence_group.dtype != torch.int32:
        raise TypeError("sequence_group must use torch.int32")
    if sequence_group.ndim not in (2, 3):
        raise ValueError("sequence_group must be 2-D or 3-D")
    if not sequence_group.is_contiguous():
        raise ValueError("sequence_group must be contiguous")
    required_prefix = current_step
    if sequence_group.shape[-1] < required_prefix:
        raise ValueError("sequence_group does not contain the requested prefix")
    prefix_rows = sequence_group.shape[0]
    if sequence_group.ndim == 3:
        prefix_rows *= sequence_group.shape[1]
    if prefix_rows != rows:
        raise ValueError("sequence_group row count must match logits")
    flat_prefixes = sequence_group.view(prefix_rows, sequence_group.shape[-1])

    table_specs = (
        ("first_token_ids", first_token_ids, torch.int32),
        ("prefix1_offsets", prefix1_offsets, torch.int32),
        ("prefix1_values", prefix1_values, torch.int32),
        ("prefix1_pair_keys", prefix1_pair_keys, torch.int64),
        ("prefix2_value_offsets", prefix2_value_offsets, torch.int32),
        ("prefix2_values", prefix2_values, torch.int32),
    )
    for name, tensor, dtype in table_specs:
        _require_cuda_contiguous(name, tensor, device=logits.device, dtype=dtype, ndim=1)
    trie_count = 1
    if first_token_offsets is not None:
        _require_cuda_contiguous(
            "first_token_offsets",
            first_token_offsets,
            device=logits.device,
            dtype=torch.int32,
            ndim=1,
        )
        trie_count = first_token_offsets.numel() - 1
        if trie_count < 1:
            raise ValueError("first_token_offsets must contain at least two entries")
    if trie_ids is not None:
        _require_cuda_contiguous(
            "trie_ids", trie_ids, device=logits.device, dtype=torch.int32, ndim=1
        )
        if trie_ids.numel() != rows:
            raise ValueError("trie_ids must contain one ID per logits row")
        if bool(((trie_ids < 0) | (trie_ids >= trie_count)).any()):
            raise ValueError("trie_id is outside the constraint table")
    elif trie_count > 1:
        raise ValueError("multi-trie selection requires trie_ids")
    if prefix1_offsets.numel() != trie_count * table_vocab_size + 1:
        raise ValueError("prefix1_offsets must contain trie_count * table_vocab_size + 1 entries")
    if prefix2_value_offsets.numel() != prefix1_pair_keys.numel() + 1:
        raise ValueError("prefix2_value_offsets must contain pair_count + 1 entries")
    _require_cuda_contiguous(
        "temperatures", temperatures, device=logits.device, dtype=torch.float32, ndim=1
    )
    if temperatures.numel() not in (1, rows):
        raise ValueError("temperatures must contain one value or one value per row")
    for name, degree in (
        ("max_first_degree", max_first_degree),
        ("max_prefix1_degree", max_prefix1_degree),
        ("max_prefix2_degree", max_prefix2_degree),
    ):
        if isinstance(degree, bool) or not isinstance(degree, int):
            raise TypeError(f"{name} must be an int")
        if degree < 0:
            raise ValueError(f"{name} must be non-negative")
    if current_step == 0:
        candidate_count = max_first_degree if trie_count > 1 else first_token_ids.numel()
        if trie_count > 1 and candidate_count == 0:
            raise ValueError("max_first_degree must be provided for multi-trie step 0")
    elif current_step == 1:
        if prefix1_values.numel() and max_prefix1_degree == 0:
            raise ValueError("max_prefix1_degree must be provided for CUDA step 1")
        candidate_count = max_prefix1_degree
    else:
        if prefix2_values.numel() and max_prefix2_degree == 0:
            raise ValueError("max_prefix2_degree must be provided for CUDA step 2")
        candidate_count = max_prefix2_degree
    width = max(top_k, candidate_count, 1)
    block = triton.next_power_of_2(width)
    if block > MAX_SORT_BLOCK:
        raise ValueError(
            f"candidate degree requires sort block {block}, above supported {MAX_SORT_BLOCK}"
        )
    return rows, block, flat_prefixes, table_vocab_size


def _validate_outputs(
    logits: torch.Tensor,
    rows: int,
    top_k: int,
    out_token_ids: torch.Tensor,
    out_logprobs: torch.Tensor,
    out_valid_mask: torch.Tensor,
    out_valid_counts: torch.Tensor,
) -> None:
    outputs = (
        ("out_token_ids", out_token_ids, torch.int32, (rows, top_k)),
        ("out_logprobs", out_logprobs, torch.float32, (rows, top_k)),
        ("out_valid_mask", out_valid_mask, torch.bool, (rows, top_k)),
        ("out_valid_counts", out_valid_counts, torch.int32, (rows,)),
    )
    for name, tensor, dtype, shape in outputs:
        if tensor.device != logits.device:
            raise ValueError(f"{name} must be on {logits.device}")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must use {dtype}")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def _validate_nonoverlapping_outputs(outputs: tuple[tuple[str, torch.Tensor], ...]) -> None:
    """Reject write/write aliasing between contiguous caller-owned outputs."""

    ranges: list[tuple[str, torch.device, int, int]] = []
    for name, tensor in outputs:
        begin = int(tensor.data_ptr())
        end = begin + int(tensor.numel()) * int(tensor.element_size())
        for other_name, other_device, other_begin, other_end in ranges:
            if tensor.device == other_device and begin < other_end and other_begin < end:
                raise ValueError(f"{name} must not overlap {other_name}")
        ranges.append((name, tensor.device, begin, end))


@triton.jit(
    do_not_specialize=[
        "logits_vocab_size",
        "table_vocab_size",
        "vocab_offset",
        "pair_count",
        "first_count",
        "prefix1_value_count",
        "prefix2_value_count",
        "logits_row_stride",
        "logits_vocab_stride",
        "prefix_stride",
        "temperature_count",
        "top_k",
    ]
)
def _stage_topk_kernel(
    pointers,
    logits_vocab_size,
    table_vocab_size,
    vocab_offset,
    pair_count,
    first_count,
    prefix1_value_count,
    prefix2_value_count,
    logits_row_stride,
    logits_vocab_stride,
    prefix_stride,
    temperature_count,
    top_k,
    device_step,
    WRITE_WIRE_LOGPROBS: tl.constexpr,
    PACK_PREFILL_RESULT: tl.constexpr,
    TRIE_COUNT: tl.constexpr,
    BLOCKS: tl.constexpr,
    OUTPUT_BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    WINDOW_BLOCK: tl.constexpr,
):
    # Scalar specialization controls do not recurse into tuple arguments.
    # Keep strides and extents explicit, as in the original kernel, so the
    # compiler retains its normalization reduction layout and fp32 rounding.
    # The device remains authoritative, including stopped/error states. Each
    # branch covers its complete stage degree; no Host prediction truncates it.
    stage = tl.minimum(tl.maximum(tl.load(device_step), 0), 2)
    if stage == 0:
        _fused_topk_kernel(
            *pointers,
            logits_vocab_size,
            table_vocab_size,
            vocab_offset,
            pair_count,
            first_count,
            prefix1_value_count,
            prefix2_value_count,
            logits_row_stride,
            logits_vocab_stride,
            prefix_stride,
            temperature_count,
            top_k,
            WRITE_WIRE_LOGPROBS=WRITE_WIRE_LOGPROBS,
            PACK_PREFILL_RESULT=PACK_PREFILL_RESULT,
            TRIE_COUNT=TRIE_COUNT,
            step=0,
            device_step=None,
            DEVICE_STEP=False,
            BLOCK=BLOCKS[0],
            OUTPUT_BLOCK=OUTPUT_BLOCK,
            TOPK=TOPK,
            WINDOW_BLOCK=WINDOW_BLOCK,
        )
    elif stage == 1:
        _fused_topk_kernel(
            *pointers,
            logits_vocab_size,
            table_vocab_size,
            vocab_offset,
            pair_count,
            first_count,
            prefix1_value_count,
            prefix2_value_count,
            logits_row_stride,
            logits_vocab_stride,
            prefix_stride,
            temperature_count,
            top_k,
            WRITE_WIRE_LOGPROBS=WRITE_WIRE_LOGPROBS,
            PACK_PREFILL_RESULT=PACK_PREFILL_RESULT,
            TRIE_COUNT=TRIE_COUNT,
            step=1,
            device_step=None,
            DEVICE_STEP=False,
            BLOCK=BLOCKS[1],
            OUTPUT_BLOCK=OUTPUT_BLOCK,
            TOPK=TOPK,
            WINDOW_BLOCK=WINDOW_BLOCK,
        )
    else:
        _fused_topk_kernel(
            *pointers,
            logits_vocab_size,
            table_vocab_size,
            vocab_offset,
            pair_count,
            first_count,
            prefix1_value_count,
            prefix2_value_count,
            logits_row_stride,
            logits_vocab_stride,
            prefix_stride,
            temperature_count,
            top_k,
            WRITE_WIRE_LOGPROBS=WRITE_WIRE_LOGPROBS,
            PACK_PREFILL_RESULT=PACK_PREFILL_RESULT,
            TRIE_COUNT=TRIE_COUNT,
            step=2,
            device_step=None,
            DEVICE_STEP=False,
            BLOCK=BLOCKS[2],
            OUTPUT_BLOCK=OUTPUT_BLOCK,
            TOPK=TOPK,
            WINDOW_BLOCK=WINDOW_BLOCK,
        )


def _launch_rec_constrained_topk_cuda_out(
    logits: torch.Tensor,
    first_token_ids: torch.Tensor,
    prefix1_offsets: torch.Tensor,
    prefix1_values: torch.Tensor,
    prefix1_pair_keys: torch.Tensor,
    prefix2_value_offsets: torch.Tensor,
    prefix2_values: torch.Tensor,
    temperatures: torch.Tensor,
    current_step: int,
    top_k: int,
    out_token_ids: torch.Tensor,
    out_logprobs: torch.Tensor,
    out_valid_mask: torch.Tensor,
    out_valid_counts: torch.Tensor,
    vocab_offset: int,
    out_wire_logprobs: torch.Tensor | None,
    out_prefill_status: torch.Tensor | None = None,
    out_prefill_payload: torch.Tensor | None = None,
    *,
    device_step: torch.Tensor | None = None,
    stage_blocks: tuple[int, ...] | None = None,
    rows: int,
    block: int,
    flat_prefixes: torch.Tensor,
    table_vocab_size: int,
    trie_ids: torch.Tensor | None = None,
    first_token_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if stage_blocks is not None:
        _stage_topk_kernel[(rows,)](
            (
                logits,
                flat_prefixes,
                trie_ids if trie_ids is not None else flat_prefixes,
                first_token_offsets if first_token_offsets is not None else prefix1_offsets,
                first_token_ids,
                prefix1_offsets,
                prefix1_values,
                prefix1_pair_keys,
                prefix2_value_offsets,
                prefix2_values,
                temperatures,
                out_token_ids,
                out_logprobs,
                out_wire_logprobs if out_wire_logprobs is not None else out_logprobs,
                out_valid_mask,
                out_valid_counts,
                out_prefill_status if out_prefill_status is not None else out_valid_counts,
                out_prefill_payload if out_prefill_payload is not None else out_token_ids,
            ),
            logits.shape[1],
            table_vocab_size,
            vocab_offset,
            prefix1_pair_keys.numel(),
            first_token_ids.numel(),
            prefix1_values.numel(),
            prefix2_values.numel(),
            logits.stride(0),
            logits.stride(1),
            flat_prefixes.shape[1],
            temperatures.numel(),
            top_k,
            device_step,
            WRITE_WIRE_LOGPROBS=out_wire_logprobs is not None,
            PACK_PREFILL_RESULT=out_prefill_status is not None and out_prefill_payload is not None,
            TRIE_COUNT=1 if first_token_offsets is None else first_token_offsets.numel() - 1,
            BLOCKS=stage_blocks,
            OUTPUT_BLOCK=triton.next_power_of_2(top_k),
            TOPK=getattr(tl, "topk", None),
            WINDOW_BLOCK=2048,
            num_warps=8 if block >= 2048 else 4,
        )
        return out_token_ids, out_logprobs, out_valid_mask, out_valid_counts
    _fused_topk_kernel[(rows,)](
        logits,
        flat_prefixes,
        trie_ids if trie_ids is not None else flat_prefixes,
        first_token_offsets if first_token_offsets is not None else prefix1_offsets,
        first_token_ids,
        prefix1_offsets,
        prefix1_values,
        prefix1_pair_keys,
        prefix2_value_offsets,
        prefix2_values,
        temperatures,
        out_token_ids,
        out_logprobs,
        out_wire_logprobs if out_wire_logprobs is not None else out_logprobs,
        out_valid_mask,
        out_valid_counts,
        out_prefill_status if out_prefill_status is not None else out_valid_counts,
        out_prefill_payload if out_prefill_payload is not None else out_token_ids,
        logits_vocab_size=logits.shape[1],
        table_vocab_size=table_vocab_size,
        vocab_offset=vocab_offset,
        pair_count=prefix1_pair_keys.numel(),
        first_count=first_token_ids.numel(),
        prefix1_value_count=prefix1_values.numel(),
        prefix2_value_count=prefix2_values.numel(),
        logits_row_stride=logits.stride(0),
        logits_vocab_stride=logits.stride(1),
        prefix_stride=flat_prefixes.shape[1],
        temperature_count=temperatures.numel(),
        top_k=top_k,
        WRITE_WIRE_LOGPROBS=out_wire_logprobs is not None,
        PACK_PREFILL_RESULT=(out_prefill_status is not None and out_prefill_payload is not None),
        TRIE_COUNT=1 if first_token_offsets is None else first_token_offsets.numel() - 1,
        step=current_step,
        device_step=device_step,
        DEVICE_STEP=device_step is not None,
        BLOCK=block,
        OUTPUT_BLOCK=triton.next_power_of_2(top_k),
        TOPK=getattr(tl, "topk", None),
        WINDOW_BLOCK=2048,
        num_warps=8 if block >= 2048 else 4,
    )
    return out_token_ids, out_logprobs, out_valid_mask, out_valid_counts


def rec_constrained_topk_cuda_out(
    logits: torch.Tensor,
    sequence_group: torch.Tensor,
    first_token_ids: torch.Tensor,
    prefix1_offsets: torch.Tensor,
    prefix1_values: torch.Tensor,
    prefix1_pair_keys: torch.Tensor,
    prefix2_value_offsets: torch.Tensor,
    prefix2_values: torch.Tensor,
    temperatures: torch.Tensor,
    current_step: int,
    top_k: int,
    max_prefix1_degree: int,
    max_prefix2_degree: int,
    out_token_ids: torch.Tensor,
    out_logprobs: torch.Tensor,
    out_valid_mask: torch.Tensor,
    out_valid_counts: torch.Tensor,
    vocab_offset: int = 0,
    table_vocab_size: int | None = None,
    out_wire_logprobs: torch.Tensor | None = None,
    *,
    trie_ids: torch.Tensor | None = None,
    first_token_offsets: torch.Tensor | None = None,
    max_first_degree: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write constrained Top-K results into caller-owned CUDA tensors."""

    rows, block, flat_prefixes, resolved_table_vocab_size = _validate_inputs(
        logits,
        sequence_group,
        first_token_ids,
        prefix1_offsets,
        prefix1_values,
        prefix1_pair_keys,
        prefix2_value_offsets,
        prefix2_values,
        temperatures,
        current_step,
        top_k,
        max_prefix1_degree,
        max_prefix2_degree,
        vocab_offset,
        table_vocab_size,
        trie_ids,
        first_token_offsets,
        max_first_degree,
    )
    _validate_outputs(
        logits,
        rows,
        top_k,
        out_token_ids,
        out_logprobs,
        out_valid_mask,
        out_valid_counts,
    )
    if out_wire_logprobs is not None:
        _require_cuda_contiguous(
            "out_wire_logprobs",
            out_wire_logprobs,
            device=logits.device,
            dtype=torch.float32,
            ndim=2,
        )
        if tuple(out_wire_logprobs.shape) != (rows, top_k):
            raise ValueError(
                "out_wire_logprobs must have shape "
                f"{(rows, top_k)}, got {tuple(out_wire_logprobs.shape)}"
            )
    writable_outputs = [
        ("out_token_ids", out_token_ids),
        ("out_logprobs", out_logprobs),
        ("out_valid_mask", out_valid_mask),
        ("out_valid_counts", out_valid_counts),
    ]
    if out_wire_logprobs is not None:
        writable_outputs.append(("out_wire_logprobs", out_wire_logprobs))
    _validate_nonoverlapping_outputs(tuple(writable_outputs))
    return _launch_rec_constrained_topk_cuda_out(
        logits,
        first_token_ids,
        prefix1_offsets,
        prefix1_values,
        prefix1_pair_keys,
        prefix2_value_offsets,
        prefix2_values,
        temperatures,
        current_step,
        top_k,
        out_token_ids,
        out_logprobs,
        out_valid_mask,
        out_valid_counts,
        vocab_offset,
        out_wire_logprobs,
        rows=rows,
        block=block,
        flat_prefixes=flat_prefixes,
        table_vocab_size=resolved_table_vocab_size,
        trie_ids=trie_ids,
        first_token_offsets=first_token_offsets,
    )


def _rec_constrained_topk_cuda_out_unchecked(
    logits: torch.Tensor,
    sequence_group: torch.Tensor,
    first_token_ids: torch.Tensor,
    prefix1_offsets: torch.Tensor,
    prefix1_values: torch.Tensor,
    prefix1_pair_keys: torch.Tensor,
    prefix2_value_offsets: torch.Tensor,
    prefix2_values: torch.Tensor,
    temperatures: torch.Tensor,
    current_step: int,
    top_k: int,
    max_prefix1_degree: int,
    max_prefix2_degree: int,
    out_token_ids: torch.Tensor,
    out_logprobs: torch.Tensor,
    out_valid_mask: torch.Tensor,
    out_valid_counts: torch.Tensor,
    vocab_offset: int,
    table_vocab_size: int,
    out_wire_logprobs: torch.Tensor | None,
    out_prefill_status: torch.Tensor | None = None,
    out_prefill_payload: torch.Tensor | None = None,
    device_step: torch.Tensor | None = None,
    specialize_steps: bool = False,
    *,
    trie_ids: torch.Tensor | None = None,
    first_token_offsets: torch.Tensor | None = None,
    max_first_degree: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch on startup-validated CUDA prefill tensors (Worker-internal)."""

    rows = int(logits.shape[0])
    first_count = (
        max_first_degree
        if first_token_offsets is not None and first_token_offsets.numel() > 2
        else first_token_ids.numel()
    )
    candidate_count = (
        int(first_count)
        if current_step == 0
        else (int(max_prefix1_degree) if current_step == 1 else int(max_prefix2_degree))
    )
    if device_step is not None:
        candidate_count = max(first_count, max_prefix1_degree, max_prefix2_degree)
    block = triton.next_power_of_2(max(top_k, candidate_count, 1))
    stage_blocks = None
    if specialize_steps and device_step is not None:
        stage_blocks = tuple(
            triton.next_power_of_2(max(top_k, degree, 1))
            for degree in (first_count, max_prefix1_degree, max_prefix2_degree)
        )
    flat_prefixes = sequence_group.view(rows, sequence_group.shape[-1])
    return _launch_rec_constrained_topk_cuda_out(
        logits,
        first_token_ids,
        prefix1_offsets,
        prefix1_values,
        prefix1_pair_keys,
        prefix2_value_offsets,
        prefix2_values,
        temperatures,
        current_step,
        top_k,
        out_token_ids,
        out_logprobs,
        out_valid_mask,
        out_valid_counts,
        vocab_offset,
        out_wire_logprobs,
        out_prefill_status,
        out_prefill_payload,
        device_step=device_step,
        stage_blocks=stage_blocks,
        rows=rows,
        block=block,
        flat_prefixes=flat_prefixes,
        table_vocab_size=table_vocab_size,
        trie_ids=trie_ids,
        first_token_offsets=first_token_offsets,
    )


def rec_constrained_topk_cuda(
    logits: torch.Tensor,
    sequence_group: torch.Tensor,
    first_token_ids: torch.Tensor,
    prefix1_offsets: torch.Tensor,
    prefix1_values: torch.Tensor,
    prefix1_pair_keys: torch.Tensor,
    prefix2_value_offsets: torch.Tensor,
    prefix2_values: torch.Tensor,
    temperatures: torch.Tensor,
    current_step: int,
    top_k: int,
    max_prefix1_degree: int = 0,
    max_prefix2_degree: int = 0,
    vocab_offset: int = 0,
    table_vocab_size: int | None = None,
    *,
    trie_ids: torch.Tensor | None = None,
    first_token_offsets: torch.Tensor | None = None,
    max_first_degree: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate and return int32 tokens, fp32 logprobs, mask, and counts."""

    rows, _, _, _ = _validate_inputs(
        logits,
        sequence_group,
        first_token_ids,
        prefix1_offsets,
        prefix1_values,
        prefix1_pair_keys,
        prefix2_value_offsets,
        prefix2_values,
        temperatures,
        current_step,
        top_k,
        max_prefix1_degree,
        max_prefix2_degree,
        vocab_offset,
        table_vocab_size,
        trie_ids,
        first_token_offsets,
        max_first_degree,
    )
    token_ids = torch.empty((rows, top_k), dtype=torch.int32, device=logits.device)
    logprobs = torch.empty((rows, top_k), dtype=torch.float32, device=logits.device)
    valid_mask = torch.empty((rows, top_k), dtype=torch.bool, device=logits.device)
    valid_counts = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    return rec_constrained_topk_cuda_out(
        logits,
        sequence_group,
        first_token_ids,
        prefix1_offsets,
        prefix1_values,
        prefix1_pair_keys,
        prefix2_value_offsets,
        prefix2_values,
        temperatures,
        current_step,
        top_k,
        max_prefix1_degree,
        max_prefix2_degree,
        token_ids,
        logprobs,
        valid_mask,
        valid_counts,
        vocab_offset,
        table_vocab_size,
        trie_ids=trie_ids,
        first_token_offsets=first_token_offsets,
        max_first_degree=max_first_degree,
    )


__all__ = [
    "INVALID_LOGIT",
    "MAX_SORT_BLOCK",
    "MAX_TOP_K",
    "validate_cuda_sort_capacity",
    "rec_constrained_topk_cuda",
    "rec_constrained_topk_cuda_out",
]
