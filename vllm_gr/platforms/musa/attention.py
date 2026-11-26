# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Eager grouped-query attention over a shared prompt and private Beam suffixes."""

import torch


def eager_beam_decode(
    query: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    block_table: torch.Tensor,
    prefix_length: int,
    suffix_key: torch.Tensor,
    suffix_value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Attend one token per beam to the paged prompt and its own suffix.

    Query is [beam, query_heads, dim], pages are [page, token, kv_heads, dim],
    and suffixes are [beam, kv_heads, token, dim]. Suffixes include the current
    token and contain only live positions.
    """
    beam_width, query_heads, head_dim = query.shape
    _, page_size, kv_heads, _ = key_pages.shape
    if query_heads % kv_heads:
        raise ValueError("Query head count must be divisible by KV head count")
    if not 0 <= prefix_length <= block_table.numel() * page_size:
        raise ValueError("Beam prefix length is outside the block table")
    if suffix_key.shape[2] == 0:
        raise ValueError("Beam suffix must include the current token")

    num_pages = (prefix_length + page_size - 1) // page_size
    pages = block_table[:num_pages].long()
    prefix_key = key_pages.index_select(0, pages).flatten(0, 1)[:prefix_length].float()
    prefix_value = value_pages.index_select(0, pages).flatten(0, 1)[:prefix_length].float()
    grouped_query = query.float().reshape(beam_width, kv_heads, -1, head_dim)
    prefix_scores = torch.einsum("bhgd,thd->bhgt", grouped_query, prefix_key)
    suffix_scores = torch.einsum("bhgd,bhtd->bhgt", grouped_query, suffix_key.float())
    weights = torch.softmax(torch.cat((prefix_scores, suffix_scores), dim=-1) * scale, dim=-1)
    prefix_weight, suffix_weight = weights.split((prefix_length, suffix_key.shape[2]), dim=-1)
    prefix_output = torch.einsum("bhgt,thd->bhgd", prefix_weight, prefix_value)
    suffix_output = torch.einsum("bhgt,bhtd->bhgd", suffix_weight, suffix_value.float())
    return (prefix_output + suffix_output).reshape_as(query).to(query.dtype)


def masked_beam_decode(
    query: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    block_table: torch.Tensor,
    prefix_length: torch.Tensor,
    suffix_key: torch.Tensor,
    suffix_value: torch.Tensor,
    suffix_lengths: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Use fixed-capacity buffers and read live lengths on each replay."""
    width, heads, dim = query.shape
    _, page_size, kv_heads, _ = key_pages.shape
    page_valid = torch.arange(block_table.numel(), device=query.device) * page_size < prefix_length
    pages = torch.where(page_valid, block_table, 0).long()
    prompt_valid = torch.arange(pages.numel() * page_size, device=query.device) < prefix_length
    suffix_valid = torch.arange(suffix_key.shape[2], device=query.device) < suffix_lengths.reshape(
        -1, 1
    )
    prompt_key = key_pages.index_select(0, pages).flatten(0, 1).float()
    prompt_value = value_pages.index_select(0, pages).flatten(0, 1).float()
    prompt_key = prompt_key.masked_fill(~prompt_valid[:, None, None], 0)
    prompt_value = prompt_value.masked_fill(~prompt_valid[:, None, None], 0)
    suffix_key = suffix_key.float().masked_fill(~suffix_valid[:, None, :, None], 0)
    suffix_value = suffix_value.float().masked_fill(~suffix_valid[:, None, :, None], 0)
    grouped = query.float().reshape(width, kv_heads, heads // kv_heads, dim)
    prompt_scores = torch.einsum("bhgd,thd->bhgt", grouped, prompt_key)
    suffix_scores = torch.einsum("bhgd,bhtd->bhgt", grouped, suffix_key)
    scores = torch.cat((prompt_scores, suffix_scores), dim=-1) * scale
    valid = torch.cat((prompt_valid.expand(width, -1), suffix_valid.expand(width, -1)), dim=-1)
    weights = scores.masked_fill(~valid[:, None, None, :], -torch.inf).softmax(dim=-1)
    weights = weights.masked_fill(~valid[:, None, None, :], 0)
    prompt_weights, suffix_weights = weights.split(
        (prompt_valid.numel(), suffix_key.shape[2]), dim=-1
    )
    output = torch.einsum("bhgt,thd->bhgd", prompt_weights, prompt_value)
    output = output + torch.einsum("bhgt,bhtd->bhgd", suffix_weights, suffix_value)
    return output.reshape_as(query).to(query.dtype)


def write_suffix_kv(
    unshared_key: torch.Tensor,
    unshared_value: torch.Tensor,
    curr_key: torch.Tensor,
    curr_value: torch.Tensor,
    slot: int,
    decode_step: int,
    beam_session: object = None,
) -> None:
    """Append the current token to the request-major decode pool."""
    width = curr_key.shape[0]
    if torch.musa.is_current_stream_capturing():
        step = beam_session.context.decode_step_buf
        for pool, current in ((unshared_key, curr_key), (unshared_value, curr_value)):
            history = pool[slot, :width]
            offsets = torch.arange(history.shape[2], device=pool.device)
            write = offsets[None, None, :, None] == step
            history.copy_(torch.where(write, current.unsqueeze(2), history))
        return
    unshared_key[slot, :width, :, decode_step].copy_(curr_key)
    unshared_value[slot, :width, :, decode_step].copy_(curr_value)


def write_paged_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> None:
    """Scatter valid slots into the logical NHD view, preserving its strides."""
    if key_cache.dtype != key.dtype or value_cache.dtype != value.dtype:
        raise ValueError("Eager MUSA Beam requires unquantized KV")
    if torch.musa.is_current_stream_capturing():
        from vllm._custom_ops import reshape_and_cache_flash

        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
        return
    slots = slot_mapping.long()
    valid = slots >= 0
    slots = slots[valid]
    block_size = key_cache.shape[1]
    pages = slots // block_size
    offsets = slots % block_size
    key_cache[pages, offsets] = key[: slot_mapping.numel()][valid]
    value_cache[pages, offsets] = value[: slot_mapping.numel()][valid]


def eager_paged_prefill(
    query: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    block_table: torch.Tensor,
    sequence_length: int,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    """Attend the final query positions to their logical paged KV sequence."""
    query_length, query_heads, head_dim = query.shape
    _, page_size, kv_heads, _ = key_pages.shape
    if not query_length <= sequence_length <= block_table.numel() * page_size:
        raise ValueError("Invalid paged prefill geometry")
    pages = block_table[: (sequence_length + page_size - 1) // page_size].long()
    key = key_pages.index_select(0, pages).flatten(0, 1)[:sequence_length].float()
    value = value_pages.index_select(0, pages).flatten(0, 1)[:sequence_length].float()
    grouped = query.float().reshape(query_length, kv_heads, -1, head_dim)
    scores = torch.einsum("qhgd,khd->qhgk", grouped, key) * scale
    if causal:
        positions = torch.arange(sequence_length, device=query.device)
        query_positions = positions[-query_length:]
        future = positions[None, :] > query_positions[:, None]
        scores = scores.masked_fill(future[:, None, None, :], -torch.inf)
    result = torch.einsum("qhgk,khd->qhgd", scores.softmax(dim=-1), value)
    return result.reshape_as(query).to(query.dtype)


class EagerBeamAttention:
    """Consume the host sequence geometry already carried by Beam metadata."""

    reshape_and_cache = staticmethod(write_paged_kv)
    cache_unshared_kv = staticmethod(write_suffix_kv)

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_flash_attn_version(*, requires_alibi=False, head_size=None) -> None:
        if requires_alibi:
            raise ValueError("Eager MUSA Beam does not support ALiBi")

    @staticmethod
    def supports_sinks() -> bool:
        return False

    @staticmethod
    def cascade_attention(
        *,
        output,
        query,
        key_cache,
        value_cache,
        attn_metadata,
        sliding_window_size,
        scale,
        logits_soft_cap,
        vllm_flash_attn_version,
        s_aux=None,
        alibi_slopes=None,
        beam_pool=None,
        prefill_flash_attn_version=None,
    ) -> None:
        if (
            sliding_window_size not in (None, (-1, -1), [-1, -1])
            or logits_soft_cap
            or s_aux is not None
            or alibi_slopes is not None
        ):
            raise ValueError("Unsupported eager MUSA Beam attention configuration")
        metadata = attn_metadata
        if metadata.prefix_indices is None:
            offsets = metadata.std_cu_seqlens_q_cpu
            if offsets is None:
                offsets = metadata.std_cu_seqlens_q_device.tolist()
            lengths = metadata.std_seqlens_kv_cpu
            if lengths is None:
                lengths = metadata.std_seqlens_kv_device.tolist()
            for row, length in enumerate(lengths):
                start, end = offsets[row : row + 2]
                output[start:end].copy_(
                    eager_paged_prefill(
                        query[start:end],
                        key_cache,
                        value_cache,
                        metadata.std_block_table[row],
                        length,
                        scale,
                        metadata.causal,
                    )
                )
            return
        if metadata.std_indices is not None or metadata.num_reqs != 1 or beam_pool is None:
            raise ValueError("Eager MUSA Beam decode requires one dedicated Beam request")
        session = metadata.beam_session
        width = session.beam_width
        length = metadata.incremental_suffix_kv_len
        slot = session.slot
        if torch.musa.is_current_stream_capturing():
            output[:width].copy_(
                masked_beam_decode(
                    query[:width],
                    key_cache,
                    value_cache,
                    metadata.prefix_block_table[0],
                    metadata.prefix_seqlens_kv_device[0],
                    beam_pool.unshared_key[slot, :width],
                    beam_pool.unshared_value[slot, :width],
                    session.context.decode_step_buf + 1,
                    scale,
                )
            )
            return
        output[:width].copy_(
            eager_beam_decode(
                query[:width],
                key_cache,
                value_cache,
                metadata.prefix_block_table[0],
                metadata.prefix_seqlens_kv_cpu[0],
                beam_pool.unshared_key[slot, :width, :, :length],
                beam_pool.unshared_value[slot, :width, :, :length],
                scale,
            )
        )

    def prefill_attention(
        self,
        *,
        output,
        query,
        key_cache,
        value_cache,
        attn_metadata,
        sliding_window_size,
        scale,
        logits_soft_cap,
        vllm_flash_attn_version,
        s_aux=None,
        alibi_slopes=None,
        dense_prefill="off",
    ) -> None:
        if (
            dense_prefill != "off"
            or sliding_window_size not in (None, (-1, -1), [-1, -1])
            or logits_soft_cap
            or s_aux is not None
            or alibi_slopes is not None
        ):
            raise ValueError("Unsupported eager MUSA Beam prefill configuration")
        blocks = getattr(attn_metadata, "block_table_tensor", None)
        if blocks is None:
            blocks = attn_metadata.block_table
        if torch.musa.is_current_stream_capturing():
            from flash_attn_interface import flash_attn_varlen_func

            flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                out=output,
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=attn_metadata.seq_lens,
                max_seqlen_k=attn_metadata.max_seq_len,
                block_table=blocks,
                softmax_scale=scale,
                causal=attn_metadata.causal,
            )
            return
        offsets = attn_metadata.query_start_loc.tolist()
        lengths = attn_metadata.seq_lens.tolist()
        for row, length in enumerate(lengths):
            start, end = offsets[row : row + 2]
            if start == end:
                continue
            output[start:end].copy_(
                eager_paged_prefill(
                    query[start:end],
                    key_cache,
                    value_cache,
                    blocks[row],
                    length,
                    scale,
                    attn_metadata.causal,
                )
            )


EAGER_BEAM_ATTENTION = EagerBeamAttention()


def reorder_suffix_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    slot: int,
    parent_ids: torch.Tensor,
    decode_step: torch.Tensor,
    active_mask: torch.Tensor,
    execute: torch.Tensor,
) -> None:
    """Copy live parent histories into the next beam order.

    Materialize the gathered rows before writing so swaps and repeated parents
    read the old generation. Leave future positions and other slots intact.
    """
    width = parent_ids.numel()
    destinations = torch.arange(width, device=parent_ids.device)
    parents = torch.where(active_mask & execute, parent_ids, destinations).long()
    for pool in (key, value):
        history = pool[slot, :width]
        positions = torch.arange(history.shape[2], device=pool.device)
        live = positions[None, None, :, None] <= decode_step
        history.copy_(torch.where(live, history.index_select(0, parents), history))
