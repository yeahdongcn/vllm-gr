# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA Backend owning the canonical six-table device representation."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto
from typing import TypedDict, cast

import torch

from vllm_gr.constraints.backend import (
    ConstraintBackendInfo,
    ConstraintMetadata,
    ConstraintOutput,
    build_constraint_output,
    validate_select_inputs,
)
from vllm_gr.ops.cuda.rec_constrained_topk import (
    _rec_constrained_topk_cuda_out_unchecked,
    rec_constrained_topk_cuda,
    rec_constrained_topk_cuda_out,
    validate_cuda_sort_capacity,
)


class _BackendState(Enum):
    INITIALIZING = auto()
    READY = auto()
    CLOSED = auto()


_ConstraintTables = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


class _TrieArguments(TypedDict, total=False):
    trie_ids: torch.Tensor
    first_token_offsets: torch.Tensor | None
    max_first_degree: int


class CUDAConstraintBackend:
    """Dispatch full-window normalized Top-K over GPU-resident canonical tables."""

    __slots__ = (
        "_first_token_ids",
        "_first_token_offsets",
        "_max_first_degree",
        "_info",
        "_prefix1_offsets",
        "_prefix1_pair_keys",
        "_prefix1_values",
        "_prefix2_value_offsets",
        "_prefix2_values",
        "_state",
        "vocab_slice",
    )

    def __init__(
        self,
        *,
        first_token_ids: torch.Tensor,
        prefix1_offsets: torch.Tensor,
        prefix1_values: torch.Tensor,
        prefix1_pair_keys: torch.Tensor,
        prefix2_value_offsets: torch.Tensor,
        prefix2_values: torch.Tensor,
        info: ConstraintBackendInfo,
        vocab_slice: tuple[int, int] | None = None,
        first_token_offsets: torch.Tensor | None = None,
        max_first_degree: int = 0,
    ) -> None:
        validate_cuda_sort_capacity(
            max_first_degree
            if info.table_metadata.trie_count > 1
            else int(first_token_ids.numel()),
            info.table_metadata.max_prefix1_degree,
            info.table_metadata.max_prefix2_degree,
        )
        self._first_token_ids: torch.Tensor | None = first_token_ids
        self._first_token_offsets = first_token_offsets
        self._max_first_degree = max_first_degree
        if info.table_metadata.trie_count > 1 and first_token_offsets is None:
            raise ValueError("multi-trie CUDA tables require first_token_offsets")
        self._prefix1_offsets: torch.Tensor | None = prefix1_offsets
        self._prefix1_values: torch.Tensor | None = prefix1_values
        self._prefix1_pair_keys: torch.Tensor | None = prefix1_pair_keys
        self._prefix2_value_offsets: torch.Tensor | None = prefix2_value_offsets
        self._prefix2_values: torch.Tensor | None = prefix2_values
        self._info = info
        self.vocab_slice = vocab_slice
        self._state = _BackendState.INITIALIZING

    @property
    def info(self) -> ConstraintBackendInfo:
        return self._info

    @property
    def is_ready(self) -> bool:
        """Whether admission can use this backend without initialization work."""
        return self._state is _BackendState.READY

    def _tables(self) -> _ConstraintTables:
        tables = (
            self._first_token_ids,
            self._prefix1_offsets,
            self._prefix1_values,
            self._prefix1_pair_keys,
            self._prefix2_value_offsets,
            self._prefix2_values,
        )
        if self._state is _BackendState.CLOSED or any(table is None for table in tables):
            raise RuntimeError("CUDA constraint backend is closed")
        return cast(_ConstraintTables, tables)

    def _invoke(
        self,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
        *,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        out_wire_logprobs: torch.Tensor | None = None,
        out_prefill_status: torch.Tensor | None = None,
        out_prefill_payload: torch.Tensor | None = None,
        device_step: torch.Tensor | None = None,
        specialize_steps: bool = False,
        validate: bool = True,
        build_output: bool = True,
    ) -> ConstraintOutput | None:
        table_vocab_size = self._info.table_metadata.vocab_size
        vocab_offset = 0
        expected_logits_vocab_size = table_vocab_size
        if logits.shape[1] != table_vocab_size:
            offset, slice_size = self.vocab_slice or (None, None)
            if offset is None or slice_size is None or logits.shape[1] != slice_size:
                raise ValueError(
                    f"CUDA constraint logits vocab {logits.shape[1]} does not match "
                    f"table vocab_size {table_vocab_size} or the configured vocabulary slice"
                )
            if offset < 0 or offset + slice_size > table_vocab_size:
                raise ValueError("configured CUDA vocabulary slice is outside table vocabulary")
            vocab_offset = offset
            expected_logits_vocab_size = slice_size
        if validate:
            validate_select_inputs(
                logits,
                metadata,
                self._info,
                expected_logits_vocab_size=expected_logits_vocab_size,
            )
            if logits.device.type not in ("cuda", "musa"):
                raise ValueError(
                    f"CUDA constraint backend requires CUDA logits, got {logits.device}"
                )
        tables = self._tables()
        trie_kwargs: _TrieArguments = {}
        if self._info.table_metadata.trie_count > 1:
            if metadata.trie_ids is None:
                raise ValueError("multi-trie CUDA selection requires trie_ids")
            trie_kwargs = {
                "trie_ids": metadata.trie_ids,
                "first_token_offsets": self._first_token_offsets,
                "max_first_degree": self._max_first_degree,
            }
        if validate:
            if (
                self._first_token_offsets is not None
                and self._first_token_offsets.device != logits.device
            ):
                raise RuntimeError("root offsets must share logits.device")
            for table in tables:
                if table.device != logits.device:
                    raise RuntimeError(
                        "all CUDA constraint tables must share logits.device; "
                        f"got table={table.device}, logits={logits.device}"
                    )
        op_args = (
            logits,
            metadata.prefix_token_ids,
            *tables,
            metadata.temperatures,
            metadata.current_step,
            metadata.top_k,
            self._info.table_metadata.max_prefix1_degree,
            self._info.table_metadata.max_prefix2_degree,
        )
        if outputs is None:
            token_ids, logprobs, valid_mask, _ = rec_constrained_topk_cuda(
                *op_args,
                vocab_offset=vocab_offset,
                table_vocab_size=table_vocab_size,
                **trie_kwargs,
            )
        else:
            token_ids, logprobs, valid_mask, valid_counts = outputs
            launch_out: Callable[..., tuple[torch.Tensor, ...]]
            if validate:
                launch_out = rec_constrained_topk_cuda_out
            else:
                launch_out = _rec_constrained_topk_cuda_out_unchecked
            launch_kwargs = {
                "vocab_offset": vocab_offset,
                "table_vocab_size": table_vocab_size,
                "out_wire_logprobs": out_wire_logprobs,
            }
            if not validate:
                launch_kwargs["device_step"] = device_step
                launch_kwargs["specialize_steps"] = specialize_steps
                launch_kwargs["out_prefill_status"] = out_prefill_status
                launch_kwargs["out_prefill_payload"] = out_prefill_payload
            launch_out(
                *op_args,
                token_ids,
                logprobs,
                valid_mask,
                valid_counts,
                **launch_kwargs,
                **trie_kwargs,
            )
        if not build_output:
            # Fixed-address (V1 Worker) path: the fused kernel now outputs
            # full-window normalized logprobs directly (the scoring contract),
            # so nothing needs rescoring here.
            return None
        return build_constraint_output(
            token_ids,
            logprobs,
            valid_mask,
            logits=logits,
            top_k=metadata.top_k,
        )

    def _select_for_readiness(
        self, logits: torch.Tensor, metadata: ConstraintMetadata
    ) -> ConstraintOutput:
        if self._state is not _BackendState.INITIALIZING:
            raise RuntimeError("CUDA readiness can only run while initializing")
        output = self._invoke(logits, metadata)
        assert output is not None
        return output

    def _mark_ready(self) -> None:
        if self._state is not _BackendState.INITIALIZING:
            raise RuntimeError("only an initializing CUDA constraint backend can become ready")
        self._state = _BackendState.READY

    def warmup_stage_topk(self, dtype: torch.dtype) -> None:
        """Warm V2's device-step kernel for every supported candidate-capacity bucket."""
        device = self._tables()[0].device
        vocab_size = (
            self.vocab_slice[1]
            if self.vocab_slice is not None
            else self.info.table_metadata.vocab_size
        )
        logits = torch.zeros((1, vocab_size), dtype=dtype, device=device)
        prefixes = torch.zeros((1, 3), dtype=torch.int32, device=device)
        temperatures = torch.ones(1, dtype=torch.float32, device=device)
        step = torch.zeros(1, dtype=torch.int32, device=device)
        trie_ids = torch.zeros(1, dtype=torch.int32, device=device)
        max_top_k = self.info.max_top_k
        token_ids = torch.empty((1, max_top_k), dtype=torch.int32, device=device)
        logprobs = torch.empty((1, max_top_k), dtype=torch.float32, device=device)
        valid_mask = torch.empty((1, max_top_k), dtype=torch.bool, device=device)
        valid_counts = torch.empty(1, dtype=torch.int32, device=device)
        # K itself, row counts and strides do not specialize the kernel. One K
        # per power-of-two output capacity covers all admitted request widths;
        # each launch compiles all three device-selected SID branches together.
        for power in range((max_top_k - 1).bit_length() + 1):
            top_k = min(1 << power, max_top_k)
            self.select_out_unchecked(
                logits,
                ConstraintMetadata(prefixes, temperatures, 0, top_k, trie_ids),
                out_token_ids=token_ids,
                out_logprobs=logprobs,
                out_valid_mask=valid_mask,
                out_valid_counts=valid_counts,
                device_step=step,
                specialize_steps=True,
            )
        torch.accelerator.synchronize(device)

    def select(self, logits: torch.Tensor, metadata: ConstraintMetadata) -> ConstraintOutput:
        if self._state is not _BackendState.READY:
            raise RuntimeError("CUDA constraint backend is not ready")
        output = self._invoke(logits, metadata)
        assert output is not None
        return output

    def select_out(
        self,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
        *,
        out_token_ids: torch.Tensor,
        out_logprobs: torch.Tensor,
        out_valid_mask: torch.Tensor,
        out_valid_counts: torch.Tensor,
        out_wire_logprobs: torch.Tensor | None = None,
    ) -> ConstraintOutput:
        """Write CUDA candidates into caller-owned fixed-address tensors."""

        if self._state is not _BackendState.READY:
            raise RuntimeError("CUDA constraint backend is not ready")
        output = self._invoke(
            logits,
            metadata,
            outputs=(
                out_token_ids,
                out_logprobs,
                out_valid_mask,
                out_valid_counts,
            ),
            out_wire_logprobs=out_wire_logprobs,
        )
        assert output is not None
        return output

    def select_out_unchecked(
        self,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
        *,
        out_token_ids: torch.Tensor,
        out_logprobs: torch.Tensor,
        out_valid_mask: torch.Tensor,
        out_valid_counts: torch.Tensor,
        out_wire_logprobs: torch.Tensor | None = None,
        out_prefill_status: torch.Tensor | None = None,
        out_prefill_payload: torch.Tensor | None = None,
        device_step: torch.Tensor | None = None,
        specialize_steps: bool = False,
    ) -> None:
        """Write to a prevalidated persistent Worker workspace.

        Backend initialization and StageRunner admission establish the
        device, dtype, table, shape, and capacity invariants.  Keeping only the
        backend state check here avoids repeating the complete public input
        validation between LM-Head and the CUDA TopK launch.
        """

        if self._state is not _BackendState.READY:
            raise RuntimeError("CUDA constraint backend is not ready")
        self._invoke(
            logits,
            metadata,
            outputs=(
                out_token_ids,
                out_logprobs,
                out_valid_mask,
                out_valid_counts,
            ),
            out_wire_logprobs=out_wire_logprobs,
            out_prefill_status=out_prefill_status,
            out_prefill_payload=out_prefill_payload,
            device_step=device_step,
            specialize_steps=specialize_steps,
            validate=False,
            build_output=False,
        )

    def close(self) -> None:
        if self._state is _BackendState.CLOSED:
            return
        self._first_token_ids = None
        self._first_token_offsets = None
        self._prefix1_offsets = None
        self._prefix1_values = None
        self._prefix1_pair_keys = None
        self._prefix2_value_offsets = None
        self._prefix2_values = None
        self._state = _BackendState.CLOSED


__all__ = ["CUDAConstraintBackend"]
