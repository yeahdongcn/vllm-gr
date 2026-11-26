# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Host-side device policy shared by Beam frontends and EngineCore."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm_gr.config import GRConfig
from vllm_gr.config.models import (
    ConstraintBackendKind,
    ConstraintMode,
    configured_constraint_mode,
    runtime_constraint_table,
)

if TYPE_CHECKING:
    from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest


def _device_type(config: Any, *, default: str | None = None) -> str | None:
    device = getattr(getattr(config, "device_config", None), "device", default)
    return str(device).split(":", 1)[0] if device is not None else None


def validate_gr_device_config(config: Any, request: BeamBatchRequest | None = None) -> None:
    """Check execution support without importing a device runtime."""
    device = _device_type(config)
    if device == "musa":
        from vllm_gr.platforms.interface import BeamKVReorderProvider
        from vllm_gr.platforms.registry import get_active_backend

        if not isinstance(get_active_backend(), BeamKVReorderProvider):
            raise ValueError("MUSA Beam requires an active GR Beam provider")
    elif device not in (None, "cuda", "npu"):
        raise ValueError(
            "Asynchronous scheduling requires a CUDA ModelRunner or NPU ModelRunner V1"
        )
    gr_config = GRConfig.from_vllm_config(config)
    if gr_config.beam_sid_stage_vocab and device != "npu":
        raise ValueError("beam.sid_stage_vocab requires NPU ModelRunner V1")
    if device == "npu":
        from vllm_gr.platforms.ascend.admission import validate_npu_device_config

        validate_npu_device_config(config, gr_config, request)


def validate_decode_token_budget(request: BeamBatchRequest, config: Any) -> None:
    if _device_type(config) == "npu":
        from vllm_gr.platforms.ascend.admission import validate_decode_token_budget

        validate_decode_token_budget(request, config)


def decode_batch_capacity(config: Any, beam_width: int) -> int | None:
    """Return the physical Decode batch cap used for frontend grouping."""
    if _device_type(config) != "npu":
        return None
    return config.scheduler_config.max_num_batched_tokens // beam_width


def uses_retained_candidate_widths(config: GRConfig, device: object) -> bool:
    """Whether the worker derives candidate K from each stage's retained width."""
    options = config.deployment
    return (
        str(device).split(":", 1)[0] == "npu"
        and options.candidate_width is not None
        and options.candidate_width == options.beam_width
    )


def validate_dynamic_width_support(config: Any) -> None:
    if _device_type(config, default="cuda") not in ("cuda", "npu"):
        raise ValueError("dynamic_beam_widths and result_width require CUDA or NPU beam_search")


def validate_frontend_constraint_mode(
    config: Any, gr_config: GRConfig, catalog_strategy: str | None
) -> None:
    device = _device_type(config, default="cuda")
    if configured_constraint_mode(gr_config) is ConstraintMode.NONE:
        if device not in ("cuda", "musa", "npu"):
            raise ValueError("unconstrained beam_search requires CUDA, MUSA, or NPU")
        return

    table = runtime_constraint_table(gr_config)
    backend = ConstraintBackendKind.NPU if device == "npu" else ConstraintBackendKind.CUDA
    if table is None or not table.enabled or table.backend is not backend:
        raise ValueError(
            f"beam_search requires an enabled canonical {backend.value.upper()} constraint table"
        )
    if catalog_strategy is not None:
        raise ValueError("catalog_strategy is not applicable to the constraint_table backend")
