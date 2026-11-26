# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Normalize engine inputs and admit plans without storing provider state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from enum import Enum
from importlib import metadata
from typing import Any

from vllm_gr.config.backend_plan import (
    BACKEND_PLAN_KEY,
    BackendPlan,
    BackendRequest,
    PlanData,
)

from .loader import external_provider_entry_points

BACKEND_OPTIONS_KEY = "vllm_gr_backend_options"

# Only settings common to every rank belong in the engine fingerprint. Native
# allocation results, device ordinals, rank IDs and handles deliberately do not.
_NATIVE_FIELDS = {
    "model_config": (
        "dtype",
        "enforce_eager",
        "quantization",
        "max_model_len",
        "is_encoder_decoder",
        "is_hybrid",
        "is_multimodal_model",
    ),
    "cache_config": ("block_size", "cache_dtype", "enable_prefix_caching"),
    "scheduler_config": (
        "async_scheduling",
        "enable_chunked_prefill",
        "max_num_seqs",
        "max_num_batched_tokens",
        "long_prefill_token_threshold",
    ),
    "parallel_config": (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
        "use_ubatching",
    ),
}


def _primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return _primitive(value.value)
    if isinstance(value, Mapping):
        return {key: _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    return value


def _version(distribution: str | None) -> str | None:
    if distribution is None:
        return None
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def provider_compatibility(provider: Any) -> PlanData:
    """Capture versions outside resolve(), which must be pure."""
    from vllm_gr.version import __version__

    distributions = []
    for entry in external_provider_entry_points():
        if entry.name == provider.backend_id and entry.dist is not None:
            distributions.append((entry.dist.metadata["Name"], entry.dist.version))
    return PlanData(
        {
            "gr": __version__,
            "vllm": _version("vllm"),
            "vendor_distribution": getattr(provider, "vendor_distribution", None),
            "vendor": _version(getattr(provider, "vendor_distribution", None)),
            "provider_distributions": sorted(distributions),
            "provider_class": f"{type(provider).__module__}.{type(provider).__qualname__}",
            "adapter_revision": getattr(provider, "adapter_revision", "legacy-v2"),
        }
    )


def normalize_backend_request(config: Any, provider: Any) -> BackendRequest:
    from vllm_gr.config import GRConfig

    native: dict[str, Any] = {}
    for section, fields in _NATIVE_FIELDS.items():
        source = getattr(config, section, None)
        values = {name: _primitive(getattr(source, name, None)) for name in fields}
        # torch.dtype is deliberately encoded by its stable public name, never
        # by a general default=str serializer for arbitrary runtime objects.
        if section == "model_config" and values["dtype"] is not None:
            values["dtype"] = str(values["dtype"])
        native[section] = values
    native["use_v2_model_runner"] = bool(getattr(config, "use_v2_model_runner", False))
    model = getattr(config, "model_config", None)
    text_config = getattr(model, "hf_text_config", None)
    native["geometry"] = {
        key: getattr(text_config, key, None)
        for key in (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_hidden_layers",
            "vocab_size",
        )
    }
    additional = getattr(config, "additional_config", None)
    options = additional.get(BACKEND_OPTIONS_KEY, {}) if isinstance(additional, Mapping) else {}
    gr = GRConfig.from_vllm_config(config)
    gr_fields = asdict(gr)
    gr_fields["beam_execution_mode"] = "v1" if gr.beam_max_width is not None else None
    return BackendRequest(
        config=PlanData(_primitive(gr_fields)),
        native=PlanData(native),
        compatibility=provider_compatibility(provider),
        options=PlanData(options),
    )


def resolve_backend_plan(provider: Any, request: BackendRequest) -> BackendPlan:
    from .services import planning_provider

    plan = planning_provider(provider).resolve(request)
    if not isinstance(plan, BackendPlan):
        raise TypeError("Backend resolve() must return BackendPlan")
    if (plan.backend_id, plan.provider_api_version, plan.request) != (
        provider.backend_id,
        provider.api_version,
        request,
    ):
        raise ValueError("Backend resolve() changed request or provider identity")
    return plan


def normalize_worker_backend_request(
    config: Any, provider: Any, admitted_plan: BackendPlan
) -> BackendRequest:
    """Compare admission inputs while allowing native model-length autofit."""
    request = normalize_backend_request(config, provider)
    native = request.native.to_dict()
    model = native["model_config"]
    admitted_model = admitted_plan.request.native.to_dict().get("model_config", {})
    admitted_length = (
        admitted_model.get("max_model_len") if isinstance(admitted_model, Mapping) else None
    )
    actual_length = model["max_model_len"]
    if (
        type(admitted_length) is int
        and type(actual_length) is int
        and 0 < actual_length <= admitted_length
    ):
        # The plan retains the admitted limit; native load/profiling may fit a
        # smaller runtime allocation before first bind or a later full reload.
        # Do not overwrite the live config: validate_worker sees its actual size.
        # Increases, invalid lengths and every other input remain fingerprinted.
        model["max_model_len"] = admitted_length
        request = replace(request, native=PlanData(native))
    return request


def admit_backend_plan(config: Any) -> BackendPlan | None:
    """Attach only serializable plan data after EngineArgs has normalized GR."""
    from vllm_gr.config import GRConfig

    from .loader import activate_current_backend

    gr = GRConfig.from_vllm_config(config)
    if not gr.enabled and gr.beam_max_width is None:
        additional = getattr(config, "additional_config", None)
        if isinstance(additional, Mapping) and additional.get(BACKEND_OPTIONS_KEY):
            raise ValueError("Backend options require an enabled GR engine")
        if isinstance(additional, Mapping) and BACKEND_PLAN_KEY in additional:
            config.additional_config = {
                key: value for key, value in additional.items() if key != BACKEND_PLAN_KEY
            }
        return None
    provider = activate_current_backend()
    plan = resolve_backend_plan(provider, normalize_backend_request(config, provider))
    additional = getattr(config, "additional_config", None)
    if additional is not None and not isinstance(additional, Mapping):
        raise TypeError("Backend plans require mapping additional_config")
    config.additional_config = {**(additional or {}), BACKEND_PLAN_KEY: plan.to_wire()}
    return plan
