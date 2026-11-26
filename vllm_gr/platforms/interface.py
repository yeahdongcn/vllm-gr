# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Hardware-neutral contracts for vllm-gr backend providers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch

    from vllm_gr.config.backend_plan import BackendPlan, BackendRequest
    from vllm_gr.constraints.execution import ConstrainedTopKExecution

    from .config import BackendConfigService
    from .identity import PlatformIdentity

#: Provider API implemented by built-in and external providers in this release.
PROVIDER_API_VERSION = 2

#: API version assumed for entry points that do not declare one.
LEGACY_PROVIDER_API_VERSION = 1


@runtime_checkable
class ConfigProvider(Protocol):
    """Optional host policy for native defaults and engine admission."""

    def create_config_service(self) -> BackendConfigService: ...


class BeamKVReorder(Protocol):
    def __call__(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot: int,
        parent_ids: torch.Tensor,
        decode_step: int,
        active_mask: torch.Tensor,
    ) -> None: ...


@runtime_checkable
class BeamKVReorderProvider(Protocol):
    def get_beam_kv_reorder(self) -> BeamKVReorder: ...


class FullVocabularySelector(Protocol):
    """Write Beam candidates into the stage runner's preallocated buffers."""

    def __call__(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        token_ids: torch.Tensor,
        logprobs: torch.Tensor,
        valid_mask: torch.Tensor,
        valid_counts: torch.Tensor,
    ) -> None: ...


@runtime_checkable
class FullVocabularyProvider(Protocol):
    """Optional replacement for the default GPU full-vocabulary selector."""

    def get_full_vocabulary_selector(self) -> FullVocabularySelector: ...


class ModelRunnerPatchHooks(Protocol):
    """Required hooks for the V1 ModelRunner patch dispatcher.

    Additional stage sampling and model-forward factories are optional
    capabilities discovered by the dispatcher. Both platforms retain native
    sampling and bookkeeping for ordinary inference.
    """

    def discover_worker_runners(self) -> tuple[tuple[str, type[Any]], ...]: ...

    def make_patched_prepare_inputs(
        self,
        original: Callable[..., Any],
    ) -> Callable[..., Any]: ...


@runtime_checkable
class ModelRunnerInstaller(Protocol):
    """Install and verify the selected native runner's own method surface."""

    def discover_worker_runners(self) -> tuple[tuple[str, type[Any]], ...]: ...

    def install(self) -> None: ...

    def is_patched(self, runner_class: type[Any]) -> bool: ...


class ModelRunnerAdapter(Protocol):
    """Per-runner callbacks with the selected native signatures and return values."""

    def prepare_inputs(self, *args: Any, **kwargs: Any) -> Any: ...

    def execute_model(self, *args: Any, **kwargs: Any) -> Any: ...

    # Some native runners complete sampling entirely inside execute_model.
    @property
    def sample_tokens(self) -> Callable[..., Any] | None: ...


class BeamAttentionExecution(Protocol):
    """Platform-specific Beam Attention execution primitives."""

    def is_available(self) -> bool: ...

    def get_flash_attn_version(
        self,
        *,
        requires_alibi: bool = False,
        head_size: int | None = None,
    ) -> int | None: ...

    def supports_sinks(self) -> bool: ...

    def reshape_and_cache(
        self,
        key: Any,
        value: Any,
        key_cache: Any,
        value_cache: Any,
        slot_mapping: Any,
        kv_cache_dtype: str,
        k_scale: Any,
        v_scale: Any,
    ) -> None: ...

    def cache_unshared_kv(
        self,
        unshared_key: Any,
        unshared_value: Any,
        curr_key: Any,
        curr_value: Any,
        slot: int,
        decode_step: int,
        beam_session: Any = None,
    ) -> None: ...

    def prefill_attention(
        self,
        output: Any,
        query: Any,
        key_cache: Any,
        value_cache: Any,
        attn_metadata: Any,
        sliding_window_size: list[int] | tuple[int, int] | None,
        scale: float,
        logits_soft_cap: float,
        vllm_flash_attn_version: int,
        s_aux: Any = None,
        alibi_slopes: Any = None,
    ) -> None: ...

    def cascade_attention(
        self,
        output: Any,
        query: Any,
        key_cache: Any,
        value_cache: Any,
        attn_metadata: Any,
        sliding_window_size: list[int] | tuple[int, int] | None,
        scale: float,
        logits_soft_cap: float,
        vllm_flash_attn_version: int,
        s_aux: Any = None,
        alibi_slopes: Any = None,
        beam_pool: Any = None,  # Per-layer pool (GPU incremental decode only)
    ) -> None: ...


class GRBackendProvider(Protocol):
    """Process-local provider for one stable GR backend identity."""

    @property
    def backend_id(self) -> str: ...

    @property
    def device_type(self) -> str:
        """Tensor placement of the supported platform; not the identity."""

    @property
    def api_version(self) -> int: ...

    @property
    def gr_version_range(self) -> str | None: ...

    @property
    def vllm_version_range(self) -> str | None: ...

    @property
    def vendor_version_range(self) -> str | None: ...

    @property
    def vendor_distribution(self) -> str | None: ...

    def matches(self, identity: PlatformIdentity) -> bool: ...

    def check_environment(self) -> dict[str, str]: ...

    def apply_runtime_patches(self, *, force: bool = False) -> int: ...

    def get_model_runner_hooks(self) -> ModelRunnerInstaller | ModelRunnerPatchHooks: ...

    def get_attention_execution(self) -> BeamAttentionExecution: ...

    def create_constrained_topk_execution(
        self,
        *,
        device: str | torch.device | None = None,
    ) -> ConstrainedTopKExecution | None: ...


class GRPlanningProvider(GRBackendProvider, Protocol):
    """Planning extension for provider API version 2.

    Providers without plan_api_version keep the narrow API-v2 delegation path.
    New factories receive an immutable engine plan, never provider.current_plan.
    """

    plan_api_version: int
    adapter_revision: str

    def resolve(self, request: BackendRequest) -> BackendPlan: ...

    def validate_worker(self, *, plan: BackendPlan, native_runner: Any) -> None: ...

    def create_model_runner_adapter(self, *, plan: BackendPlan, native_runner: Any) -> Any: ...

    def create_attention_executor(self, *, plan: BackendPlan, device: Any) -> Any: ...

    def create_constrained_topk_execution(
        self, *, plan: BackendPlan | None = None, device: str | torch.device | None = None
    ) -> ConstrainedTopKExecution | None: ...
