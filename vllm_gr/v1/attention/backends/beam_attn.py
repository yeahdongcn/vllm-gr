# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Adapted from https://github.com/vllm-project/vllm

"""Beam attention over a shared paged prefix and persistent per-beam suffix KV."""

from functools import cached_property
from typing import Any, ClassVar

import torch
from vllm import envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_gr.utils.npu_adapter import is_cuda_available, is_npu_available
from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BeamAttentionMetadata,
    BeamAttentionMetadataBuilder,
    require_beam_decode_session,
)
from vllm_gr.v1.attention.backends.beam_attn_metadata_tensors import (
    _get_fp8_dtype_for_flashattn,
)

__all__ = [
    "BeamAttentionBackend",
    "BeamAttentionImpl",
    "BeamAttentionMetadata",
    "BeamAttentionMetadataBuilder",
]

logger = init_logger(__name__)
NPU_DEVICE = is_npu_available()
CUDA_DEVICE = is_cuda_available()

# Worker calls outside a vLLM context reuse the most recently bound engine mode.
_NPU_DECODE_EXECUTION_MODE: str | None = None


def _get_npu_decode_execution() -> str:
    """Lazily resolve ``vllm_gr_npu_decode_execution`` (cascade | xllm).

    An unset config (None) resolves per platform: ``xllm`` on NPU when the
    xllm-ops operator library is installed, ``cascade`` everywhere else (CUDA,
    NPU without the library, or config not yet installed in this process).
    Explicit ``cascade`` / ``xllm`` values always win.
    """
    global _NPU_DECODE_EXECUTION_MODE
    from vllm.config import get_current_vllm_config_or_none

    from vllm_gr.config import GRConfig

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return _NPU_DECODE_EXECUTION_MODE or "cascade"
    bound = getattr(vllm_config, "_gr_npu_decode_execution", None)
    if bound is not None:
        _NPU_DECODE_EXECUTION_MODE = bound
        return bound
    configured = GRConfig.from_vllm_config(vllm_config).npu_decode_execution
    if configured is not None:
        mode = configured.value
    elif NPU_DEVICE and not CUDA_DEVICE:
        from vllm_gr.ops.npu import beam_ops

        mode = "xllm" if beam_ops.is_available() else "cascade"
        if mode == "cascade":
            logger.info(
                "vllm_gr_npu_decode_execution is unset and the xllm-ops library "
                "is unavailable (%s); defaulting to the cascade decode path. "
                "Install xllm-ops (tools/npu/install_xllm_ops.sh) to enable the "
                "fused xllm decode path by default.",
                beam_ops.unavailable_reason(),
            )
    else:
        mode = "cascade"
    object.__setattr__(vllm_config, "_gr_npu_decode_execution", mode)
    _NPU_DECODE_EXECUTION_MODE = mode
    return mode


def _get_platform_execution_module() -> Any:
    from vllm_gr.platforms.errors import BackendNotActiveError
    from vllm_gr.platforms.registry import get_active_backend

    if CUDA_DEVICE and not NPU_DEVICE and _get_npu_decode_execution() == "xllm":
        raise RuntimeError("xllm decode requires an Ascend NPU")
    try:
        provider = get_active_backend()
    except BackendNotActiveError:
        # Standalone backend probes can precede general-plugin initialization.
        pass
    else:
        return provider.get_attention_execution()

    if NPU_DEVICE and not CUDA_DEVICE:
        from vllm_gr.platforms.ascend import NPU_BACKEND

        return NPU_BACKEND.get_attention_execution()
    if CUDA_DEVICE and not NPU_DEVICE:
        from vllm_gr.platforms.cuda import CUDA_BACKEND

        return CUDA_BACKEND.get_attention_execution()
    raise RuntimeError(
        "BeamAttention requires exactly one active CUDA or NPU device "
        f"(CUDA_DEVICE={CUDA_DEVICE}, NPU_DEVICE={NPU_DEVICE})."
    )


def _resolve_dense_prefill_mode() -> str | None:
    """Opt-in dense CUDA Prefill mode, or None to keep the paged path.

    Captured Prefill graphs hold fresh prompts only when prefix caching,
    chunked Prefill and KV connectors are all off; otherwise a request can carry
    context that only the paged cache holds, so the mode is ignored. Each call
    still checks its CPU lengths (eager batches and non-Beam decode rows also
    reach the Prefill entry).
    """
    if not CUDA_DEVICE or envs.VLLM_BATCH_INVARIANT:
        # The paged path pins num_splits=1; cuDNN-vs-FA2 selection by request
        # count would make results depend on batch composition.
        return None
    from vllm.config import get_current_vllm_config_or_none

    from vllm_gr.config.models import deployment_for

    config = get_current_vllm_config_or_none()
    if (
        config is None
        or config.cache_config.enable_prefix_caching
        or config.scheduler_config.enable_chunked_prefill
        # A KV connector can load earlier prompt tokens into the paged cache.
        or getattr(config, "kv_transfer_config", None) is not None
    ):
        return None
    mode = deployment_for(config).dense_prefill
    return mode if mode != "off" else None


def _dense_prefill_kwargs(
    impl: Any, key: torch.Tensor | None, value: torch.Tensor | None
) -> dict[str, Any]:
    """Fresh K/V for the dense Prefill, or nothing to keep the paged path.

    Uses getattr defaults because tests and wrappers drive ``prefill_forward``
    with partially constructed implementations.
    """
    mode = getattr(impl, "_dense_prefill_mode", None)
    if (
        mode is None
        # The paged path attends over quantized K/V; keep its numerics.
        or is_quantized_kv_cache(getattr(impl, "kv_cache_dtype", "auto"))
        or key is None
        or value is None
        or getattr(impl, "kv_sharing_target_layer_name", None) is not None
    ):
        return {}
    return {"dense_key": key, "dense_value": value, "dense_mode": mode}


class BeamAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    @cached_property
    def execution(self) -> Any:
        """Bootstrap implementation until this layer's runner publishes its binding."""
        return _get_platform_execution_module()

    def validate_execution(self, execution: Any) -> None:
        """Verify the loaded layer can use this executor before publishing it."""
        for primitive in (
            "cascade_attention",
            "reshape_and_cache",
            "cache_unshared_kv",
            "prefill_attention",
        ):
            if not callable(getattr(execution, primitive, None)):
                raise RuntimeError(f"Beam attention executor is missing {primitive}()")
        if execution is self.execution:
            return
        version = execution.get_flash_attn_version(
            requires_alibi=self.alibi_slopes is not None, head_size=self.head_size
        )
        if version != self.vllm_flash_attn_version:
            raise RuntimeError(
                "Bound attention executor differs from the loaded layer's FlashAttention version"
            )
        if self.sinks is not None and not execution.supports_sinks():
            raise RuntimeError("Bound attention executor does not support the loaded layer's sinks")

    def _kv_cache_views(
        self, kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(kv_cache, tuple):
            return kv_cache
        # vLLM 0.30 allocates a logical [B, H, N, K+V] view, with
        # layout-dependent strides. The kernels consume [B, N, H, D]
        # views into that same allocation, including interleaved layers.
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        return key_cache, value_cache

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        num_dcp_pcp_tokens=None,
        draft_attn_metadatas=None,
    ):
        """Refresh ordinary FIA tasks; Beam uses persistent device buffers.

        Beam decode refreshes persistent fixed-address device buffers before
        replay. NPU prefill updates its native attention task groups directly
        in ``NPUPrefillGraphRunner`` and does not use this shared hook.
        """
        if NPU_DEVICE and not CUDA_DEVICE:
            from vllm_gr.v1.beam import BEAM_DATA_VAR

            info = BEAM_DATA_VAR.get()
            if not isinstance(info, dict) or info.get("beam_session") is None:
                from vllm_gr.v1.attention.backends.npu_standard_graph import (
                    update_standard_graph_params,
                )

                update_standard_graph_params(update_stream, forward_context, num_tokens)

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        if NPU_DEVICE and not CUDA_DEVICE:
            unsupported = []
            if sliding_window is not None:
                unsupported.append("sliding_window")
            if alibi_slopes is not None:
                unsupported.append("alibi_slopes")
            if logits_soft_cap not in (None, 0.0):
                unsupported.append("logits_soft_cap")
            if unsupported:
                raise NotImplementedError(
                    "NPU CUSTOM Beam Attention does not support "
                    + ", ".join(unsupported)
                    + "; neither the cascade nor xllm execution path implements these settings."
                )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self._dense_prefill_mode = _resolve_dense_prefill_mode()
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.attn_type = attn_type
        platform_execution = self.execution
        self.vllm_flash_attn_version = platform_execution.get_flash_attn_version(
            requires_alibi=alibi_slopes is not None,
            head_size=head_size,
        )
        self.batch_invariant_enabled = envs.VLLM_BATCH_INVARIANT if CUDA_DEVICE else False

        if is_quantized_kv_cache(self.kv_cache_dtype):
            if NPU_DEVICE:
                raise NotImplementedError("BeamAttention does not support fp8 kv-cache on NPU.")
            if not BeamAttentionBackend.supports_kv_cache_dtype(self.kv_cache_dtype):
                raise NotImplementedError(
                    "BeamAttention does not support fp8 kv-cache on this device."
                )

        self.sinks = sinks
        if self.sinks is not None:
            if NPU_DEVICE:
                raise NotImplementedError("Sinks are not yet supported on NPU.")
            assert platform_execution.supports_sinks(), (
                "Sinks are only supported in FlashAttention 3"
            )
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of heads in the layer"
            )
        from vllm.config import get_current_vllm_config_or_none

        from vllm_gr.config import GRConfig

        config = GRConfig.from_vllm_config(get_current_vllm_config_or_none())
        short_suffix_attention = (
            CUDA_DEVICE
            and config.beam_max_width is not None
            and config.enable_short_suffix_attention
        )
        if short_suffix_attention:
            from vllm_gr.v1.attention.backends.beam_attn_gpu import (
                validate_short_suffix_attention,
            )

            validate_short_suffix_attention(
                num_heads,
                num_kv_heads,
                head_size,
                self.vllm_flash_attn_version,
                self.sliding_window,
                self.logits_soft_cap,
                self.alibi_slopes,
                self.sinks,
            )
            logger.info_once(
                "Beam prefix uses FlashAttention %s with canonical out/LSE; "
                "short-suffix fusion is enabled independently.",
                self.vllm_flash_attn_version,
            )
        # Separate prefill from the version that defines decode prefix/LSE
        # semantics. Resolve before capture; never change the global config.
        self.prefill_flash_attn_version = self.vllm_flash_attn_version
        if not CUDA_DEVICE and config.prefill_flash_attn_version not in (None, "auto"):
            raise ValueError("beam.prefill_flash_attn_version is CUDA-only")
        if CUDA_DEVICE and config.prefill_flash_attn_version is not None:
            from vllm_gr.v1.attention.backends.beam_attn_gpu import (
                resolve_prefill_flash_attn_version,
            )

            assert self.vllm_flash_attn_version is not None
            self.prefill_flash_attn_version = resolve_prefill_flash_attn_version(
                config.prefill_flash_attn_version,
                self.vllm_flash_attn_version,
                head_size=head_size,
                kv_cache_dtype=kv_cache_dtype,
                requires_alibi=self.alibi_slopes is not None,
                has_sinks=self.sinks is not None,
                batch_invariant=self.batch_invariant_enabled,
            )
            logger.info_once(
                "Beam attention: requested prefill=%s, effective prefill FA%s, decode FA%s",
                config.prefill_flash_attn_version,
                self.prefill_flash_attn_version,
                self.vllm_flash_attn_version,
            )
        # The fused suffix consumes model-dtype queries, not quantized Q.
        self.supports_quant_query_input = not short_suffix_attention
        self.use_cascade = True

    def _prefill_cascade_kwargs(self) -> dict[str, int]:
        version = self.prefill_flash_attn_version
        if CUDA_DEVICE and version is not None and version != self.vllm_flash_attn_version:
            return {"prefill_flash_attn_version": version}
        return {}

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: BeamAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if CUDA_DEVICE:
            assert self.vllm_flash_attn_version is not None, "FlashAttention version not detected."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for BeamAttentionImpl"
            )
        if attn_metadata is None:
            return output.fill_(0)

        if getattr(attn_metadata, "is_prefill_graph_exec", False):
            return self.prefill_forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )

        platform_execution = self.execution

        key_cache, value_cache = self._kv_cache_views(kv_cache)

        # --- Incremental decode path for beam decode ---
        # Beam decode requires a live session: append current KV to its pool
        # and run cascade attention over the paged prefix and pooled suffix.
        beam_session = getattr(attn_metadata, "beam_session", None)
        is_beam_decode = getattr(attn_metadata, "is_beam_decode", False)

        if is_beam_decode or getattr(attn_metadata, "prefix_indices", None) is not None:
            beam_session = require_beam_decode_session(attn_metadata)
            # Cache only the layer index. The pool object is session-specific:
            # multi-request GPU execution replaces each physical [B, W, ...]
            # pool with a request-major [1, B*W, ...] view. Reusing the pool
            # cached by the startup B=1 capture would bypass that view and make
            # later kernels address neighbouring request slots by logically
            # indexing past W.
            layer_idx = getattr(layer, "_beam_pool_index", None)
            if layer_idx is None:
                layer_name = getattr(layer, "layer_name", "")
                layer_idx = extract_layer_index(layer_name)
                layer._beam_pool_index = layer_idx
            pools = getattr(beam_session, "pools", ())
            if not 0 <= layer_idx < len(pools) or pools[layer_idx] is None:
                raise RuntimeError("Beam decode requires a KV pool for the attention layer")
            pool = pools[layer_idx]

            # Extract the beam tokens' KV from the key/value tensors (they
            # contain all scheduled tokens).  The gather runs inside the
            # captured graph: fancy indexing allocates from the graph pool
            # (address-stable across replays) and re-gathers from the
            # fixed-address prefix_indices buffer each replay.
            if attn_metadata.prefix_indices is not None:
                # Beam tokens at prefix_indices positions (always set on the
                # incremental decode path, mixed or pure batch alike).
                select_rows = (
                    getattr(platform_execution, "select_beam_rows", None)
                    if key.device.type == "npu"
                    else None
                )
                if select_rows is not None:
                    beam_key = select_rows(key, attn_metadata)
                    beam_value = select_rows(value, attn_metadata)
                else:
                    beam_key = key[attn_metadata.prefix_indices]  # [W, H, D]
                    beam_value = value[attn_metadata.prefix_indices]  # [W, H, D]
            else:
                # Fallback kept for callers that do not set prefix_indices;
                # the incremental path always sets it (see beam_attn_metadata).
                beam_key = key
                beam_value = value

            # cache_unshared_kv: write current step's KV into pool.
            # Use Graph-compatible Triton kernel when decode_step_buf is
            # available on CUDA (reads decode_step from device buffer instead
            # of host Python int, enabling CUDA Graph capture/replay).
            decode_step_buf = getattr(beam_session.context, "decode_step_buf", None)
            if decode_step_buf is not None and decode_step_buf.is_cuda:
                from vllm_gr.ops.cuda.beam_kv_cache import cache_unshared_kv_graph

                cache_unshared_kv_graph(
                    pool.unshared_key,
                    pool.unshared_value,
                    beam_key,
                    beam_value,
                    beam_session.slot,
                    decode_step_buf,
                    active_mask=getattr(attn_metadata, "beam_active_mask", None),
                )
            else:
                # Non-CUDA (NPU xllm/cascade, CPU unit tests) dispatches through
                # the platform module. Cascade uses the shared pool writer.
                # xLLM needs the session's fixed-address device buffers for
                # ACL graph capture/replay.
                platform_execution.cache_unshared_kv(
                    pool.unshared_key,
                    pool.unshared_value,
                    beam_key,
                    beam_value,
                    beam_session.slot,
                    beam_session.decode_step,
                    beam_session=beam_session,
                )

            # The suffix pool stores unquantized model activations. Its dtype
            # follows the current attention K/V, independently of a possibly
            # FP8 paged prefix cache.
            sliding_window_size = (
                list(self.sliding_window) if self.sliding_window is not None else None
            )

            # cascade_attention with pool slice (incremental path)
            platform_execution.cascade_attention(
                output=output,
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                attn_metadata=attn_metadata,
                sliding_window_size=sliding_window_size,
                scale=self.scale,
                logits_soft_cap=self.logits_soft_cap,
                vllm_flash_attn_version=self.vllm_flash_attn_version,
                s_aux=self.sinks,
                alibi_slopes=self.alibi_slopes,
                beam_pool=pool,
                **self._prefill_cascade_kwargs(),
            )

            return output

        # --- Standard path (non-beam or prefill) ---
        if self.kv_sharing_target_layer_name is None and key is not None and value is not None:
            if NPU_DEVICE:
                k_scale = None
                v_scale = None
            else:
                k_scale = layer._k_scale
                v_scale = layer._v_scale
            platform_execution.reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                k_scale,
                v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            dtype = BeamAttentionBackend.get_fp8_dtype_for_flashattn(self.kv_cache_dtype)
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        sliding_window_size = list(self.sliding_window) if self.sliding_window is not None else None

        platform_execution.cascade_attention(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            attn_metadata=attn_metadata,
            sliding_window_size=sliding_window_size,
            scale=self.scale,
            logits_soft_cap=self.logits_soft_cap,
            vllm_flash_attn_version=self.vllm_flash_attn_version,
            s_aux=self.sinks,
            alibi_slopes=self.alibi_slopes,
            **self._prefill_cascade_kwargs(),
        )

        return output

    def prefill_forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: BeamAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Streamlined fast-path for pure prefill execution in Full Graph mode."""
        platform_execution = self.execution

        # Validate the captured layout before submitting even the cache write.
        # A BSND dummy has no block table and cannot fall through to paged TND.
        use_bsnd = NPU_DEVICE and getattr(attn_metadata, "prefill_fia_layout", "TND") == "BSND"
        if use_bsnd:
            batch_size = attn_metadata.num_reqs
            lengths = attn_metadata.seq_lens_cpu
            tokens = query.shape[0]
            if (
                batch_size <= 0
                or tokens <= 0
                or tokens % batch_size
                or lengths is None
                or len(lengths) != batch_size
                or any(length != tokens // batch_size for length in lengths)
            ):
                raise ValueError(
                    "BSND prefill requires fixed [P]*B metadata; cannot fall back to TND"
                )
            if key is None or value is None or self.kv_sharing_target_layer_name is not None:
                raise ValueError("BSND prefill requires this layer's key/value without KV sharing")
            if key.shape[0] != tokens or value.shape[0] != tokens:
                raise ValueError("BSND prefill Q/K/V token counts must match")
            if not attn_metadata.causal or attn_metadata.npu_causal_atten_mask is None:
                raise ValueError("BSND padded prefill requires the static causal mask")
            if (
                self.sliding_window not in (None, (-1, -1))
                or self.logits_soft_cap
                or self.sinks is not None
                or self.alibi_slopes is not None
                or self.kv_cache_dtype.startswith("fp8")
            ):
                raise ValueError("BSND prefill does not support this attention configuration")

        key_cache, value_cache = self._kv_cache_views(kv_cache)

        if self.kv_sharing_target_layer_name is None and key is not None and value is not None:
            k_scale = None if NPU_DEVICE else getattr(layer, "_k_scale", None)
            v_scale = None if NPU_DEVICE else getattr(layer, "_v_scale", None)
            platform_execution.reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                k_scale,
                v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            dtype = BeamAttentionBackend.get_fp8_dtype_for_flashattn(self.kv_cache_dtype)
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        # Both warmup and capture use current-layer K/V; replay runs this graph.
        if use_bsnd:
            platform_execution.prefill_attention_bsnd(
                output=output,
                query=query,
                key=key,
                value=value,
                attn_metadata=attn_metadata,
                scale=self.scale,
            )
            return output

        sliding_window_size = list(self.sliding_window) if self.sliding_window is not None else None

        capture_state = getattr(platform_execution, "_GRAPH_CAPTURE_STATE", None)
        if NPU_DEVICE and not CUDA_DEVICE and capture_state is not None:
            from vllm.config import CUDAGraphMode
            from vllm.forward_context import get_forward_context

            from vllm_gr.v1.beam import BEAM_DATA_VAR

            context = get_forward_context()
            beam_info = BEAM_DATA_VAR.get()
            if (
                context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                and context.capturing
                and not capture_state["capturing"]
                and (not isinstance(beam_info, dict) or beam_info.get("beam_session") is None)
            ):
                from vllm_gr.v1.attention.backends.npu_standard_graph import (
                    capture_standard_fia,
                )

                capture_standard_fia(
                    output,
                    query,
                    key_cache,
                    value_cache,
                    attn_metadata,
                    self.scale,
                    layer.layer_name,
                    context.model_instance.vllm_config.model_config.max_model_len,
                )
                return output

        # Dense Prefill is CUDA-only; NPU never resolves the helper.
        dense_kwargs = _dense_prefill_kwargs(self, key, value) if CUDA_DEVICE else {}
        platform_execution.prefill_attention(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            attn_metadata=attn_metadata,
            sliding_window_size=sliding_window_size,
            scale=self.scale,
            logits_soft_cap=self.logits_soft_cap,
            vllm_flash_attn_version=self.prefill_flash_attn_version,
            s_aux=self.sinks,
            alibi_slopes=self.alibi_slopes,
            **dense_kwargs,
        )

        return output


@register_backend(AttentionBackendEnum.CUSTOM)
class BeamAttentionBackend(AttentionBackend):
    accept_output_buffer = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]

    @staticmethod
    def get_supported_kernel_block_sizes(spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        # Preserve the Ascend-compatible block size that
        # the Ascend provider applies to the running config (128).
        # vLLM 0.30's platform hook otherwise resolves an unspecified size back
        # to a 16-multiple geometry that the Ascend kernels and prefix-caching
        # block tables cannot represent.
        from vllm.config.vllm import get_current_vllm_config_or_none

        current = get_current_vllm_config_or_none()
        if current is not None and cls.supports_block_size(current.cache_config.block_size):
            return current.cache_config.block_size
        return super().get_preferred_block_size(default_block_size)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        # vLLM 0.30 dropped this from the base class; vllm-ascend's NPU runner
        # calls it on the selected backend to size the pooled K/V cache, so the
        # shape must match what the Ascend kernels and this module's gather
        # helpers expect: a leading "2" that splits K from V, then a
        # token-major block view. Mirror vllm-ascend's own layout switch so an
        # explicit VLLM_KV_CACHE_LAYOUT=LBHNC/HND stays consistent with it.
        from vllm import envs as vllm_envs

        if vllm_envs.VLLM_KV_CACHE_LAYOUT in ("LBHNC", "HND"):
            return (2, num_blocks, num_kv_heads, block_size, head_size)
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return CUDA_DEVICE

    @staticmethod
    def get_impl_cls() -> type["BeamAttentionImpl"]:
        return BeamAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["BeamAttentionMetadataBuilder"]:
        return BeamAttentionMetadataBuilder

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # Cache writes and paged prefix FlashAttention consume strides.
        # Prefer token-major storage, matching the former NHD default.
        return (
            KVCacheLayout.LBNHC,
            KVCacheLayout.LBHNC,
            KVCacheLayout.BLNHC,
            KVCacheLayout.BLHNC,
            KVCacheLayout.BHLNC,
            KVCacheLayout.LHBNC,
        )

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        return _get_fp8_dtype_for_flashattn(kv_cache_dtype)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            if NPU_DEVICE:
                return False  # fp8 kv-cache not yet supported on NPU
            if not CUDA_DEVICE:
                return False
            platform_execution = _get_platform_execution_module()
            return (
                platform_execution.get_flash_attn_version() == 3
                and current_platform.is_device_capability_family(90)
            )
        return kv_cache_dtype in ["auto", "float16", "bfloat16"]

    @classmethod
    def supports_sink(cls) -> bool:
        if NPU_DEVICE:
            return False  # sinks not yet supported on NPU
        if not CUDA_DEVICE:
            return False
        platform_execution = _get_platform_execution_module()
        return platform_execution.is_available() and platform_execution.supports_sinks()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # NPU does not use CUDA-style compute capability; accept it.
        if NPU_DEVICE:
            return True
        return capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_combination(
        cls,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_mla,
        has_sink,
        use_sparse,
        use_mm_prefix,
        device_capability,
    ) -> str | None:
        if not NPU_DEVICE:
            if not CUDA_DEVICE:
                return "No FlashAttention varlen support."
            if not _get_platform_execution_module().is_available():
                return "No FlashAttention varlen support."
        if dtype not in cls.supported_dtypes:
            return f"Unsupported dtype {dtype}."
        if not cls.supports_head_size(head_size):
            return f"Unsupported head_size={head_size}."
        if not NPU_DEVICE and not cls.supports_kv_cache_dtype(kv_cache_dtype):
            return f"Unsupported kv_cache_dtype={kv_cache_dtype}."
        if block_size is not None and block_size % 16 != 0:
            return f"block_size={block_size} not multiple of 16."
        if not cls.supports_compute_capability(device_capability):
            return "Requires GPU compute capability >= 8.0."
        if not NPU_DEVICE and has_sink and not cls.supports_sink():
            return "No FlashAttention sink support."
        if use_mla:
            return "No MLA support."
        if use_sparse:
            return "No sparse support."
        if use_mm_prefix:
            return "No multimodal prefix support."
        return None
