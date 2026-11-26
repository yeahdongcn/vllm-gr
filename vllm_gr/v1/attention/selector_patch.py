# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Hardware-neutral CUSTOM attention selection for GR runtime initialization."""

from typing import Any, cast

from vllm_gr.config import GRAttentionBackend, GRConfig

_ORIGINAL_ATTR = "_vllm_gr_custom_attention_original"


def _get_current_gr_attention_backend() -> GRAttentionBackend | None:
    from vllm.config import get_current_vllm_config_or_none

    return GRConfig.from_vllm_config(get_current_vllm_config_or_none()).attention_backend


def _get_custom_beam_backend() -> type:
    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

    return cast(type[Any], BeamAttentionBackend)


def clear_attention_backend_cache() -> None:
    """Clear native selection results while preserving the live GR wrapper."""
    import vllm.v1.attention.selector as selector_mod

    current = selector_mod._cached_get_attn_backend
    native = getattr(current, _ORIGINAL_ATTR, current)
    native.cache_clear()


def apply_custom_attention_backend_patch(force: bool = False) -> int:
    """Preserve CUSTOM intent without importing hardware execution modules.

    ``force`` is accepted only for compatibility with existing callers and has
    no effect. Both values inspect the live selector on every call. Keep an
    installed wrapper; if upstream replaces the target, wrap the new callable
    rather than restoring a stale saved original.
    """
    del force
    try:
        import vllm.v1.attention.selector as selector_mod
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except ImportError:
        return 0

    original = selector_mod._cached_get_attn_backend
    if _ORIGINAL_ATTR in getattr(original, "__dict__", {}):
        return 0

    def _patched_cached_get_attn_backend(
        backend: Any,
        attn_selector_config: Any,
        num_heads: int | None = None,
        **selector_options: Any,
    ) -> Any:
        # Preserve the original CUSTOM import error rather than hiding it.
        if backend == AttentionBackendEnum.CUSTOM:
            return _get_custom_beam_backend()
        if backend is None and _get_current_gr_attention_backend() is GRAttentionBackend.CUSTOM:
            return _get_custom_beam_backend()
        return original(backend, attn_selector_config, num_heads=num_heads, **selector_options)

    setattr(_patched_cached_get_attn_backend, _ORIGINAL_ATTR, original)
    selector_mod._cached_get_attn_backend = _patched_cached_get_attn_backend
    return 1
