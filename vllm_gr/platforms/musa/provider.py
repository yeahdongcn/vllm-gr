# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Moore Threads backend discovery and runtime initialization."""

from dataclasses import dataclass
from typing import ClassVar

from vllm_gr.config.backend_plan import BackendPlan, BackendRequest
from vllm_gr.platforms.compat import require_runtime_version
from vllm_gr.platforms.cuda.provider import CUDABackendProvider
from vllm_gr.platforms.identity import PlatformIdentity, current_platform_identity
from vllm_gr.platforms.interface import (
    BeamAttentionExecution,
    BeamKVReorder,
    FullVocabularySelector,
)


@dataclass(frozen=True)
class MUSABackendProvider(CUDABackendProvider):
    backend_id: str = "musa"
    device_type: str = "musa"
    vllm_version_range: ClassVar[str] = "==0.32.0"

    def matches(self, identity: PlatformIdentity) -> bool:
        return (
            identity.platform_enum == "OOT"
            and identity.device_type == "musa"
            and identity.has_lineage("vllm_musa.platform.MUSAPlatformBase")
        )

    matches_legacy_identity = matches

    def check_environment(self) -> dict[str, str]:
        identity = current_platform_identity()
        if not self.matches(identity):
            raise RuntimeError(f"MUSA provider requires vLLM-MUSA, got {identity.describe()}")
        return {"platform": "musa", "vllm": require_runtime_version("vllm", "0.32.0")}

    def get_beam_kv_reorder(self) -> BeamKVReorder:
        from .attention import reorder_suffix_kv

        return reorder_suffix_kv

    def get_attention_execution(self) -> BeamAttentionExecution:
        from .attention import EAGER_BEAM_ATTENTION

        return EAGER_BEAM_ATTENTION

    def get_full_vocabulary_selector(self) -> FullVocabularySelector:
        from .topk import full_vocabulary_topk_out

        return full_vocabulary_topk_out

    def resolve(self, request: BackendRequest) -> BackendPlan:
        native = request.native.to_dict()
        gr = request.config.to_dict()
        if (
            not native["model_config"]["enforce_eager"]
            or gr["beam_graph_enabled"]
            or gr["deployment"]["prefill_graph_enabled"]
        ):
            raise ValueError("MUSA Beam requires eager execution with both GR graph modes disabled")
        if native["use_v2_model_runner"] or any(
            native["parallel_config"][name] != 1
            for name in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
            )
        ):
            raise ValueError("MUSA Beam requires single-device ModelRunner V1")
        if request.has_constraint_table or gr["attention_backend"] != "CUSTOM":
            raise ValueError("MUSA Beam requires unconstrained requests and CUSTOM attention")
        return super().resolve(request)


MUSA_BACKEND = MUSABackendProvider()
