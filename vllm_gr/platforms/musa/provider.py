# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Moore Threads backend discovery and runtime initialization."""

from dataclasses import dataclass
from typing import ClassVar

from vllm_gr.config.backend_plan import BackendPlan, BackendRequest
from vllm_gr.platforms.compat import require_runtime_version
from vllm_gr.platforms.cuda.provider import CUDABackendProvider
from vllm_gr.platforms.identity import PlatformIdentity, current_platform_identity


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

    def resolve(self, request: BackendRequest) -> BackendPlan:
        raise ValueError("The MUSA provider currently supports runtime initialization only")


MUSA_BACKEND = MUSABackendProvider()
