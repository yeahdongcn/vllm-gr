# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Synthetic platform identities for provider matching and selection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_gr.platforms.ascend import NPU_BACKEND
from vllm_gr.platforms.cuda import CUDA_BACKEND
from vllm_gr.platforms.identity import platform_identity
from vllm_gr.platforms.musa.provider import MUSA_BACKEND


class _VendorPlatform:
    """A vendor subclass that is not imported from any installed runtime."""

    _enum = "OOT"
    device_type = "npu"
    device_name = "npu"


_VendorPlatform.__module__ = "vllm_ascend.platform"
_VendorPlatform.__qualname__ = "NPUPlatform"


class _VendorSubclass(_VendorPlatform):
    pass


class _UnknownOutOfTreePlatform:
    _enum = "OOT"
    device_type = "npu"
    device_name = "npu"


_UnknownOutOfTreePlatform.__module__ = "some_vendor_accel.platform"
_UnknownOutOfTreePlatform.__qualname__ = "AccelPlatform"


class _RocmPlatform:
    """AMD ROCm shares the tensor device type with NVIDIA CUDA."""

    _enum = "ROCM"
    device_type = "cuda"
    device_name = "cuda"


def test_identity_reads_enum_placement_and_class_lineage() -> None:
    identity = platform_identity(_VendorSubclass())

    assert identity.platform_enum == "OOT"
    assert identity.device_type == "npu"
    assert identity.device_name == "npu"
    assert identity.platform_class.endswith("_VendorSubclass")
    assert identity.has_lineage("vllm_ascend.platform.NPUPlatform")
    assert "builtins.object" in identity.platform_lineage


def test_identity_is_immutable_and_hashable() -> None:
    identity = platform_identity(_VendorPlatform())

    with pytest.raises(AttributeError):
        setattr(identity, "device_type", "cuda")
    assert hash(identity)


def test_identity_tolerates_a_platform_without_enumeration() -> None:
    identity = platform_identity(SimpleNamespace(device_type=" CUDA "))

    assert identity.platform_enum == "UNSPECIFIED"
    assert identity.device_type == "cuda"
    assert identity.device_name == ""
    assert identity.has_lineage("builtins.object")


def test_cuda_provider_rejects_the_shared_rocm_placement() -> None:
    """ROCm reports ``device_type == "cuda"`` but is not the CUDA provider."""
    assert CUDA_BACKEND.matches(platform_identity(_RocmPlatform())) is False
    assert (
        CUDA_BACKEND.matches(platform_identity(SimpleNamespace(_enum="CUDA", device_type="cuda")))
        is True
    )


def test_ascend_provider_requires_the_supported_class_lineage() -> None:
    assert NPU_BACKEND.matches(platform_identity(_VendorPlatform())) is True
    assert NPU_BACKEND.matches(platform_identity(_VendorSubclass())) is True
    # Same enum and placement, different vendor: rejected by matches(), not by
    # a table inside core.
    assert NPU_BACKEND.matches(platform_identity(_UnknownOutOfTreePlatform())) is False


@pytest.mark.l1
@pytest.mark.cpu_test
def test_musa_provider_requires_native_musa_platform_identity() -> None:
    class MUSAPlatformBase:
        _enum = "OOT"
        device_type = "musa"

    MUSAPlatformBase.__module__ = "vllm_musa.platform"
    MUSAPlatformBase.__qualname__ = "MUSAPlatformBase"

    class MtmlMUSAPlatform(MUSAPlatformBase):
        pass

    native_platform = MtmlMUSAPlatform()
    assert MUSA_BACKEND.matches(platform_identity(native_platform))
    assert not MUSA_BACKEND.matches(
        platform_identity(SimpleNamespace(_enum="OOT", device_type="musa"))
    )

    native_platform._enum = "CUDA"
    assert not MUSA_BACKEND.matches(platform_identity(native_platform))

    native_platform._enum = "OOT"
    native_platform.device_type = "cuda"
    assert not MUSA_BACKEND.matches(platform_identity(native_platform))


def test_identity_describe_names_the_selection_inputs() -> None:
    identity = platform_identity(_RocmPlatform())

    described = identity.describe()
    assert "platform='ROCM'" in described
    assert "device_type='cuda'" in described
    assert "_RocmPlatform" in described


def test_identity_record_is_import_light() -> None:
    """Building an identity must not import a vendor runtime or torch."""
    import subprocess
    import sys
    from pathlib import Path

    code = """
import sys
from types import SimpleNamespace

from vllm_gr.platforms.identity import platform_identity

identity = platform_identity(SimpleNamespace(_enum="CUDA", device_type="cuda"))
assert identity.platform_enum == "CUDA"
for forbidden in ("torch", "torch_npu", "vllm", "vllm_ascend"):
    assert forbidden not in sys.modules, forbidden
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_matches_is_the_only_acceptance_point_for_a_synthetic_identity() -> None:
    """A synthetic identity selects without core holding a platform table."""
    from vllm_gr.platforms.loader import load_backend_plugins
    from vllm_gr.platforms.registry import (
        _reset_backend_registry_for_tests,
        select_backend,
    )

    _reset_backend_registry_for_tests()
    try:
        load_backend_plugins()
        identity = platform_identity(_VendorSubclass())
        assert select_backend(identity).backend_id == "ascend"
    finally:
        _reset_backend_registry_for_tests()


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
