# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""
Setup script for vllm-gr with hardware-dependent installation.

This setup.py implements platform-aware dependency routing so users can run
`pip install vllm-gr` and automatically receive the correct platform-specific
dependencies (CUDA/NPU/CPU) without requiring extras like `[gpu]`.
"""

import logging
import os
import runpy
import sys
import warnings
from email.parser import Parser
from pathlib import Path

from packaging.version import Version
from setuptools import setup
from setuptools_scm import get_version

logger = logging.getLogger("vllm_gr.setup")

_VALID_DEVICES = frozenset(("cuda", "musa", "npu", "cpu"))
VLLM_GR_BASE_VERSION = "0.30.0"
ROOT_DIR = Path(__file__).resolve().parent
_SCM_OVERRIDE = "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_GR"

# Module-level cache so that detect_target_device() is evaluated exactly once
# per setup.py invocation, avoiding duplicate log output and potential
# inconsistency if env vars change between calls.
_cached_device: str | None = None


def warn_deprecated_extra_install() -> None:
    """Warn when installing via deprecated [gpu] or [npu] extras."""
    argv = " ".join(sys.argv).lower()
    for extra in ("[gpu]", "[npu]", ".[gpu", ".[npu"):
        if extra in argv:
            warnings.warn(
                f"Installing vllm-gr via the deprecated extra {extra!r} is "
                "supported for one transition release only. "
                "Use `pip install -e .` instead — platform dependencies are "
                "auto-selected by setup.py.",
                DeprecationWarning,
                stacklevel=2,
            )
            return


def detect_target_device() -> str:
    """
    Detect the target device for installation.

    Priority order:
    1. VLLM_GR_TARGET_DEVICE environment variable (highest priority)
    2. READTHEDOCS=1 -> cpu
    3. Torch backend detection (cuda, npu)
    4. No torch -> default cuda
    5. Torch present but no GPU/NPU backend -> cpu

    The result is cached after the first call.
    """
    global _cached_device
    if _cached_device is not None:
        return _cached_device

    target_device = os.environ.get("VLLM_GR_TARGET_DEVICE")
    if target_device:
        if target_device.lower() in _VALID_DEVICES:
            logger.info("Using target device from VLLM_GR_TARGET_DEVICE: %s", target_device.lower())
            _cached_device = target_device.lower()
            return _cached_device
        logger.warning(
            "Invalid VLLM_GR_TARGET_DEVICE '%s', falling back to auto-detection",
            target_device,
        )

    if os.environ.get("READTHEDOCS"):
        logger.info("Detected READTHEDOCS=1; using cpu requirements for docs build")
        _cached_device = "cpu"
        return _cached_device

    try:
        import torch

        if getattr(torch.version, "musa", None) is not None:
            logger.info("Detected MUSA backend from torch")
            _cached_device = "musa"
            return _cached_device

        if torch.version.cuda is not None:
            logger.info("Detected CUDA backend from torch")
            _cached_device = "cuda"
            return _cached_device

        if hasattr(torch, "npu"):
            try:
                if torch.npu.is_available():
                    logger.info("Detected NPU backend from torch")
                    _cached_device = "npu"
                    return _cached_device
            except Exception:
                pass

        logger.info("No GPU/NPU backend detected in torch, defaulting to CPU")
        _cached_device = "cpu"
        return _cached_device

    except ImportError:
        logger.info("PyTorch not found, defaulting to CUDA installation")
        _cached_device = "cuda"
        return _cached_device


def get_vllm_gr_version() -> str:
    """
    Get the vllm-gr version with optional device-specific suffix.

    Version format examples:
        - 0.30.0 (cuda, no suffix)
        - 0.30.0+npu
        - 0.30.0+cpu

    Environment variables:
        VLLM_GR_VERSION_OVERRIDE: Override version completely
        VLLM_GR_TARGET_DEVICE: Override device detection
    """
    override = (
        os.getenv("VLLM_GR_VERSION_OVERRIDE")
        or os.getenv(_SCM_OVERRIDE)
        or os.getenv("SETUPTOOLS_SCM_PRETEND_VERSION")
    )
    if override:
        version = override
    elif (ROOT_DIR / "PKG-INFO").is_file():
        # An sdist has already declared its distribution identity. Preserve it
        # through metadata preparation and wheel building, even on another target.
        metadata = Parser().parsestr((ROOT_DIR / "PKG-INFO").read_text(encoding="utf-8"))
        version = metadata["Version"]
        if not version:
            raise ValueError("sdist PKG-INFO is missing its Version")
    else:
        # PEP 517 executes setup.py without necessarily adding the tree to sys.path.
        version_tools = Path(__file__).resolve().parent / "tools/packaging/git_version.py"
        checkout_version = runpy.run_path(str(version_tools))["checkout_version"]
        version = checkout_version(ROOT_DIR, VLLM_GR_BASE_VERSION)
        device = detect_target_device()

        if device == "cuda":
            pass
        elif device in ("musa", "npu", "cpu"):
            sep = "+" if "+" not in version else "."
            version += f"{sep}{device}"
        else:
            raise RuntimeError(f"Unknown target device: {device}")

    # Write the final normalized version, including any device suffix.
    # Scope the override to this distribution and restore it after the write:
    # PEP 517 frontends can invoke the backend repeatedly in the same process.
    version = str(Version(version))
    previous = os.environ.get(_SCM_OVERRIDE)
    try:
        os.environ[_SCM_OVERRIDE] = version
        version = get_version(
            root=str(ROOT_DIR),
            dist_name="vllm-gr",
            write_to="vllm_gr/_version.py",
        )
    finally:
        if previous is None:
            os.environ.pop(_SCM_OVERRIDE, None)
        else:
            os.environ[_SCM_OVERRIDE] = previous
    logger.info("Generated version: %s", version)
    return version


def load_requirements(file_path: Path) -> list[str]:
    """Load requirements from a file, supporting -r directive for recursive loading."""
    requirements: list[str] = []

    if not file_path.exists():
        logger.warning("Requirements file not found: %s", file_path)
        return requirements

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r "):
                nested_path = file_path.parent / line[3:].strip()
                requirements.extend(load_requirements(nested_path))
            else:
                requirements.append(line)

    return requirements


def get_install_requires() -> list[str]:
    """Get dependencies based on detected platform."""
    warn_deprecated_extra_install()

    device = detect_target_device()
    requirements_dir = Path(__file__).parent / "requirements"
    requirements_file = requirements_dir / f"{device}.txt"

    logger.info("Loading requirements from: %s", requirements_file)
    requirements = load_requirements(requirements_file)

    if not requirements:
        logger.warning("No requirements loaded for device '%s'", device)
    else:
        logger.info("Loaded %d requirements for %s", len(requirements), device)

    return requirements


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    setup(
        version=get_vllm_gr_version(),
        install_requires=get_install_requires(),
    )
