# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from pathlib import Path

from vllm_gr.config.schema import ConfigFieldSpec

CONSTRAINT_TABLE_FORMAT = "constraint_table_v1"
MULTI_TRIE_CONSTRAINT_TABLE_FORMAT = "constraint_table_v2"
CONSTRAINT_TABLE_MAX_DECODE_STEPS = 3
# Global Beam width ceiling.  This constant also gates the raw Beam operators,
# so it is raised only together with the kernels that make W > 256 correct:
# the hierarchical Top-K used to size every local list from LOCAL_TILE_SIZE
# (256) regardless of the selection width, which left the merge stage reading
# slots the local stage never wrote.  With _beam_selection_tiling shared by
# allocation and launch, a list is now at least as wide as the selection, so
# widths above the fused-kernel whitelist ({64, 128, 256}) are correct on the
# hierarchical path.
#
MAX_BEAM_WIDTH = 1024
# Unconstrained Top-K supports the full Beam capacity. Constraint-table
# operators have a separate output limit, checked when loading the backend.
MAX_CANDIDATE_WIDTH = MAX_BEAM_WIDTH
MAX_CONSTRAINT_CANDIDATE_WIDTH = 256
# Conservative startup resource policy, including the implicit max-width bank.
# This bounds retained width count; it is not a guarantee that graphs fit memory.
MAX_NPU_BEAM_GRAPH_WIDTHS = 8
SUPPORTED_BEAM_BATCH_SIZES = tuple(range(1, 9))


DEFAULT_BEAM_BATCH_SIZES = (1, 2, 4)
DEFAULT_PREFILL_TOKEN_BUCKETS = tuple(range(128, 8193, 128))


def absolute_config_path(value: str, name: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute for resolved configuration")
    # Resolve parent traversal on the driver. A published absolute path must
    # survive different mount points and symlinks on the worker unchanged.
    return str(path.resolve() if ".." in path.parts else path)


@dataclass(frozen=True, slots=True)
class DeploymentOptions:
    """Settings serialized with the engine, including request/worker defaults."""

    beam_width: tuple[int, ...] | None = None
    candidate_width: tuple[int, ...] | None = None
    result_width: int | None = None
    prefill_graph_enabled: bool = True
    dense_prefill: str = "off"
    prefill_token_buckets: tuple[int, ...] | None = DEFAULT_PREFILL_TOKEN_BUCKETS
    beam_batch_sizes: tuple[int, ...] = DEFAULT_BEAM_BATCH_SIZES
    prefill_kv_bound: int | None = None
    npu_bsnd_capture_specs: str = ""
    npu_stage_lm_head: bool = True
    head_slice: int | str = "auto"
    head_slice_min_rows: int = 1024
    head_slice_max_mb: float = 128
    unconstrained_head_layout: str | None = "transposed"
    topk_keep: int = 8
    topk_select: str = "i64"
    prefix_policy: str = "off"
    prefix_min_chain_us: float = 150.0
    prefix_calibration: bool = True
    prefix_calibration_path: str | None = None
    trie_retention_seconds: float = 60

    @classmethod
    def from_mapping(cls, data: Mapping) -> DeploymentOptions:
        if not isinstance(data, Mapping):
            raise ValueError("GR deployment settings must be a mapping")
        unknown = data.keys() - OPTION_KEYS
        if unknown:
            raise ValueError(f"Unknown gr deployment settings: {sorted(unknown, key=str)}")
        return cls(**data)

    def __post_init__(self) -> None:
        if self.beam_batch_sizes is None:
            object.__setattr__(self, "beam_batch_sizes", DEFAULT_BEAM_BATCH_SIZES)
        for key in (
            "beam_width",
            "candidate_width",
            "prefill_token_buckets",
            "beam_batch_sizes",
        ):
            value = getattr(self, key)
            if value is None:
                continue
            if (
                not isinstance(value, (list, tuple))
                or not value
                or any(type(v) is not int or v < 1 for v in value)
            ):
                raise ValueError(f"gr.{key} requires a nonempty list of positive integers")
            if key in ("prefill_token_buckets", "beam_batch_sizes") and list(value) != sorted(
                set(value)
            ):
                raise ValueError(f"gr.{key} must be strictly increasing")
            object.__setattr__(self, key, tuple(value))
        for key in (
            "result_width",
            "prefill_kv_bound",
            "topk_keep",
        ):
            value = getattr(self, key)
            if value is None and key in ("result_width", "prefill_kv_bound"):
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"gr.{key} requires a positive integer")
        if any(v > 8 for v in self.beam_batch_sizes):
            raise ValueError("GR beam batch sizes must be in 1..8")
        if self.topk_keep > 256 or self.topk_select not in ("i32", "i64", "flash"):
            raise ValueError(
                "gr.topk_keep must be in 1..256 and topk_select must be i32, i64 or flash"
            )
        for key in ("prefill_graph_enabled", "npu_stage_lm_head", "prefix_calibration"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"gr.{key} requires a boolean")
        if self.head_slice != "auto" and (type(self.head_slice) is not int or self.head_slice < 0):
            raise ValueError("gr.head_slice requires auto or a nonnegative integer")
        if type(self.head_slice_min_rows) is not int or self.head_slice_min_rows < 0:
            raise ValueError("gr.head_slice_min_rows requires a nonnegative integer")
        for name in ("head_slice_max_mb", "trie_retention_seconds", "prefix_min_chain_us"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f"gr.{name} requires a finite nonnegative number")
        if self.prefix_policy not in (
            "auto",
            "off",
        ) or self.unconstrained_head_layout not in (None, "transposed"):
            raise ValueError("Unsupported prefix_policy or unconstrained_head_layout")
        if self.dense_prefill not in ("off", "fa2", "cudnn"):
            raise ValueError("gr.dense_prefill must be off, fa2 or cudnn")
        if not isinstance(self.npu_bsnd_capture_specs, str):
            raise ValueError("gr.npu_bsnd_capture_specs requires a string")
        if self.prefix_calibration_path is not None and not isinstance(
            self.prefix_calibration_path, str
        ):
            raise ValueError("gr.prefix_calibration_path requires a path")
        if self.prefix_calibration_path is not None:
            object.__setattr__(
                self,
                "prefix_calibration_path",
                absolute_config_path(self.prefix_calibration_path, "gr.prefix_calibration_path"),
            )
        self._validate_step_widths()

    def _validate_step_widths(self) -> None:
        widths, candidates = self.beam_width, self.candidate_width
        if widths is None:
            if candidates is not None:
                raise ValueError("gr.candidate_width requires gr.beam_width")
            return
        if candidates is None or len(candidates) != len(widths):
            raise ValueError("gr.candidate_width must have one entry for each gr.beam_width entry")
        if max(widths) > MAX_BEAM_WIDTH or max(candidates) > MAX_CANDIDATE_WIDTH:
            raise ValueError(f"gr.beam_width and gr.candidate_width must be in 1..{MAX_BEAM_WIDTH}")
        if widths[0] != candidates[0]:
            raise ValueError("first gr.candidate_width must equal first gr.beam_width")
        if any(w > previous * k for previous, k, w in zip(widths, candidates[1:], widths[1:])):
            raise ValueError("gr.beam_width exceeds the number of available candidates for a step")
        if (len(set(widths)) > 1 or len(set(candidates)) > 1) and len(widths) > 3:
            raise ValueError(
                "varying gr.beam_width or gr.candidate_width supports up to three steps"
            )
        if self.result_width is not None and self.result_width > widths[-1]:
            raise ValueError("gr.result_width must not exceed the final gr.beam_width")

    def to_mapping(self) -> dict:
        return {
            f.name: list(v) if isinstance(v, tuple) else v
            for f in fields(self)
            if (v := getattr(self, f.name)) != f.default
        }


DEFAULT_DEPLOYMENT = DeploymentOptions()


OPTION_KEYS = frozenset(f.name for f in fields(DeploymentOptions))


class GRAttentionBackend(str, Enum):
    CUSTOM = "CUSTOM"


class ConstraintMode(str, Enum):
    """Resolved switch between no constraint and a canonical table."""

    NONE = "none"
    TABLE = "table"


class ConstraintBackendKind(str, Enum):
    """Resolved constraint-table placement, including automatic selection."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    NPU = "npu"


@dataclass(frozen=True, slots=True)
class ConstraintTableSpec:
    """Canonical constraint-table source; runtime loads one artifact."""

    path: str | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec("validate_constraint_table_path", canonical_path=("path",))
        },
    )
    """Absolute artifact path; YAML loading resolves relative paths first."""
    backend: ConstraintBackendKind = field(
        default=ConstraintBackendKind.AUTO,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_constraint_table_backend",
                canonical_path=("backend",),
            )
        },
    )
    """Execution placement of the constraint table."""
    max_top_k: int | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_optional_max_top_k",
                canonical_path=("max_top_k",),
            )
        },
    )
    """Candidate capacity required by table-mode deployments."""
    format: str = field(
        default=CONSTRAINT_TABLE_FORMAT,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_constraint_table_format",
                canonical_path=("format",),
            )
        },
    )
    """Supported constraint-table artifact format."""
    artifact_digest: str | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_artifact_digest",
                canonical_path=("digests", "artifact"),
            )
        },
    )
    """Expected artifact SHA-256 digest, required when enabled."""
    tokenizer_digest: str | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_tokenizer_digest",
                canonical_path=("digests", "tokenizer"),
            )
        },
    )
    """Expected tokenizer SHA-256 digest, required when enabled."""
    enabled: bool = field(
        default=False,
        metadata={"gr": ConfigFieldSpec("validate_constraint_table_enabled")},
    )
    """Enable constraint-table filtering for generation."""

    def __post_init__(self) -> None:
        from vllm_gr.config.validation import validate_config_fields

        validate_config_fields(self)
        if self.enabled:
            missing = [
                name
                for name, value in (
                    ("path", self.path),
                    ("format", self.format),
                    ("artifact_digest", self.artifact_digest),
                    ("tokenizer_digest", self.tokenizer_digest),
                )
                if value is None
            ]
            if missing:
                raise ValueError("enabled vllm-gr constraint_table requires " + ", ".join(missing))


@dataclass(frozen=True, slots=True)
class ConstraintConfig:
    """Resolved constraint mode and its optional table specification."""

    mode: ConstraintMode = ConstraintMode.NONE
    table: ConstraintTableSpec | None = None


class NPUDecodeExecution(str, Enum):
    """Execution mode for the NPU incremental beam decode branch."""

    CASCADE = "cascade"
    XLLM = "xllm"


@dataclass(frozen=True)
class GRConfig:
    """Validated vLLM-GR configuration reconstructed in each process."""

    attention_backend: GRAttentionBackend | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_attention_backend",
                canonical_path=("attention_backend",),
            )
        },
    )
    """Optional attention backend override."""
    # Beam-decode FULL CUDA graph (startup dummy capture, runtime replay-only).
    # beam_max_width is the single upper bound on beam width: it sizes the
    # worker's persistent beam KV pool AND the captured CUDA graph shape
    # (num_tokens = width). None means beam decode is disabled — no pool
    # allocated, no graph captured. When beam_graph_enabled is True and
    # beam_max_width is set, a FULL graph is captured at startup; requests
    # with beam_width <= beam_max_width are served (graph replay for the
    # exact-width case, eager otherwise).
    #
    # beam_max_width is the ROW count only. The per-row candidate count K is
    # ``gr.candidate_width`` and is independent of it: K follows
    # beam_max_width when unset, and a beam wider than MAX_CANDIDATE_WIDTH must
    # state K explicitly (see resolve_candidate_width).
    beam_graph_enabled: bool = field(
        default=True,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_graph_enabled",
                canonical_path=("gr", "beam_graph_enabled"),
            )
        },
    )
    """Enable Beam graph capture when a Beam pool is configured."""
    beam_max_width: int | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_max_width", canonical_path=("gr", "beam_max_width")
            )
        },
    )
    """Maximum Beam row count; None disables the persistent Beam pool."""
    beam_max_decode_steps: int = field(
        default=3,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_max_decode_steps",
                canonical_path=("gr", "max_decode_steps"),
            )
        },
    )
    """Maximum Beam decode steps, excluding the SID begin and end tokens."""
    # None = unset: the runtime picks per-platform defaults (xllm on NPU when
    # the operator library is present, cascade otherwise and on CUDA).
    npu_decode_execution: NPUDecodeExecution | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_npu_decode_execution",
                canonical_path=("gr", "npu_decode_execution"),
            )
        },
    )
    """NPU decode mode; None leaves platform selection to the runtime."""
    # None = unset (the Driver applies platform defaults).  True moves the beam
    # decision (beam_search_group / beam_search_rec_final_select +
    # select_unshared_kv) into the V1 stage runner, which owns the result.
    # CUDA beam deployments default to True; NPU requires xllm execution.
    # Explicit False is preserved; it does not enable a legacy frontend path.
    beam_worker_decision: bool | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_worker_decision",
                canonical_path=("gr", "worker_decision"),
            )
        },
    )
    """Worker decision override; None defers selection and False opts out."""
    # Exact logical B captured by the NPU Beam graph adapter. This is a graph
    # signature dimension, not a scheduler capacity: other B values do not use
    # the Beam FULL graph (native PIECEWISE/NONE dispatch remains available).
    beam_graph_batch_size: int = field(
        default=1,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_graph_batch_size",
                canonical_path=("gr", "decode_graph_batch_size"),
            )
        },
    )
    """Exact logical batch size captured by the Beam graph adapter."""

    # Like beam_graph_enabled, this flag is dormant without a Beam pool.
    enable_short_suffix_attention: bool = field(
        default=True,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_enable_short_suffix_attention",
                canonical_path=("gr", "enable_short_suffix_attention"),
            )
        },
    )
    """Fuse CUDA suffix attention and prefix merging by default; False opts out."""

    prefill_flash_attn_version: int | str | None = field(
        default="auto",
        metadata={
            "gr": ConfigFieldSpec(
                "validate_prefill_flash_attn_version",
                canonical_path=("gr", "prefill_flash_attn_version"),
            )
        },
    )
    """CUDA prefill only: auto (default) prefers FA3; None inherits decode; or explicit 2/3."""

    beam_graph_widths: tuple[int, ...] | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_graph_widths",
                canonical_path=("gr", "decode_graph_widths"),
            )
        },
    )
    """Additional exact NPU Decode input widths to capture at startup."""

    beam_sid_stage_vocab: bool = field(
        default=False,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_sid_stage_vocab",
                canonical_path=("gr", "sid_stage_vocab"),
            )
        },
    )
    """Opt into catalog-free NPU SA/SB/SC domains and stage-local probabilities."""

    beam_adapter_cache_enabled: bool = field(
        default=False,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_beam_adapter_cache_enabled",
                canonical_path=("gr", "adapter_cache_enabled"),
            )
        },
    )
    """Opt into CUDA stage-adapter and Decode graph reuse across batch sizes."""

    constraint: ConstraintConfig | None = field(
        default=None,
        metadata={"gr": ConfigFieldSpec("validate_constraint_config")},
    )
    """Constraint mode and optional table descriptor."""
    beam_candidate_width: int | None = field(
        default=None,
        metadata={
            "gr": ConfigFieldSpec(
                "validate_candidate_width", canonical_path=("gr", "beam_candidate_width")
            )
        },
    )
    """Engine-wide candidate width per parent beam."""

    deployment: DeploymentOptions = field(
        default=DEFAULT_DEPLOYMENT,
        metadata={"gr": ConfigFieldSpec("validate_deployment")},
    )
    """Resolved startup settings shared by the driver and workers."""

    def __post_init__(self) -> None:
        from vllm_gr.config.validation import validate_config_fields

        validate_config_fields(self)
        self._resolve_beam_widths()
        if self.beam_graph_widths is not None and (
            self.beam_max_width is None or max(self.beam_graph_widths) > self.beam_max_width
        ):
            raise ValueError("beam.graph_widths requires max_width >= every captured width")
        if self.beam_graph_active:
            capture_widths = set(self.beam_graph_widths or ()) | {self.beam_max_width}
            if len(capture_widths) > MAX_NPU_BEAM_GRAPH_WIDTHS:
                raise ValueError(
                    "beam.graph_widths allows at most "
                    f"{MAX_NPU_BEAM_GRAPH_WIDTHS} captured widths including max_width, "
                    f"got {len(capture_widths)}; use fewer widths or disable beam graphs"
                )
        constraint = self.constraint
        unconstrained = constraint is not None and constraint.mode is ConstraintMode.NONE
        if self.beam_sid_stage_vocab and (
            not unconstrained
            or self.beam_worker_decision is not True
            or self.beam_max_decode_steps > CONSTRAINT_TABLE_MAX_DECODE_STEPS
        ):
            raise ValueError(
                "gr.sid_stage_vocab requires gr.constraint_mode=none, "
                "gr.worker_decision=True, and "
                "gr.max_decode_steps <= 3"
            )

    def _resolve_beam_widths(self) -> None:
        widths = self.deployment.beam_width
        if widths is None:
            result_width = self.deployment.result_width
            if (
                result_width is not None
                and self.beam_max_width is not None
                and result_width > self.beam_max_width
            ):
                raise ValueError("gr.result_width must not exceed beam_max_width")
            return
        candidates = self.deployment.candidate_width
        assert candidates is not None  # Validated together by DeploymentOptions.
        if len(widths) > self.beam_max_decode_steps:
            raise ValueError(
                "gr.max_decode_steps must be at least the number of entries in gr.beam_width"
            )
        if (
            self.constraint is not None
            and self.constraint.mode is ConstraintMode.TABLE
            and len(widths) != CONSTRAINT_TABLE_MAX_DECODE_STEPS
        ):
            raise ValueError("constraint tables require exactly three Beam steps")
        max_width = max(widths)
        # Matching dynamic schedules select candidates per step; a scalar K
        # would instead override request defaults during admission.
        candidate_capacity = (
            None if widths == candidates and len(set(widths)) > 1 else max(candidates)
        )
        if self.beam_max_width is not None and self.beam_max_width != max_width:
            raise ValueError("beam_max_width conflicts with gr.beam_width")
        if (
            self.beam_candidate_width is not None
            and self.beam_candidate_width != candidate_capacity
        ):
            raise ValueError("beam_candidate_width conflicts with gr.candidate_width")
        object.__setattr__(self, "beam_max_width", max_width)
        object.__setattr__(self, "beam_candidate_width", candidate_capacity)

    @property
    def beam_graph_active(self) -> bool:
        """True iff the startup beam-graph capture should run."""
        return self.beam_graph_enabled and self.beam_max_width is not None

    @property
    def enabled(self) -> bool:
        return (
            self.constraint is not None
            or self.attention_backend is not None
            or self.npu_decode_execution is not None
            or self.beam_worker_decision is not None
            or self.beam_sid_stage_vocab
        )

    @property
    def constraint_config(self) -> ConstraintConfig:
        return (
            self.constraint
            if self.constraint is not None
            else ConstraintConfig(ConstraintMode.NONE)
        )

    @classmethod
    def from_vllm_config(cls, vllm_config: object | None) -> GRConfig:
        """Parse the GR config from a ``VllmConfig``'s additional_config.

        The parse is cached on the ``vllm_config`` object itself
        (``_vllm_gr_parsed_config``): the same engine hands the SAME
        ``vllm_config`` to this classmethod on every request/frame path
        (``_engine_constraint_table_active``, ``_beam_graph_enabled_for_run``,
        beam metadata construction, ...), and re-running the full
        validator-suite + ``__post_init__`` each time costs ~0.13ms per
        request.  ``write_to_vllm_config`` replaces the cached snapshot so
        write→read round trips observe the new state.  Hosts that reject
        attribute writes (frozen mocks) silently fall back to per-call
        parsing.
        """
        if vllm_config is None:
            return GRConfig()
        cached = getattr(vllm_config, "_vllm_gr_parsed_config", None)
        if cached is not None:
            return cached
        from vllm_gr.config.parser import parse_gr_config

        parsed = parse_gr_config(vllm_config)
        try:
            object.__setattr__(vllm_config, "_vllm_gr_parsed_config", parsed)
        except (AttributeError, TypeError):
            pass
        return parsed

    def write_to_vllm_config(self, vllm_config: object) -> None:
        from vllm_gr.config.parser import write_canonical_gr_config

        write_canonical_gr_config(vllm_config, self)


def runtime_constraint_table(config: GRConfig) -> ConstraintTableSpec | None:
    """Return the configured artifact, if this engine has a table."""
    return config.constraint.table if config.constraint is not None else None


def configured_constraint_mode(config: GRConfig) -> ConstraintMode | None:
    """Preserve the distinction between omitted constraints and explicit NONE."""
    return config.constraint.mode if config.constraint is not None else None


def resolve_candidate_width(config: GRConfig) -> int:
    """Resolve the effective per-row candidate width K.

    The single derivation point: ``gr.candidate_width`` wins when
    set, otherwise K follows ``beam_max_width`` (the pre-decoupling behaviour).
    Every workspace bucket key, admission gate, and kernel launch must call
    this instead of spelling out ``(width, width, dtype)``.
    """

    configured = config.beam_candidate_width
    if configured is not None:
        return configured
    if config.beam_max_width is None:
        raise ValueError("vllm-gr cannot resolve candidate_width without beam_max_width")
    candidate_width = int(config.beam_max_width)
    if candidate_width > MAX_CANDIDATE_WIDTH:
        # Following the beam width is only valid while it fits the operator
        # ceiling; a wider beam must state its per-row candidate width.
        raise ValueError(
            f"vllm-gr beam_max_width={candidate_width} exceeds the candidate width "
            f"ceiling {MAX_CANDIDATE_WIDTH}; K follows the Beam width when "
            "gr.candidate_width is unset. Set "
            "gr.candidate_width to the per-row candidate count, or "
            f"keep beam_max_width <= {MAX_CANDIDATE_WIDTH}."
        )
    return candidate_width


def resolve_constraint_backend(config: GRConfig, device_type: str) -> GRConfig:
    """Bind AUTO to the driver device before publishing the worker payload."""
    constraint = config.constraint
    if (
        constraint is None
        or constraint.table is None
        or constraint.table.backend is not ConstraintBackendKind.AUTO
    ):
        return config
    backend = ConstraintBackendKind("cuda" if device_type == "musa" else device_type)
    if backend is ConstraintBackendKind.AUTO:
        raise ValueError("auto requires a concrete platform device type")
    return replace(
        config, constraint=replace(constraint, table=replace(constraint.table, backend=backend))
    )


def deployment_for(vllm_config: object | None) -> DeploymentOptions:
    if vllm_config is None:
        return DEFAULT_DEPLOYMENT
    return GRConfig.from_vllm_config(vllm_config).deployment
