# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""EngineCore process startup and GR admission integration."""

from __future__ import annotations

import time
from functools import wraps

from vllm.logger import init_logger

from vllm_gr.v1.core.kv_cache_utils import apply_kv_hash_memo_patch
from vllm_gr.v1.core.sched.scheduler_patch import apply_scheduler_patch
from vllm_gr.v1.engine.wire import register_engine_core_request_types

logger = init_logger(__name__)

_ENGINE_STEP_PATCH_MARKER = "_vllm_gr_progress_step"
_CLIENT_REQUEST_PATCH_MARKER = "_vllm_gr_gr_input_dispatch"
_PIS_PATCH_MARKER = "_vllm_gr_process_input_sockets"

# Upstream EngineCoreProc.process_input_sockets captured before the adapter
# is installed.  Parity tests inspect this object against the pinned vLLM.
UPSTREAM_PROCESS_INPUT_SOCKETS = None


def make_patched_engine_init(original):
    """Publish the loaded table's immutable capacity before accepting requests."""
    if getattr(original, "_gr_trie_capacity_patch", False):
        return original

    @wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        from vllm_gr.config import GRConfig
        from vllm_gr.config.backend_plan import read_backend_plan
        from vllm_gr.v1.engine.core import loaded_constraint_trie_count
        from vllm_gr.v1.engine.worker_readiness import verify_engine_backend_readiness

        verify_engine_backend_readiness(self)

        config = GRConfig.from_vllm_config(self.vllm_config)
        plan = read_backend_plan(self.vllm_config)
        if plan is not None and plan.backend_id == "musa" and config.beam_max_width:
            # Eager Beam has one pending execute state. Consume it before
            # vLLM can enqueue the next stage on the local worker.
            self.step_fn = self.step
            self.batch_queue = None
            self.batch_queue_size = 1
        if config.beam_max_width is None:
            return
        counts = self.model_executor.collective_rpc(loaded_constraint_trie_count)
        if not counts or any(type(n) is not int or n < 1 or n != counts[0] for n in counts):
            raise RuntimeError("Workers disagree on the loaded constraint table Trie count")
        self._gr_trie_count = self.scheduler._gr_trie_count = counts[0]

    setattr(initialize, "_gr_trie_capacity_patch", True)
    return initialize


def make_patched_process_engine_step(original_step):
    """Preserve V1 progress-aware backoff while forwarding native outputs."""

    if getattr(original_step, _ENGINE_STEP_PATCH_MARKER, False):
        return original_step

    @wraps(original_step)
    def patched_process_engine_step(self):
        progress_before = getattr(self.scheduler, "_gr_progress", 0)
        outputs, model_executed = self.step_fn()
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        self.post_step(model_executed)

        if (
            not model_executed
            and self.scheduler.has_requests()
            and getattr(self.scheduler, "_gr_progress", 0) == progress_before
            and not getattr(self, "batch_queue", None)
        ):
            time.sleep(0.001)

        return model_executed

    setattr(patched_process_engine_step, _ENGINE_STEP_PATCH_MARKER, True)
    return patched_process_engine_step


def make_patched_handle_client_request(original):
    """Dispatch GR wire types and isolate expected ADD admission rejections.

    The socket adapter queues each complete GR batch without admitting it.
    This wrapper admits the batch on the core thread, where Scheduler
    failures can be returned to the requesting client. Native ADD keeps
    upstream shutdown and DP handling.
    """
    if getattr(original, _CLIENT_REQUEST_PATCH_MARKER, False):
        return original

    @wraps(original)
    def patched_handle_client_request(self, request_type, request, *args, **kwargs):
        from vllm.v1.engine import EngineCoreRequestType

        from vllm_gr.v1.beam.batch_contracts import GRAdmissionError
        from vllm_gr.v1.engine.core import (
            dispatch_gr_engine_core_input,
            send_gr_rejection,
        )

        if (
            request_type == EngineCoreRequestType.SUBMIT_GR_SESSION
            and request.items
            and self._reject_add_in_shutdown(request.items[0].request)
        ):
            return None

        try:
            if dispatch_gr_engine_core_input(self, request_type, request):
                return None
        except GRAdmissionError as error:
            if request_type != EngineCoreRequestType.SUBMIT_GR_SESSION:
                raise
            native_request = request.items[0].request
            logger.warning(
                "Rejecting GR batch %s during admission: %s",
                request.batch_id,
                error,
            )
            send_gr_rejection(self, native_request, str(error))
            return None
        if request_type == EngineCoreRequestType.ADD:
            # Admission runs here, after the socket thread has queued ADD.
            # Preserve upstream shutdown/DP handling and catch only explicit
            # GR rejections. Internal failures still reach the fatal handler.
            try:
                return original(self, request_type, request, *args, **kwargs)
            except GRAdmissionError as error:
                native_request, _request_wave = request
                logger.warning(
                    "Rejecting request %s during GR admission: %s",
                    native_request.request_id,
                    error,
                )
                # A bare error output is indistinguishable from an engine
                # fault at the frontend, which then reports the rejection as
                # a server error. Carry the reason with the terminal status.
                send_gr_rejection(self, native_request, str(error))
                return None
        return original(self, request_type, request, *args, **kwargs)

    setattr(patched_handle_client_request, _CLIENT_REQUEST_PATCH_MARKER, True)
    return patched_handle_client_request


# ---------------------------------------------------------------------------
# Child-process patch applicator — called from patched run_engine_core
# ---------------------------------------------------------------------------


def apply_engine_core_request_patches() -> None:
    """Install V1 session admission and native request dispatch."""
    from vllm.v1.engine.core import DPEngineCoreProc, EngineCore, EngineCoreProc

    from vllm_gr.v1.engine.hot_update import install_core_hot_update_hooks

    install_core_hot_update_hooks(EngineCore)

    from vllm_gr.v1.engine.core import (
        _handle_submit_gr_session,
        is_gr_session_retired,
    )
    from vllm_gr.v1.engine.core_socket import process_input_sockets

    logger.info("[vllm-gr] apply_engine_core_child_patches called.")

    register_engine_core_request_types()
    from vllm_gr.v1.token_buffer import (
        install_numpy_request_factory,
        install_numpy_scheduler_data,
    )

    install_numpy_request_factory()
    install_numpy_scheduler_data()

    EngineCore.__init__ = make_patched_engine_init(EngineCore.__init__)

    EngineCore.is_gr_session_retired = is_gr_session_retired
    EngineCore._handle_submit_gr_session = _handle_submit_gr_session
    EngineCoreProc._handle_submit_gr_session = _handle_submit_gr_session

    # Narrow GR dispatch on the busy-loop request handler so in-process
    # EngineCore and any GR type that misses the socket adapter still land.
    EngineCoreProc._handle_client_request = make_patched_handle_client_request(
        EngineCoreProc._handle_client_request
    )
    DPEngineCoreProc._handle_client_request = make_patched_handle_client_request(
        DPEngineCoreProc._handle_client_request
    )

    # vLLM-version adapter: capture the upstream loop once, then install ours.
    global UPSTREAM_PROCESS_INPUT_SOCKETS
    current_pis = EngineCoreProc.process_input_sockets
    if UPSTREAM_PROCESS_INPUT_SOCKETS is None and current_pis is not process_input_sockets:
        UPSTREAM_PROCESS_INPUT_SOCKETS = current_pis
    EngineCoreProc.process_input_sockets = process_input_sockets
    setattr(EngineCoreProc.process_input_sockets, _PIS_PATCH_MARKER, True)

    # Preserve V1 progress-aware backoff without legacy step chaining.
    EngineCoreProc._process_engine_step = make_patched_process_engine_step(
        EngineCoreProc._process_engine_step
    )

    EngineCore._patched_for_beam_requests = True
    EngineCoreProc._patched_for_beam_requests = True
    logger.debug("Engine-core request patches applied.")


def apply_engine_core_child_patches() -> None:
    """Compatibility helper that installs all EngineCore-side capabilities.

    This is called by the process-entry wrapper in a freshly spawned
    interpreter. Each capability has its own idempotent applicator so a
    partially initialized process can be checked and repaired explicitly.
    """
    from vllm_gr.v1.worker.model_runner_patch import apply_worker_patches

    apply_engine_core_request_patches()
    apply_kv_hash_memo_patch()
    apply_scheduler_patch()
    apply_worker_patches()


# Re-entry guard to detect recursive calls
_run_engine_core_active = False


def patch_engine_core_process_entrypoint() -> None:
    """Wrap the EngineCore process entrypoint for spawn-based runtimes."""
    from vllm.v1.engine.core import EngineCoreProc

    # Save the original so the wrapper can delegate without recursion
    # even if run_patch() is called again in a spawned child process.
    current_run_engine_core = EngineCoreProc.run_engine_core
    if current_run_engine_core is run_engine_core:
        return
    original_run_engine_core = getattr(
        run_engine_core,
        "_original_run_engine_core",
        current_run_engine_core,
    )
    if original_run_engine_core is run_engine_core:
        original_run_engine_core = current_run_engine_core
    setattr(run_engine_core, "_original_run_engine_core", original_run_engine_core)
    EngineCoreProc.run_engine_core = run_engine_core


def run_engine_core(
    *args,
    dp_rank: int = 0,
    local_dp_rank: int = 0,
    **kwargs,
):
    """Wrapper for EngineCoreProc.run_engine_core that applies patches
    before the engine starts.  This is a module-level function so that
    multiprocessing 'spawn' can pickle it by qualified name.

    In the spawned child process, all modules are freshly imported, so
    EngineCoreProc.run_engine_core is the original unpatched version.
    We apply our patches first, then delegate to the original.
    """
    global _run_engine_core_active

    # Re-entry guard: detect if we're already inside run_engine_core
    if _run_engine_core_active:
        raise RuntimeError(
            "run_engine_core re-entry detected. Check that "
            "engine_core_patch.patch_engine_core_process_entrypoint() saved the original "
            "EngineCoreProc.run_engine_core before installing the wrapper."
        )

    from vllm.v1.engine.core import EngineCoreProc

    # Guard against recursion: if vllm_gr/init.py was auto-imported in
    # the child process, run_patch() will have replaced
    # EngineCoreProc.run_engine_core with *this* function again.
    # Use the original saved by patch_engine_core_process_entrypoint() if
    # available; otherwise fall back to the current class attribute
    # (which is the unpatched original in a fresh child interpreter).
    original_run = getattr(run_engine_core, "_original_run_engine_core", None)

    # Validate the saved original
    if original_run is not None and original_run is not run_engine_core:
        # We have a saved original that isn't ourselves - validate it
        if not callable(original_run):
            raise RuntimeError(
                f"run_engine_core._original_run_engine_core is not callable: "
                f"type={type(original_run)}"
            )
    else:
        # No saved original (or it points to ourselves, which is invalid)
        # Fall back to the current class attribute (should be the original in a fresh process)
        if original_run is run_engine_core:
            logger.warning(
                "run_engine_core._original_run_engine_core points to itself. "
                "This may indicate corrupted patch state from a previous run. "
                "Falling back to EngineCoreProc.run_engine_core."
            )
        original_run = EngineCoreProc.run_engine_core

    # Final validation: ensure we didn't end up with ourselves
    if original_run is run_engine_core:
        raise RuntimeError(
            "run_engine_core recursion detected: EngineCoreProc.run_engine_core "
            "points to this wrapper. This indicates that patching was applied "
            "but the original was not saved, or the child process imported "
            "vllm_gr before starting."
        )

    # Validate that original_run looks like a real function from vllm
    # (Skip validation for test mocks)
    if hasattr(original_run, "__module__"):
        original_module = original_run.__module__
        expected_module = "vllm.v1.engine.core"
        # Skip warning for unittest mocks (used in tests)
        if original_module not in (expected_module, "unittest.mock", "mock"):
            logger.warning(
                "Unexpected module for original run_engine_core: %s (expected %s). "
                "Proceeding anyway, but this may indicate an issue.",
                original_module,
                expected_module,
            )

    # Set re-entry guard and apply patches
    _run_engine_core_active = True
    try:
        # Use the shared initialization mechanism with an EngineCore-only
        # plan. In a spawned child this repairs targets that became importable
        # only after the frontend initialized without importing API modules.
        from vllm_gr.patch import initialize_engine_core_beam_patches

        initialize_engine_core_beam_patches()
        original_run(
            *args,
            dp_rank=dp_rank,
            local_dp_rank=local_dp_rank,
            **kwargs,
        )
    finally:
        _run_engine_core_active = False


def validate_engine_core_entrypoint_patch() -> str | None:
    from vllm.v1.engine.core import EngineCoreProc

    if EngineCoreProc.run_engine_core is not run_engine_core:
        return "EngineCore process entrypoint wrapper is not installed"
    return None


def validate_engine_core_request_patch() -> str | None:
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    from vllm_gr.v1.engine.core import (
        _handle_submit_gr_session,
        is_gr_session_retired,
    )
    from vllm_gr.v1.engine.core_socket import process_input_sockets

    valid = (
        getattr(EngineCore, "_handle_submit_gr_session", None) is _handle_submit_gr_session
        and getattr(EngineCore.__init__, "_gr_trie_capacity_patch", False)
        and getattr(EngineCoreProc, "_handle_submit_gr_session", None) is _handle_submit_gr_session
        and getattr(EngineCore, "is_gr_session_retired", None) is is_gr_session_retired
        and getattr(EngineCoreProc._process_engine_step, _ENGINE_STEP_PATCH_MARKER, False)
        and getattr(EngineCoreProc._handle_client_request, _CLIENT_REQUEST_PATCH_MARKER, False)
        and EngineCoreProc.process_input_sockets is process_input_sockets
        and getattr(EngineCoreProc.process_input_sockets, _PIS_PATCH_MARKER, False)
    )
    return None if valid else "EngineCore GR session dispatch is not fully installed"


def validate_engine_core_request_group() -> str | None:
    from vllm_gr.v1.engine.wire import validate_request_types_patch

    failures = tuple(
        detail
        for detail in (
            validate_request_types_patch(),
            validate_engine_core_request_patch(),
        )
        if detail is not None
    )
    return "; ".join(failures) or None
