# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm
"""Atomic GR session admission and EngineCore socket transport."""

from __future__ import annotations

from copy import copy
from typing import Any

from vllm.logger import init_logger
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequestType,
    FinishReason,
)

from vllm_gr.v1.beam.batch_contracts import GRAdmissionError
from vllm_gr.v1.engine.core_socket import process_input_sockets as process_input_sockets

logger = init_logger(__name__)


def loaded_constraint_trie_count(worker: Any) -> int:
    """Read the validated, live table once through the executor startup RPC."""
    backend = getattr(worker.model_runner, "constraint_backend", None)
    return backend.info.table_metadata.trie_count if backend is not None else 1


def is_gr_session_retired(self: Any, session_id: str) -> bool:
    """Ordered Core-thread query; never block the thread that drains BatchQueue."""
    return session_id not in getattr(self.scheduler, "_gr_v1_sessions", {})


def send_gr_rejection(engine: Any, engine_request: Any, message: str) -> None:
    """Answer a refused GR session with the terminal payload it never produced.

    Upstream's error output carries no ``gr_batch_result``, and a frontend can
    only read that absence as a broken engine. Sending the failure the session
    would have reported keeps the reason, and its status class, intact.
    """
    from vllm_gr.v1.beam.batch_contracts import admission_rejected

    request_id = engine_request.request_id
    output = EngineCoreOutput(request_id, [], finish_reason=FinishReason.ERROR)
    output.kv_transfer_params = {"gr_batch_result": admission_rejected(request_id, message)}
    engine.output_queue.put_nowait(
        (
            engine_request.client_index,
            EngineCoreOutputs(outputs=[output], finished_requests=[request_id]),
        )
    )


def _handle_submit_gr_session(self, batch_request) -> None:
    """Keep a caller-formed batch atomic across the input-thread boundary."""
    if not batch_request.items:
        logger.error("Rejecting an empty beam_search batch")
        return
    try:
        from vllm_gr.v1.engine.beam_admission import validate_gr_request

        validate_gr_request(
            batch_request,
            config=getattr(self, "vllm_config", None),
            trie_count=getattr(self, "_gr_trie_count", None),
        )
    except ValueError as error:
        # A malformed batch is a request rejection, not a preprocessing fault.
        if getattr(self, "output_queue", None) is None:
            raise GRAdmissionError(str(error)) from error
        send_gr_rejection(self, batch_request.items[0].request, str(error))
        return
    except Exception:
        handler = getattr(self, "_handle_request_preproc_error", None)
        if handler is None:
            raise
        handler(batch_request.items[0].request)
        return
    if hasattr(self, "input_queue"):
        self.input_queue.put_nowait((EngineCoreRequestType.SUBMIT_GR_SESSION, batch_request))
        return
    _admit_gr_session(self, batch_request)


def _admit_gr_session(self, batch_request) -> None:
    """Preprocess and admit one complete GR session on the core thread."""
    if not batch_request.items:
        logger.error("Rejecting an empty beam_search batch")
        return
    engine_requests = tuple(item.request for item in batch_request.items)
    try:
        from vllm_gr.config.execution import validate_gr_execution_config
        from vllm_gr.platforms.errors import GRDeviceRequestError
        from vllm_gr.v1.engine.beam_admission import (
            validate_gr_device_config,
            validate_gr_request,
        )

        validate_gr_request(
            batch_request,
            config=getattr(self, "vllm_config", None),
            trie_count=getattr(self, "_gr_trie_count", None),
        )
    except ValueError as error:
        # The request is answerable, so it must not take the preprocessing
        # error path: that one logs a traceback and sends a bare
        # FinishReason.ERROR, which the frontend can only read as a fault.
        # In-process callers have no output socket, so the rejection travels
        # to them as the exception the multiprocess handler encodes.
        if getattr(self, "output_queue", None) is None:
            raise GRAdmissionError(str(error)) from error
        send_gr_rejection(self, engine_requests[0], str(error))
        return
    try:
        # Deployment configuration stays on the preprocessing error path: an
        # engine started with the wrong settings is a server fault, not
        # something the caller can fix by resending.
        validate_gr_execution_config(self.vllm_config)
        validate_gr_device_config(self.vllm_config, batch_request)
        from vllm_gr.config.backend_plan import read_backend_plan

        plan = read_backend_plan(self.vllm_config)
        serial_musa = plan is not None and plan.backend_id == "musa"
        if not serial_musa and (self.batch_queue is None or self.batch_queue_size < 2):
            raise RuntimeError("beam_search requires a non-blocking EngineCore batch queue")
        preprocessed_batch = []
        for engine_request in engine_requests:
            owned_request = copy(engine_request)
            # Native Request construction only reads nested sampling fields. GR
            # replaces stop_token_ids below; it never mutates the original list.
            # The Worker separately detaches every mutable field it retains.
            owned_request.sampling_params = copy(engine_request.sampling_params)
            preprocessed_batch.append(self.preprocess_add_request(owned_request))
    except GRDeviceRequestError as error:
        # Device-specific request limits have the same answerable rejection
        # contract as shared validation. Deployment faults still use the
        # preprocessing handler below.
        if getattr(self, "output_queue", None) is None:
            raise GRAdmissionError(str(error)) from error
        send_gr_rejection(self, engine_requests[0], str(error))
        return
    except Exception:
        handler = getattr(self, "_handle_request_preproc_error", None)
        if handler is None:
            raise  # InprocClient has no output-socket error handler.
        # The batch has one external result channel, owned by the first item.
        handler(engine_requests[0])
        return
    for member_index, preprocessed in enumerate(preprocessed_batch):
        native_request = preprocessed[0]
        # GR owns stop-token/EOS handling. Native tokens are placeholders for
        # the device-owned batch state and cannot terminate members alone.
        native_request.sampling_params.ignore_eos = True
        native_request.sampling_params.stop_token_ids = []
        native_request.gr_batch_request = batch_request
        native_request.gr_batch_member_index = member_index
        # SUBMIT_GR_SESSION reaches this handler as one busy-loop message, so
        # native scheduling cannot observe a partially inserted batch.
        self.add_request(*preprocessed)


# ---------------------------------------------------------------------------
# Narrow GR input dispatch (socket adapter + busy-loop fallback)
# ---------------------------------------------------------------------------


def dispatch_gr_engine_core_input(self, request_type, payload) -> bool:
    """Handle vLLM-GR EngineCore request types.

    Returns True when the type is owned by GR and has been consumed, so
    callers skip upstream ADD/ABORT/generic enqueue.  Used by the socket
    adapter and by the ``_handle_client_request`` wrap so GR messages are
    never logged as unrecognized.
    """
    from vllm.v1.engine import EngineCoreRequestType

    if request_type == EngineCoreRequestType.SUBMIT_GR_SESSION:
        _admit_gr_session(self, payload)
        return True
    return False
