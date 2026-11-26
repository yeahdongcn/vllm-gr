#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
set -euo pipefail
: "${ONEREC_MODEL:?Set ONEREC_MODEL to the pinned OneRec-1.7B snapshot}"
port=${ONEREC_SMOKE_PORT:-18094}
workdir=$(mktemp -d)
url="http://127.0.0.1:$port"
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0
python -c 'import torchada; from vllm_gr.entrypoints.cli.main import dispatch_main; dispatch_main()' serve "$ONEREC_MODEL" --gr \
  --host 127.0.0.1 --port "$port" --served-model-name onerec --trust-remote-code \
  --enforce-eager --async-scheduling --enable-prefix-caching --no-enable-chunked-prefill \
  --attention-config '{"backend":"CUSTOM"}' --max-model-len 128 --max-num-batched-tokens 128 \
  --max-num-seqs 2 --max-logprobs 4 --gpu-memory-utilization 0.3 \
  --gr.beam_width '[2,2,2,2]' --gr.candidate_width '[2,2,2,2]' --gr.max_decode_steps 4 \
  --gr.beam_batch_sizes '[1]' --gr.constraint_mode none --gr.worker_decision true \
  --gr.beam_graph_enabled false --gr.prefill_graph_enabled false > "$workdir/server.log" 2>&1 &
server=$!
cleanup() {
    local status=$?
    kill "$server" 2>/dev/null || true
    wait "$server" 2>/dev/null || true
    if ((status != 0)); then
        tail -80 "$workdir/server.log"
    fi
    rm -r -- "$workdir"
}
trap cleanup EXIT
for ((i=0;i<120;i++)); do
    kill -0 "$server"
    if curl -fsS "$url/health" >/dev/null 2>&1; then break; fi
    sleep 2
done
curl -fsS "$url/health"
curl --fail-with-body -sS --max-time 90 "$url/v1/completions" \
  -H 'Content-Type: application/json' \
  --data '{"model":"onerec","prompt":[67644,458,1509,13],"use_beam_search":true,"n":2,"temperature":0.0,"ignore_eos":true,"max_tokens":4,"vllm_xargs":{"begin_token":"<|sid_begin|>","end_token":"<|sid_end|>"}}' > "$workdir/result.json"
python - "$workdir/result.json" <<'PYTHON'
import json
import sys
from pathlib import Path

response = json.loads(Path(sys.argv[1]).read_text())
expected = [
    ("<|sid_begin|><s_a_5719><s_b_324><|sid_end|>", "length"),
    ("<|sid_begin|><s_a_4695><s_b_5383><|sid_end|>", "length"),
]
actual = [(choice["text"], choice["finish_reason"]) for choice in response["choices"]]
if actual != expected:
    raise SystemExit(f"Unexpected ranked OneRec response: {actual!r}")
print("OneRec MUSA HTTP smoke passed")
PYTHON
