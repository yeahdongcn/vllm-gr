# OneRec on Moore Threads MUSA

The initial configuration uses one S5000, ModelRunner V1, eager execution,
asynchronous scheduling, and CUSTOM attention. The validated runtime is
vLLM 0.32.0 with MUSA 5.2.0 and PyTorch 2.13.0.post1. Install GR into the
prepared vLLM-MUSA environment with:

```bash
VLLM_GR_TARGET_DEVICE=musa python -m pip install .
```

Set `ONEREC_MODEL` to the local directory of
[OpenOneRec/OneRec-1.7B](https://huggingface.co/OpenOneRec/OneRec-1.7B/tree/cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b).
Launch the server from this environment:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0
python -c 'import torchada; from vllm_gr.entrypoints.cli.main import dispatch_main; dispatch_main()' serve "$ONEREC_MODEL" --gr \
  --host 127.0.0.1 --port 18094 --served-model-name onerec --trust-remote-code \
  --enforce-eager --async-scheduling --enable-prefix-caching --no-enable-chunked-prefill \
  --attention-config '{"backend":"CUSTOM"}' --max-model-len 128 --max-num-batched-tokens 128 \
  --max-num-seqs 2 --max-logprobs 4 --gpu-memory-utilization 0.3 \
  --gr.beam_width '[2,2,2,2]' --gr.candidate_width '[2,2,2,2]' --gr.max_decode_steps 4 \
  --gr.beam_batch_sizes '[1]' --gr.constraint_mode none --gr.worker_decision true \
  --gr.beam_graph_enabled false --gr.prefill_graph_enabled false
```

After the health endpoint responds, run the same token prompt used in
the offline and HTTP smoke:

```bash
curl --fail-with-body -sS --max-time 90 http://127.0.0.1:18094/v1/completions \
  -H 'Content-Type: application/json' \
  --data '{"model":"onerec","prompt":[67644,458,1509,13],"use_beam_search":true,"n":2,"temperature":0.0,"ignore_eos":true,"max_tokens":4,"vllm_xargs":{"begin_token":"<|sid_begin|>","end_token":"<|sid_end|>"}}'
```

The response should contain two ranked choices with `finish_reason=length`.
The pinned smoke returned `<s_a_5719><s_b_324>` followed by
`<s_a_4695><s_b_5383>` inside SID boundary markers.

| Feature | S5000 validation |
| --- | --- |
| Offline full-vocabulary Beam | Width 2, two scored SID tokens, CPU FP32 reference ordering |
| HTTP full-vocabulary Beam | HTTP 200, same ordering as offline |
| Canonical catalog | Four-entry table, width 2, three SID components |
| Concurrent catalog HTTP | Two concurrent clients with a single-request Beam bucket |
| Prefill and decode graphs | Offline full-vocabulary Beam, width 2, execution bucket 1, observed prefill and decode replay |
| ModelRunner V2 and multi-GPU | Validation pending |

For table format and catalog preparation, see the [OneRec quickstart](onerec_quickstart.md).
