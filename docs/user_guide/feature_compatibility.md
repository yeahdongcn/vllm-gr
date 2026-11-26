# Feature compatibility

To run Beam search, use the [offline quickstart](../getting_started/quickstart.md)
or send an [HTTP Beam request](../serving/completions.md#request). Use the
matrix to choose settings for your device.

The first release includes CUDA and Ascend inference and experimental gRPC.
CPU packages support packaging and host checks only.
The matrix describes the documented runtime paths and their configuration limits.

## Support matrix

| Feature | CUDA | Ascend NPU | Details |
| --- | --- | --- | --- |
| vLLM ModelRunner | Library and general CLI default to V1. The direct offline Beam, HTTP and OneRec gRPC examples select V2 by default. | ModelRunner V1 required for Beam search | [ModelRunner selection](../configuration/README.md#modelrunner-selection), [example defaults](../../examples/README.md#choose-a-modelrunner) |
| Catalog-constrained Beam (constraint table) | Supported | Supported | [Configure a constraint table](../configuration/README.md#constraint-table) |
| Full-vocabulary Beam without a catalog | Supported | Supported, with execution boundaries | [Beam Search on NPU](beam_search_npu.md) |
| Dynamic Beam widths | Supported | Supported with or without a table. In full-vocabulary mode, candidate and retained widths must match at each step. | [Beam settings](../configuration/README.md#beam-settings), [Ascend execution limits](beam_search_npu.md#supported-execution-boundaries) |
| Independent per-parent candidate widths | Supported | Supported with an NPU constraint table, within its `max_top_k`. | [Beam settings](../configuration/README.md#beam-settings) |
| Beam Decode graphs | Supported (`gr.beam_graph_enabled`) | FULL replay requires fused xllm Decode and an eligible batch matching a captured batch size and Decode input width. | [CUDA ModelRunner V2](../design/module/model_runner_v2.md), [Ascend capture and fallback](../design/feature/npu_decode_execution.md#decode-graph-capture) |
| Prefill graphs | Supported; explicit capture configuration | Opt-in BSND Prefill graphs | [Prefill capture](../design/feature/cuda_prefill_graph.md), [BSND Prefill](../design/feature/npu_prefill_bsnd.md) |
| Asynchronous scheduling | Supported | Supported | [Configuration](../configuration/README.md#asynchronous-scheduling), [CUDA walkthrough](onerec_quickstart.md), [Ascend example](beam_search_npu.md#configuration-and-python-entry-point) |
| Multi-Trie selection per request | Supported | Supported | [Constraint tables](../configuration/constraint_tables.md) |
| gRPC transport | Experimental, requires the `grpc` extra | Ascend-specific validation remains to be documented | [gRPC token inputs and CUDA example](../serving/grpc.md) |
| Checkpoint updates on running instances | Supported with a drained instance | Supported with a drained instance | [Model and catalog updates](weight_update.md) |
| Online catalog replacement | Supported on the asynchronous Beam path | Compile and load the new table at startup | [Model and catalog updates](weight_update.md) |
| NPU decode execution modes (`xllm`, `cascade`) | Not applicable | Supported | [Decode execution](../design/feature/npu_decode_execution.md) |
| SID stage projection for catalog-free generation | Not applicable | Supported | [Beam Search on NPU](beam_search_npu.md) |
| Caller-owned XAttention output binding | Not applicable | Optional, separate build | [XAttention output](../design/feature/npu_decode_execution.md#caller-owned-xattention-output) |

## Moore Threads MUSA

The [OneRec on MUSA quickstart](musa_quickstart.md) documents the validated
single-S5000 configuration with vLLM-MUSA 0.32.0.

| Feature | Validated configuration |
| --- | --- |
| ModelRunner and attention | V1, one S5000, CUSTOM attention, asynchronous scheduling |
| Full-vocabulary Beam | OneRec-1.7B offline and HTTP, width 2, eager |
| Catalog-constrained Beam | OneRec-1.7B, width 2, three SID components, eager. Two concurrent HTTP clients used a single-request execution bucket. |
| GR decode and prefill graphs | OneRec-1.7B offline, full-vocabulary Beam, width 2 and execution bucket 1 |
| ModelRunner V2, multi-GPU, gRPC, online catalog replacement | Validation pending |

## How to read this matrix

- "Supported" describes a documented execution path. The linked guides record
  the validation scope, including model and graph configurations where applicable.
- A feature can be supported on a device and still be excluded for a specific
  model geometry or execution mode. Check the linked page before relying on a
  combination that is not listed here.
- Startup and request validation reject unsupported configurations. A valid
  request can still fall back to native dispatch when it misses a graph capture.

<a id="related"></a>

## See also

- [Platform backend contract](../design/module/platform_backend.md) for how a device
  provider is registered and activated.
- [Supported models](../models/supported_models.md) for the model list.
