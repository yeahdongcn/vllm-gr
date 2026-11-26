# MUSA provider bootstrap

Use a prepared vLLM-MUSA 0.32.0 environment with torchada and the MUSA
runtime installed. From the repository root, install GR:

```bash
VLLM_GR_TARGET_DEVICE=musa python -m pip install .
```

Verify provider activation and GR runtime patch installation:

```bash
python -c 'import torchada; from vllm_gr.plugin import initialize_runtime; print(initialize_runtime())'
```

This provider currently implements runtime initialization. GR Beam requests
are rejected at admission until the MUSA runner and attention path are ready.
