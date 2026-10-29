# Changelog

## 0.7.0

### New
- **vLLM inference backend** (recommended), through the `vllm_evo2` plugin: continuous batching, prefix caching and paged Hyena state, 2-30x faster than vortex on generation and scoring. Install it in its own Python 3.12 environment with `pip install "evo2[vllm]"` (see README, "Inference backends"). Supported on vLLM: `evo2_1b_base`, `evo2_7b`, `evo2_7b_base`, `evo2_7b_262k`, `evo2_7b_microviridae`, `evo2_20b`.
- `Evo2(model_name, local_path=None, use_kernels=False, backend='auto', **backend_kwargs)`. Extra keyword arguments (e.g. `fp8_mode`, `max_model_len`, `gpu_memory_utilization`, `kv_cache_dtype`, `cache_dir`) go to the vLLM plugin; `Evo2.backend` reports the engine in use.
- `generate(..., seed=...)` and other plugin-only options on vLLM.
- `Dockerfile.vllm` (untested in CI).
- Tests: `test_backend_dispatch` and `test_scoring_backends` (CPU only), `test_backend_parity dump|compare`, and `--backend` / `--local_path` on `test_evo2` and `test_evo2_generation`.

### Behaviour changes
- `backend='auto'` (the default) picks vLLM when `vllm` and `vllm_evo2` are importable, else vortex. `EVO2_BACKEND=vortex|vllm` overrides `auto`; an explicit `backend=` argument always wins. Existing vortex environments are unaffected.
- vortex is imported lazily, so `import evo2` imports neither vortex nor vLLM and works in either environment.
- `forward()` keeps its return shape on both backends: `((logits, inference_params), embeddings)` on vortex, `((logits, None), embeddings)` on vLLM. `layer_names` take vortex module names on both.
- `evo2_40b` and `evo2_40b_base` are vortex-only; on vLLM they raise and point to `evo2_20b`, which has equivalent performance with higher efficiency.
- `evo2.scoring` helpers (`positional_entropies`, `score_perplexity_along_sequence`, `score_sequences`, `score_sequences_rc`) accept an `Evo2` object on either backend, and `tokenizer` defaults to the model's. Passing a vortex `StripedHyena` with its tokenizer still works.
- vortex-only arguments raise `TypeError` on vortex when given vLLM-only keywords; on vLLM, vortex-only options (`use_kernels`, `batched`, `cached_generation`, `force_prompt_threshold`, `batch_size` in `score_sequences`) are accepted and ignored with a warning.
- `utils.make_phylotag_from_gbif` returns `None` when GBIF has no match or is unreachable (previously raised `UnboundLocalError` or returned a `D__NONE...` tag) and no longer needs `requests`.

### Compatibility
- Python 3.11-3.12; vLLM backend: Python 3.12, vLLM 0.26.0 (on CUDA 12 drivers, the `+cu129` wheel), torch 2.11, `vllm-evo2>=0.1,<0.2`. Vortex backend: unchanged (`vtx>=1.1.0`, torch 2.6/2.7, Transformer Engine and flash-attn as before).
- The two stacks cannot share one environment. `vtx` remains a base dependency, so it is also installed (unused) in a vLLM environment.
- On vLLM, `Evo2.model` is the plugin's `StripedHyena2ForCausalLM` (different module names; read-only). Code that hooks vortex modules directly (e.g. the sparse autoencoder notebook) needs the vortex backend.
- Documented numerical differences on vLLM: fp32 log-softmax in `score_sequences`, `temperature=0` is greedy. Both backends match the reference losses in `test_evo2` to 1e-3.
