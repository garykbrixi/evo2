"""The vLLM backend: a thin layer over the ``vllm_evo2`` plugin.

The plugin (https://github.com/ArcInstitute/evo2-vllm, PyPI ``vllm-evo2``)
provides ``vllm_evo2.Evo2``, which already mirrors ``evo2.Evo2``'s API. This
module only decides *which weights* it loads:

1. ``local_path`` given: handed to the plugin unchanged. It accepts a converted
   checkpoint directory or a vortex ``.pt`` (converted once into the cache).
2. Otherwise the converted safetensors repo from ``HF_VLLM_MODEL_NAME_MAP`` is
   downloaded into the Hugging Face cache and used.
3. If that repo is unavailable (not published, offline, no access), the plugin
   converts the vortex ``.pt`` locally, downloading it the way ``evo2`` does.

``EVO2_VLLM_WEIGHTS=convert`` skips step 2 and always converts locally.
"""

import json
import os
import warnings

from evo2.utils import HF_VLLM_MODEL_NAME_MAP, MODEL_NAMES, VLLM_INSTALL_HINT

# The plugin's forward() hooks the live model, so the engine must run in the
# calling process. vLLM reads this at import time.
_IN_PROCESS_ENV = ("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# The 40B models run on vortex only (multi-GPU); on vLLM the 20B replaces them.
NOT_ON_VLLM = frozenset({"evo2_40b", "evo2_40b_base"})

CONVERTED_PATTERNS = ["config.json", "*.safetensors", "*.safetensors.index.json",
                      "metadata/*", "surgery/*", "README.md"]


def _import_plugin():
    os.environ.setdefault(*_IN_PROCESS_ENV)
    try:
        import vllm_evo2  # noqa: F401
    except ImportError as e:
        raise ImportError(f"vllm_evo2 is not importable ({e}).\n{VLLM_INSTALL_HINT}") from e
    return vllm_evo2


def is_converted_checkpoint(path: str) -> bool:
    """True if `path` is a directory holding a converted StripedHyena2 checkpoint."""
    cfg = os.path.join(path, "config.json")
    if not (os.path.isdir(path) and os.path.exists(cfg)):
        return False
    try:
        with open(cfg) as f:
            return json.load(f).get("model_type") == "striped_hyena2"
    except (OSError, ValueError):
        return False


def download_converted_checkpoint(model_name: str, log=print) -> str | None:
    """Fetch the converted repo for `model_name`; None if it cannot be had.

    Any failure (no such repo, gated, offline, network) returns None so the
    caller can fall back to local conversion. The reason is logged.
    """
    repo = HF_VLLM_MODEL_NAME_MAP.get(model_name)
    if repo is None:
        return None
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(repo_id=repo, allow_patterns=CONVERTED_PATTERNS)
    except Exception as e:  # RepositoryNotFound, GatedRepo, HTTP, offline cache miss
        log(f"[evo2] converted checkpoint {repo} unavailable ({type(e).__name__}); "
            f"converting the vortex checkpoint locally instead")
        return None
    if not is_converted_checkpoint(path):
        log(f"[evo2] {repo} is not a converted StripedHyena2 checkpoint; "
            f"converting the vortex checkpoint locally instead")
        return None
    return path


def load(model_name: str, local_path: str | None = None, use_kernels: bool = False,
         log=print, **engine_kwargs):
    """Build and return a `vllm_evo2.Evo2`.

    `use_kernels` is vortex-only and ignored here (the plugin always runs its
    own kernels). `engine_kwargs` go to `vllm_evo2.Evo2`: `max_model_len`,
    `gpu_memory_utilization`, `kv_cache_dtype`, `speculative`, `cfg`,
    `cache_dir`, and any `vllm.LLM` argument (the plugin warns for arguments
    outside its validated set).
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"Invalid model name {model_name}. Should be one of: {', '.join(MODEL_NAMES)}."
        )
    if model_name in NOT_ON_VLLM:
        raise ValueError(
            f"{model_name} is not available on the vLLM backend; evo2_20b replaces it. "
            f"Use Evo2('evo2_20b'), or Evo2('{model_name}', backend='vortex')."
        )
    if use_kernels:
        warnings.warn("use_kernels is a vortex option; the vLLM backend ignores it.",
                      stacklevel=3)
    vllm_evo2 = _import_plugin()

    if local_path is None and os.environ.get("EVO2_VLLM_WEIGHTS", "hub") != "convert":
        local_path = download_converted_checkpoint(model_name, log=log)

    model = vllm_evo2.Evo2(model_name, local_path=local_path, **engine_kwargs)
    return model


def describe(model) -> str:
    """One line naming the engine, for the load banner."""
    import vllm
    return (f"vllm {vllm.__version__}, checkpoint {model.checkpoint}, "
            f"max_model_len {model.max_model_len}")
