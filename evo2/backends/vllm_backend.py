"""The vLLM backend: a thin layer over the ``evo2_flash`` plugin.

The plugin (PyPI ``evo2-flash``) provides ``evo2_flash.Evo2``, which already
mirrors ``evo2.Evo2``'s API, and resolves checkpoints itself: by name it
downloads the published converted checkpoint for the model, and ``local_path``
may be a converted directory or a vortex ``.pt`` (converted once). It also picks
the right form for the GPU (``fp8_mode``). This module only refuses the models
the vLLM backend does not run, before vLLM is imported.
"""

import os
import warnings

from evo2.utils import MODEL_NAMES, VLLM_INSTALL_HINT

# The plugin's forward() hooks the live model, so the engine must run in the
# calling process. vLLM reads this at import time.
_IN_PROCESS_ENV = ("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# The 40B models run on vortex only (multi-GPU); on vLLM the 20B replaces them.
NOT_ON_VLLM = frozenset({"evo2_40b", "evo2_40b_base"})


def _import_plugin():
    os.environ.setdefault(*_IN_PROCESS_ENV)
    try:
        import evo2_flash  # noqa: F401
    except ImportError as e:
        raise ImportError(f"evo2_flash is not importable ({e}).\n{VLLM_INSTALL_HINT}") from e
    return evo2_flash


def load(model_name: str, local_path: str | None = None, use_kernels: bool = False,
         log=print, **engine_kwargs):
    """Build and return an `evo2_flash.Evo2`.

    `use_kernels` is vortex-only and ignored here (the plugin always runs its
    own kernels). `engine_kwargs` go to `evo2_flash.Evo2`: `fp8_mode`,
    `max_model_len`, `gpu_memory_utilization`, `kv_cache_dtype`, `cache_dir`,
    and `vllm.LLM` arguments (the plugin warns for ones outside its validated
    set).
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"Invalid model name {model_name}. Should be one of: {', '.join(MODEL_NAMES)}."
        )
    if model_name in NOT_ON_VLLM:
        raise ValueError(
            f"{model_name} is not available on the vLLM backend. evo2_20b has equivalent "
            f"performance with higher efficiency: use Evo2('evo2_20b'). To run "
            f"{model_name} itself, use Evo2('{model_name}', backend='vortex') in a vortex "
            "environment."
        )
    if use_kernels:
        warnings.warn("use_kernels is a vortex option; the vLLM backend ignores it.",
                      stacklevel=3)
    evo2_flash = _import_plugin()
    return evo2_flash.Evo2(model_name, local_path=local_path, **engine_kwargs)


def describe(model) -> str:
    """One line naming the engine, for the load banner."""
    import vllm
    return (f"vllm {vllm.__version__}, checkpoint {model.checkpoint}, "
            f"max_model_len {model.max_model_len}")
