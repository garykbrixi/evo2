"""CPU checks for the backend switch: no GPU, no model, no vortex or vLLM import.

    python -m evo2.test.test_backend_dispatch
"""
import os
import subprocess
import sys
from unittest import mock

import evo2.utils as U

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


def main():
    env_clear = {U.BACKEND_ENV_VAR: ""}

    with mock.patch.object(U, "vllm_backend_available", return_value=True), \
            mock.patch.dict(os.environ, env_clear):
        check("auto -> vllm when available", U.resolve_backend("auto") == "vllm")
        check("explicit vortex wins over availability", U.resolve_backend("vortex") == "vortex")
        check("explicit vllm", U.resolve_backend("vllm") == "vllm")
        check("None means auto", U.resolve_backend(None) == "vllm")
        check("case-insensitive", U.resolve_backend("VLLM") == "vllm")
        with mock.patch.dict(os.environ, {U.BACKEND_ENV_VAR: "vortex"}):
            check("EVO2_BACKEND overrides auto", U.resolve_backend("auto") == "vortex")
            check("EVO2_BACKEND does not override an explicit argument",
                  U.resolve_backend("vllm") == "vllm")

    with mock.patch.object(U, "vllm_backend_available", return_value=False), \
            mock.patch.dict(os.environ, env_clear):
        check("auto -> vortex when vllm is absent", U.resolve_backend("auto") == "vortex")
        check("explicit vllm raises ImportError when absent",
              raises(lambda: U.resolve_backend("vllm"), ImportError))
        try:
            U.resolve_backend("vllm")
        except ImportError as e:
            check("ImportError carries the install hint", "evo2[vllm]" in str(e))
        with mock.patch.dict(os.environ, {U.BACKEND_ENV_VAR: "vllm"}):
            check("EVO2_BACKEND=vllm with vllm absent raises",
                  raises(lambda: U.resolve_backend("auto"), ImportError))

    check("unknown backend raises ValueError",
          raises(lambda: U.resolve_backend("tensorrt"), ValueError))

    # The real availability probe must not import either stack.
    before = set(sys.modules)
    U.vllm_backend_available()
    check("availability probe imports neither vllm nor vortex",
          not ({"vllm", "vortex", "vllm_evo2"} & (set(sys.modules) - before)))

    # `import evo2` must not import vortex or vLLM (each backend is lazy).
    code = ("import sys, evo2; "
            "print(int('vortex' in sys.modules), int('vllm' in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    check("import evo2 imports neither backend", out.returncode == 0 and out.stdout.split() == ["0", "0"],
          out.stdout.strip() or out.stderr.strip()[-200:])

    # Every published converted repo belongs to a known model; the rest convert locally.
    check("HF_VLLM_MODEL_NAME_MAP names only known models",
          set(U.HF_VLLM_MODEL_NAME_MAP) <= set(U.MODEL_NAMES) == set(U.HF_MODEL_NAME_MAP) == set(U.CONFIG_MAP))

    # The loader must fetch the surgery sidecar, or non-FP8 GPUs never see it.
    from huggingface_hub.utils import filter_repo_objects
    from evo2.backends.vllm_backend import CONVERTED_PATTERNS
    layout = ["config.json", "model.safetensors", "metadata/hyena_in_proj_fp8_meta.safetensors",
              "surgery/surgery.json", "surgery/tensors.safetensors", "README.md"]
    got = set(filter_repo_objects(layout, allow_patterns=CONVERTED_PATTERNS))
    check("download patterns fetch every file of a converted repo, sidecar included",
          got == set(layout), str(sorted(set(layout) - got)))

    # The wrapper rejects engine kwargs on the vortex backend before loading anything.
    from evo2 import Evo2
    with mock.patch.object(U, "vllm_backend_available", return_value=False), \
            mock.patch.dict(os.environ, env_clear):
        check("vortex backend rejects engine kwargs with TypeError",
              raises(lambda: Evo2("evo2_1b_base", backend="vortex", max_model_len=8192), TypeError))
        check("invalid model name raises ValueError before backend resolution",
              raises(lambda: Evo2("evo2_3b", backend="vortex"), ValueError))

    # The 40B is vortex-only; on vLLM it must refuse before touching vLLM or the network.
    from evo2.backends import vllm_backend
    for name in ("evo2_40b", "evo2_40b_base"):
        try:
            vllm_backend.load(name)
            ok, msg = False, "no error"
        except ValueError as e:
            ok, msg = "evo2_20b" in str(e), str(e)[:60]
        except Exception as e:
            ok, msg = False, f"{type(e).__name__}: {e}"[:80]
        check(f"{name} on vLLM raises and points to evo2_20b", ok, msg)
    check("every vortex-only model has no converted repo",
          not (vllm_backend.NOT_ON_VLLM & set(U.HF_VLLM_MODEL_NAME_MAP)))

    n = len(RESULTS)
    print(f"\n{sum(RESULTS)}/{n} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
