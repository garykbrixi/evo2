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

    # Checkpoint resolution lives in the plugin; evo2 must not keep its own copy.
    check("evo2 keeps no converted-repo map of its own (the plugin resolves names)",
          not hasattr(U, "HF_VLLM_MODEL_NAME_MAP"))

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
    check("the vortex-only models are real model names",
          vllm_backend.NOT_ON_VLLM <= set(U.MODEL_NAMES))

    n = len(RESULTS)
    print(f"\n{sum(RESULTS)}/{n} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
