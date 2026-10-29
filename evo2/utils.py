import os

MODEL_NAMES = [
    'evo2_40b',
    'evo2_7b',
    'evo2_20b',
    'evo2_40b_base',
    'evo2_7b_base',
    'evo2_1b_base',
    'evo2_7b_262k',
    'evo2_7b_microviridae',
]

HF_MODEL_NAME_MAP = {
    'evo2_40b': 'arcinstitute/evo2_40b',
    'evo2_7b': 'arcinstitute/evo2_7b',
    'evo2_20b': 'arcinstitute/evo2_20b',
    'evo2_40b_base': 'arcinstitute/evo2_40b_base',
    'evo2_7b_base': 'arcinstitute/evo2_7b_base',
    'evo2_1b_base': 'arcinstitute/evo2_1b_base',
    'evo2_7b_262k': 'arcinstitute/evo2_7b_262k',
    'evo2_7b_microviridae': 'evo-design/evo-2-7b-8k-microviridae',
}

CONFIG_MAP = {
    'evo2_7b': 'configs/evo2-7b-1m.yml',
    'evo2_40b': 'configs/evo2-40b-1m.yml',
    'evo2_20b': 'configs/evo2-20b-1m.yml',
    'evo2_7b_base': 'configs/evo2-7b-8k.yml',
    'evo2_40b_base': 'configs/evo2-40b-8k.yml',
    'evo2_1b_base': 'configs/evo2-1b-8k.yml',
    'evo2_7b_262k': 'configs/evo2-7b-262k.yml',
    'evo2_7b_microviridae': 'configs/evo2-7b-8k.yml',
}


# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------

BACKENDS = ('auto', 'vortex', 'vllm')
BACKEND_ENV_VAR = 'EVO2_BACKEND'

VLLM_INSTALL_HINT = (
    "The vLLM backend needs its own Python environment (vLLM pins a newer torch than "
    "the vortex stack):\n"
    "    pip install vllm==0.26.0   # on a CUDA 12 driver use the +cu129 wheel, see README\n"
    "    pip install 'evo2[vllm]'\n"
    "See README.md, 'Inference backends'."
)


def vllm_backend_available() -> bool:
    """True if both vLLM and the vllm_evo2 plugin can be imported.

    Uses find_spec so that merely asking does not import vLLM (which is slow
    and has side effects).
    """
    import importlib.util
    return all(
        importlib.util.find_spec(name) is not None
        for name in ('vllm', 'vllm_evo2')
    )


def resolve_backend(backend: str = 'auto') -> str:
    """Return 'vortex' or 'vllm'.

    'auto' (the default) picks vLLM when it is importable and vortex otherwise.
    The two stacks cannot share an environment, so an importable vLLM means the
    user built a vLLM environment on purpose. The EVO2_BACKEND environment
    variable overrides 'auto' only; an explicit argument always wins.
    """
    if backend is None:
        backend = 'auto'
    backend = str(backend).lower()
    if backend == 'auto':
        env = os.environ.get(BACKEND_ENV_VAR, '').strip().lower()
        if env:
            backend = env
    if backend not in BACKENDS:
        raise ValueError(
            f"Invalid backend {backend!r}. Should be one of: {', '.join(BACKENDS)}."
        )
    if backend == 'auto':
        return 'vllm' if vllm_backend_available() else 'vortex'
    if backend == 'vllm' and not vllm_backend_available():
        raise ImportError(
            f"backend='vllm' was requested but vLLM and/or vllm_evo2 are not installed.\n"
            + VLLM_INSTALL_HINT
        )
    return backend


def make_phylotag_from_gbif(
        species_name: str,
        timeout: float = 30.0,
) -> str:
    """
    Returns the phylogenetic tag for a species (to get new tags not in the metadata),
    e.g. '|D__ANIMALIA;P__CHORDATA;...;S__PHASCOLARCTOS CINEREUS|', from the GBIF
    species API. Returns None (and prints a message) if GBIF has no match.
    """
    import json
    import urllib.parse
    import urllib.request

    url = "https://api.gbif.org/v1/species/match?" + urllib.parse.urlencode({"name": species_name})
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.load(response)
    except Exception as e:  # network error, HTTP error, bad JSON
        print(f"Could not find taxonomy for {species_name} ({e})")
        return None

    ranks = ("kingdom", "phylum", "class", "order", "family", "genus", "species")
    if not data.get("kingdom"):
        print(f"Could not find taxonomy for {species_name}")
        return None
    phylo_tag = ";".join(f"{rank[0]}__{data.get(rank)}" for rank in ranks)
    # GBIF's kingdom is the tag's domain field ('d__'), as before.
    phylo_tag = "d" + phylo_tag[1:]
    return ("|" + phylo_tag + "|").upper()
