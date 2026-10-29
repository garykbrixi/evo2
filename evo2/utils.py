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

# Converted (safetensors) checkpoints for the vLLM backend, one repo per model.
# The 1B and 20B also carry a `surgery/` sidecar so the same repo runs on GPUs
# without FP8 tensor cores (vllm_evo2/surgery.py). The 40B is not published for
# vLLM (the 20B replaces it); a name missing here is converted locally from its
# vortex .pt on first use.
HF_VLLM_MODEL_NAME_MAP = {
    'evo2_20b': 'gbrixi/evo2_20b_vllm',
    'evo2_7b': 'gbrixi/evo2_7b_vllm',
    'evo2_7b_base': 'gbrixi/evo2_7b_base_vllm',
    'evo2_7b_262k': 'gbrixi/evo2_7b_262k_vllm',
    'evo2_1b_base': 'gbrixi/evo2_1b_base_vllm',
    'evo2_7b_microviridae': 'gbrixi/evo2_7b_microviridae_vllm',
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
) -> dict:
    """
    Returns phylogenetic tags for a given species, to get new tags not in the metadata
    """

    import requests
    def get_taxonomy_from_gbif(species_name):
        url = f"https://api.gbif.org/v1/species/match?name={species_name}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                "kingdom": data.get("kingdom"),
                "phylum": data.get("phylum"),
                "class": data.get("class"),
                "order": data.get("order"),
                "family": data.get("family"),
                "genus": data.get("genus"),
                "species": data.get("species")
            }
        else:
            print(f"Could not find taxonomy for {species_name}")

    taxonomy = get_taxonomy_from_gbif(species_name)
    if taxonomy:
        phylo_tag = (
        f'd__{taxonomy["kingdom"]};'
        f'p__{taxonomy["phylum"]};'
        f'c__{taxonomy["class"]};'
        f'o__{taxonomy["order"]};'
        f'f__{taxonomy["family"]};'
        f'g__{taxonomy["genus"]};'
        f's__{taxonomy["species"]}'
    ).upper()
        phylo_tag = '|'+phylo_tag+'|'
    else:
        print(f"Could not find taxonomy for {species_name}")

    return phylo_tag.upper()

