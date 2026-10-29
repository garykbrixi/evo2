"""CPU checks that the public helpers work with an `evo2.Evo2` on either backend.

No GPU, no model weights, no vortex or vLLM import: the backends are fakes that
return deterministic logits.

    python -m evo2.test.test_scoring_backends
"""
import io
import json
import os
import sys
import warnings
from unittest import mock

import numpy as np
import torch

from evo2 import Evo2, scoring
from evo2 import utils as U
from evo2.backends import vllm_backend

RESULTS = []
VOCAB = 512
_TABLE = torch.randn(VOCAB, VOCAB, generator=torch.Generator().manual_seed(0))


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def raises(fn, exc, contains=None):
    try:
        fn()
    except exc as e:
        return contains is None or contains in str(e)
    return False


def fake_logits(input_ids):
    ids = torch.as_tensor(input_ids).long().cpu()
    if ids.dim() == 1:
        ids = ids[None]
    return _TABLE[ids].to(torch.bfloat16)


class ByteTokenizer:
    """Same ids as vortex's CharLevelTokenizer(512) and the plugin's tokenizer."""
    eod_id, eos_id, pad_id, vocab_size = 0, 0, 1, VOCAB

    def tokenize(self, s):
        return list(s.encode())

    def detokenize(self, ids):
        return "".join(chr(max(32, min(int(i), VOCAB))) for i in ids)


class FakeStripedHyena(torch.nn.Module):
    """vortex StripedHyena: model(ids) -> (logits, inference_params)."""

    def forward(self, input_ids):
        return fake_logits(input_ids), None


class FakePluginEvo2:
    """vllm_evo2.Evo2: forward -> (logits, embeddings)."""

    def __init__(self):
        self.tokenizer = ByteTokenizer()
        self.model = FakePluginModule()
        self.checkpoint, self.max_model_len = "/fake", 8192
        self.calls = []

    def forward(self, input_ids, return_embeddings=False, layer_names=None):
        logits = fake_logits(input_ids)
        emb = {n: logits for n in layer_names} if return_embeddings else None
        return logits, emb

    def __call__(self, *a, **k):
        return self.forward(*a, **k)

    def generate(self, prompt_seqs, **kw):
        self.calls.append(("generate", prompt_seqs, kw))
        return "generation-output"

    def score_sequences(self, seqs, **kw):
        self.calls.append(("score_sequences", seqs, kw))
        return [0.0] * len(seqs)


class FakePluginModule(torch.nn.Module):
    """Stands in for the plugin's StripedHyena2ForCausalLM (a vLLM module)."""

    def forward(self, *a, **k):
        raise AssertionError("the vLLM engine module must never be called on token ids")


FakePluginModule.__module__ = "vllm_evo2.model"


def vortex_evo2():
    m = object.__new__(Evo2)
    m.model_name, m.backend, m._vllm = "evo2_7b", "vortex", None
    m._model = FakeStripedHyena()
    m.tokenizer = ByteTokenizer()
    return m


def vllm_evo2(**kwargs):
    """Build an evo2.Evo2 through the real __init__, with the plugin faked."""
    plugin = FakePluginEvo2()
    with mock.patch("evo2.models.resolve_backend", return_value="vllm"), \
            mock.patch.object(vllm_backend, "load", return_value=plugin) as load, \
            mock.patch.object(vllm_backend, "describe", return_value="fake"):
        m = Evo2("evo2_7b", **kwargs)
    return m, plugin, load


def main():
    seqs = ["ACGTACGTTTGA", "GGCATTACA"]
    vx = vortex_evo2()
    vl, plugin, load = vllm_evo2(fp8_mode="surgery")

    # --- Evo2 wrapper on the (fake) vLLM backend --------------------------------
    check("vllm: backend kwargs pass through to the plugin",
          load.call_args.kwargs.get("fp8_mode") == "surgery", str(load.call_args))
    check("vllm: tokenizer is the plugin's", vl.tokenizer is plugin.tokenizer)
    ids = torch.tensor(vl.tokenizer.tokenize(seqs[0]), dtype=torch.int).unsqueeze(0)
    (logits, params), emb = vl(ids)
    check("vllm: forward returns ((logits, None), None)",
          params is None and emb is None and tuple(logits.shape) == (1, len(seqs[0]), VOCAB))
    (lx, _), _ = vx(ids)
    check("forward logits agree across backends (same fake weights)", torch.equal(lx, logits))
    _, emb = vl(ids, return_embeddings=True, layer_names=["blocks.28.mlp.l3"])
    check("vllm: embeddings keyed by vortex names", set(emb) == {"blocks.28.mlp.l3"})
    check("vllm: model is the plugin's module", vl.model is plugin.model)
    vl.model = vl.model
    check("vllm: same-object model assignment is a no-op", vl.model is plugin.model)
    check("vllm: assigning another module raises",
          raises(lambda: setattr(vl, "model", torch.nn.Linear(1, 1)), AttributeError))
    vl.generate(["ACGT"], n_tokens=5, seed=3)
    name, _, kw = plugin.calls[-1]
    check("vllm: generate forwards n_tokens and extra kwargs",
          name == "generate" and kw["n_tokens"] == 5 and kw["seed"] == 3)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vl.score_sequences(seqs, batch_size=4)
    check("vllm: score_sequences warns that batch_size has no effect",
          any("batch_size" in str(x.message) for x in w))
    check("vllm: 40B refused before the plugin is touched",
          raises(lambda: vllm_backend.load("evo2_40b"), ValueError, "evo2_20b"))

    # --- vortex-path argument checks (no vortex import needed) ------------------
    with mock.patch("evo2.models.resolve_backend", return_value="vortex"):
        check("vortex: vLLM-only kwargs raise TypeError naming them",
              raises(lambda: Evo2("evo2_7b", fp8_mode="auto"), TypeError, "fp8_mode"))
    check("invalid model name raises ValueError",
          raises(lambda: Evo2("evo2_3b"), ValueError, "Invalid model name"))
    check("vortex: generate rejects vLLM-only kwargs",
          raises(lambda: vx.generate(["ACGT"], seed=1), TypeError, "seed"))

    # --- scoring helpers ----------------------------------------------------------
    ent_vx = scoring.positional_entropies(seqs, vx, device="cpu")
    ent_vl = scoring.positional_entropies(seqs, vl, device="cpu")
    ent_raw = scoring.positional_entropies(seqs, vx.model, vx.tokenizer, device="cpu")
    check("positional_entropies: evo2.Evo2 on vLLM works",
          [len(e) for e in ent_vl] == [len(s) for s in seqs])
    check("positional_entropies: vLLM == vortex Evo2 == raw vortex module (back-compat)",
          all(np.allclose(a, b) and np.allclose(a, c) for a, b, c in zip(ent_vl, ent_vx, ent_raw)))
    check("positional_entropies: the plugin's engine module gives a clear TypeError",
          raises(lambda: scoring.positional_entropies(seqs, vl.model, vl.tokenizer, device="cpu"),
                 TypeError, "Pass the Evo2 object"))
    check("positional_entropies: the plugin's own Evo2 works too",
          all(np.allclose(a, b) for a, b in zip(
              scoring.positional_entropies(seqs, plugin, device="cpu"), ent_vx)))

    p_vl = scoring.score_perplexity_along_sequence(vl, seqs[0], device="cpu")
    p_vx = scoring.score_perplexity_along_sequence(vx, seqs[0], device="cpu")
    check("score_perplexity_along_sequence: works on vLLM, agrees with vortex",
          p_vl.shape == (len(seqs[0]),) and np.allclose(p_vl, p_vx))
    e_vl = scoring.score_perplexity_along_sequence(vl, seqs[0], reverse_complement=False,
                                                   entropy=True, device="cpu")
    check("score_perplexity_along_sequence(entropy=True, rc=False) == positional_entropies",
          np.allclose(e_vl, ent_vx[0]))

    s_vl = scoring.score_sequences(seqs, vl, batch_size=1, device="cpu")
    s_raw = scoring.score_sequences(seqs, vx.model, vx.tokenizer, batch_size=1, device="cpu")
    check("scoring.score_sequences: evo2.Evo2 on vLLM == raw vortex module",
          np.allclose(s_vl, s_raw), f"{s_vl} vs {s_raw}")
    r_vl = scoring.score_sequences_rc(seqs, vl, batch_size=1, device="cpu")
    r_raw = scoring.score_sequences_rc(seqs, vx.model, vx.tokenizer, 1, device="cpu")
    check("scoring.score_sequences_rc: evo2.Evo2 on vLLM == raw vortex module",
          np.allclose(r_vl, r_raw))

    # --- utils.make_phylotag_from_gbif (network mocked) ---------------------------
    koala = {"kingdom": "Animalia", "phylum": "Chordata", "class": "Mammalia",
             "order": "Diprotodontia", "family": "Phascolarctidae",
             "genus": "Phascolarctos", "species": "Phascolarctos cinereus"}

    def fake_urlopen(payload):
        cm = mock.MagicMock()
        cm.__enter__.return_value = io.BytesIO(json.dumps(payload).encode())
        return mock.Mock(return_value=cm)

    with mock.patch("urllib.request.urlopen", fake_urlopen(koala)) as op:
        tag = U.make_phylotag_from_gbif("Phascolarctos cinereus")
    check("make_phylotag_from_gbif: tag format unchanged",
          tag == "|D__ANIMALIA;P__CHORDATA;C__MAMMALIA;O__DIPROTODONTIA;F__PHASCOLARCTIDAE;"
                 "G__PHASCOLARCTOS;S__PHASCOLARCTOS CINEREUS|", tag)
    check("make_phylotag_from_gbif: species name is URL-encoded",
          "name=Phascolarctos+cinereus" in op.call_args.args[0])
    with mock.patch("urllib.request.urlopen", fake_urlopen({"matchType": "NONE"})):
        check("make_phylotag_from_gbif: no match returns None (was UnboundLocalError/'D__NONE')",
              U.make_phylotag_from_gbif("Not a species") is None)
    with mock.patch("urllib.request.urlopen", side_effect=OSError("offline")):
        check("make_phylotag_from_gbif: network error returns None",
              U.make_phylotag_from_gbif("Phascolarctos cinereus") is None)

    check("no backend stack was imported",
          not ({"vortex", "vllm", "vllm_evo2"} & set(sys.modules)),
          str({"vortex", "vllm", "vllm_evo2"} & set(sys.modules)))

    n = len(RESULTS)
    print(f"\n{sum(RESULTS)}/{n} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
