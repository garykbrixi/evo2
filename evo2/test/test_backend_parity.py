"""Cross-backend equivalence: vortex vs vLLM on the same checkpoint.

The two backends cannot share a Python environment (vLLM pins a newer torch than
the vortex stack), so this is a dump-and-compare test. Run `dump` once in each
environment, then `compare` anywhere (CPU, numpy only):

    # vortex environment
    python -m evo2.test.test_backend_parity dump --backend vortex --model_name evo2_7b --out vortex_7b.npz
    # vLLM environment
    python -m evo2.test.test_backend_parity dump --backend vllm   --model_name evo2_7b --out vllm_7b.npz
    # either
    python -m evo2.test.test_backend_parity compare vortex_7b.npz vllm_7b.npz

`dump` uses only the public evo2 API (`Evo2(...)`, `model(ids, ...)`,
`score_sequences`, `generate`), so it exercises exactly what users call. For
the sequences in evo2/test/data/prompts.csv (real genomic sequence, 6-7.6 kb)
it records:

  * per-token log p(x_t | x_<t) from the forward logits (fp32 log-softmax),
  * `score_sequences` (mean and sum reductions),
  * one intermediate embedding (default: the layer the README recommends,
    `blocks.{N-4}.mlp.l3`), per position (every `--emb_stride`-th) and mean-pooled,
  * a greedy continuation from the sequence midpoint (`top_k=1`), as in
    test_evo2_generation.py.

`compare` gates on likelihoods, never on token identity: on DNA the model sits
near maximum entropy and argmax flips under any change in rounding (vortex
disagrees with itself across precisions). Gates, all in nats per token:

  mean NLL over all sequences       |delta| <= --nll_tol   (default 1e-3, the authors' eps)
  per-sequence NLL                  |delta| <= --seq_tol   (default 5e-3)
  score_sequences (mean/sum/token)  agree with the forward-derived values
  pooled embedding cosine           >= --emb_cos           (default 0.99)

Per-token |delta log p| percentiles, per-position embedding cosine, greedy
identity to the true continuation and backend-vs-backend greedy agreement are
reported for information. Exit code 1 on a failed gate.
"""

import argparse
import csv
import json
import os
import platform
import time
from importlib import resources
from pathlib import Path

import numpy as np

EXPECTED_LOSS = {  # evo2/test/test_evo2.py
    'evo2_40b': 0.2159424, 'evo2_7b': 0.3476563, 'evo2_40b_base': 0.2149658,
    'evo2_7b_base': 0.3520508, 'evo2_1b_base': 0.501953125, 'evo2_20b': 0.2166748046875,
}


def read_prompts(input_file='prompts.csv'):
    if not Path(input_file).is_file():
        with resources.path('evo2.test.data', input_file) as p:
            input_file = p
    seqs = []
    with open(input_file, encoding='utf-8-sig', newline='') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            seqs.append(row[0])
    return seqs


def default_layer(model_name):
    import pkgutil
    import yaml
    from evo2.utils import CONFIG_MAP
    cfg = yaml.safe_load(pkgutil.get_data('evo2', CONFIG_MAP[model_name]))
    return f"blocks.{cfg['num_layers'] - 4}.mlp.l3"


def versions():
    import torch
    v = {'torch': torch.__version__, 'python': platform.python_version(),
         'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}
    for mod in ('vllm', 'vllm_evo2', 'vortex', 'evo2', 'transformer_engine'):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, '__version__', 'present')
        except Exception:
            pass
    return v


def dump(args):
    import torch
    from evo2 import Evo2

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    model = Evo2(args.model_name, backend=args.backend)
    layer = args.layer or default_layer(args.model_name)
    seqs = read_prompts(args.input)
    if args.max_len:
        seqs = [s[:args.max_len] for s in seqs]
    out = {'meta': json.dumps({'backend': model.backend, 'model_name': args.model_name,
                               'layer': layer, 'n_tokens': args.n_tokens,
                               'emb_stride': args.emb_stride, 'versions': versions(),
                               'time': time.strftime('%Y-%m-%dT%H:%M:%S')}),
           'n_seqs': len(seqs)}

    for i, seq in enumerate(seqs):
        ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).to('cuda:0')
        with torch.inference_mode():
            outputs, emb = model(ids, return_embeddings=True, layer_names=[layer])
        logits = outputs[0]                                          # (1, L, V), bf16
        lp = torch.log_softmax(logits[0].float(), dim=-1)           # (L, V) fp32
        tok = ids[0, 1:].long()
        tok_lp = lp[:-1].gather(1, tok[:, None])[:, 0]              # log p(x_t | x_<t)
        out[f'tok_logp_{i}'] = tok_lp.cpu().numpy().astype(np.float32)
        out[f'argmax_{i}'] = lp[:-1].argmax(-1).to(torch.int16).cpu().numpy()
        e = emb[layer][0].float()                                     # (L, D)
        out[f'emb_{i}'] = e[::args.emb_stride].cpu().numpy().astype(np.float32)
        out[f'emb_mean_{i}'] = e.mean(0).cpu().numpy().astype(np.float32)
        out[f'ids_{i}'] = ids[0].to(torch.int16).cpu().numpy()
        print(f"seq {i + 1}/{len(seqs)}: L={len(seq)} nll={-tok_lp.mean().item():.5f}", flush=True)
        del logits, emb, lp, e
        torch.cuda.empty_cache()

    out['score_mean'] = np.array(model.score_sequences(seqs, batch_size=1), dtype=np.float64)
    out['score_sum'] = np.array(model.score_sequences(seqs, batch_size=1, reduce_method='sum'), dtype=np.float64)

    # Greedy continuation from the midpoint (test_evo2_generation.py protocol).
    gens, targets = [], []
    for seq in seqs:
        mid = 2 * (len(seq) // 4)
        prompt, target = seq[:mid], seq[mid:mid + args.n_tokens]
        with torch.inference_mode():
            g = model.generate([prompt], n_tokens=args.n_tokens, temperature=1.0, top_k=1, top_p=1.0,
                               verbose=0)
        gens.append(g.sequences[0])
        targets.append(target)
    out['gen'] = np.array(gens)
    out['gen_target'] = np.array(targets)

    np.savez_compressed(args.out, **out)
    nll = float(np.mean([-out[f'tok_logp_{i}'].mean() for i in range(len(seqs))]))
    print(f"\nwrote {args.out}: backend={model.backend} mean_nll={nll:.7f}"
          + (f" (authors' value {EXPECTED_LOSS[args.model_name]:.7f}, delta {abs(nll - EXPECTED_LOSS[args.model_name]):.2e})"
             if args.model_name in EXPECTED_LOSS else ''))


def identity(a, b):
    n = min(len(a), len(b))
    return 100.0 * sum(x == y for x, y in zip(a[:n], b[:n])) / n if n else float('nan')


def compare(args):
    A, B = np.load(args.a, allow_pickle=False), np.load(args.b, allow_pickle=False)
    ma, mb = json.loads(str(A['meta'])), json.loads(str(B['meta']))
    print(f"A: {ma['backend']:<6} {ma['model_name']}  {ma['versions']}")
    print(f"B: {mb['backend']:<6} {mb['model_name']}  {mb['versions']}")
    if ma['model_name'] != mb['model_name']:
        print("FAIL: different models"); return 1
    n = int(A['n_seqs'])
    results = []

    def gate(name, value, tol, higher=False, fmt='{:.2e}'):
        ok = value >= tol if higher else value <= tol
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<52} {fmt.format(value)}  (tol {fmt.format(tol)})")

    print("\n== per-token log-likelihoods (forward) ==")
    nll_a, nll_b, deltas = [], [], []
    for i in range(n):
        if not np.array_equal(A[f'ids_{i}'], B[f'ids_{i}']):
            print(f"FAIL: sequence {i} tokens differ between dumps"); return 1
        la, lb = A[f'tok_logp_{i}'], B[f'tok_logp_{i}']
        nll_a.append(-la.mean()); nll_b.append(-lb.mean())
        d = np.abs(la - lb); deltas.append(d)
        gate(f"seq {i + 1} |NLL_A - NLL_B|", abs(nll_a[-1] - nll_b[-1]), args.seq_tol)
        print(f"  info  seq {i + 1} per-token |dlogp| p50 {np.percentile(d, 50):.2e}  p99 {np.percentile(d, 99):.2e}"
              f"  max {d.max():.2e}   argmax agreement {100 * np.mean(A[f'argmax_{i}'] == B[f'argmax_{i}']):.2f}%")
    mean_a, mean_b = float(np.mean(nll_a)), float(np.mean(nll_b))
    print(f"  mean NLL  A={mean_a:.7f}  B={mean_b:.7f}")
    gate("mean NLL |A - B|", abs(mean_a - mean_b), args.nll_tol)
    exp = EXPECTED_LOSS.get(ma['model_name'])
    if exp is not None:
        gate("mean NLL A vs authors' value", abs(mean_a - exp), 1e-3)
        gate("mean NLL B vs authors' value", abs(mean_b - exp), 1e-3)

    print("\n== score_sequences ==")
    for key, red in (('score_mean', 'mean'), ('score_sum', 'sum')):
        gate(f"score_sequences({red}) max |A - B| per token",
             float(np.max(np.abs(A[key] - B[key]) / (1 if red == 'mean' else
                                                      np.array([len(A[f'tok_logp_{i}']) for i in range(n)])))),
             args.seq_tol)
    # Each backend's score_sequences must agree with its own forward (same model, two code paths).
    for name, D in (('A', A), ('B', B)):
        own = np.array([D[f'tok_logp_{i}'].mean() for i in range(n)])
        gate(f"{name}: score_sequences(mean) vs forward-derived mean", float(np.max(np.abs(own - D['score_mean']))),
             args.self_tol)

    print(f"\n== embeddings ({ma['layer']}) ==")
    for i in range(n):
        ea, eb = A[f'emb_{i}'], B[f'emb_{i}']
        cos = np.sum(ea * eb, 1) / (np.linalg.norm(ea, axis=1) * np.linalg.norm(eb, axis=1) + 1e-30)
        pa, pb = A[f'emb_mean_{i}'], B[f'emb_mean_{i}']
        pooled = float(pa @ pb / (np.linalg.norm(pa) * np.linalg.norm(pb) + 1e-30))
        gate(f"seq {i + 1} pooled-embedding cosine", pooled, args.emb_cos, higher=True, fmt='{:.6f}')
        print(f"  info  seq {i + 1} per-position cosine  median {np.median(cos):.6f}  min {cos.min():.6f}"
              f"   rel L2 {np.linalg.norm(ea - eb) / (np.linalg.norm(ea) + 1e-30):.3e}")

    print(f"\n== greedy generation ({ma['n_tokens']} tokens from the midpoint) ==")
    ids_a, ids_b, agree = [], [], []
    for i in range(n):
        ga, gb, t = str(A['gen'][i]), str(B['gen'][i]), str(A['gen_target'][i])
        ids_a.append(identity(ga, t)); ids_b.append(identity(gb, t)); agree.append(identity(ga, gb))
        first = next((k for k, (x, y) in enumerate(zip(ga, gb)) if x != y), None)
        print(f"  info  seq {i + 1} identity to truth A {ids_a[-1]:.1f}%  B {ids_b[-1]:.1f}%   A vs B {agree[-1]:.1f}%"
              f"   first divergence {first}")
    print(f"  info  mean identity to truth  A {np.mean(ids_a):.2f}%  B {np.mean(ids_b):.2f}%  (not gated)")

    ok = all(results)
    print(f"\n{sum(results)}/{len(results)} gates passed -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('dump', help='run one backend and save its outputs')
    d.add_argument('--backend', choices=['auto', 'vortex', 'vllm'], default='auto')
    d.add_argument('--model_name', default='evo2_7b')
    d.add_argument('--out', required=True)
    d.add_argument('--input', default='prompts.csv', help='CSV with one sequence per row (header skipped)')
    d.add_argument('--layer', default=None, help='vortex module name for the embedding (default blocks.{N-4}.mlp.l3)')
    d.add_argument('--n_tokens', type=int, default=200)
    d.add_argument('--emb_stride', type=int, default=16)
    d.add_argument('--max_len', type=int, default=None, help='truncate sequences (debugging)')
    d.set_defaults(fn=dump)
    c = sub.add_parser('compare', help='compare two dumps')
    c.add_argument('a'); c.add_argument('b')
    c.add_argument('--nll_tol', type=float, default=1e-3)
    c.add_argument('--seq_tol', type=float, default=5e-3)
    c.add_argument('--self_tol', type=float, default=2e-3,
                   help='score_sequences vs own forward (vortex scores in bf16 log-softmax, forward here in fp32)')
    c.add_argument('--emb_cos', type=float, default=0.99)
    c.set_defaults(fn=compare)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == '__main__':
    raise SystemExit(main())
