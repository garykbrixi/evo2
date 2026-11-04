"""Smoke test: load the smallest model and run one forward.

    python -m evo2.test.test_model_load [--backend auto|vortex|vllm]
"""
import argparse

import torch

from evo2 import Evo2


def test_model_load(backend: str = 'auto'):
    model = Evo2('evo2_1b_base', backend=backend)
    ids = torch.tensor(model.tokenizer.tokenize('ACGT' * 16), dtype=torch.int).unsqueeze(0).to('cuda:0')
    outputs, _ = model(ids)
    logits = outputs[0]
    assert tuple(logits.shape) == (1, 64, 512), logits.shape
    assert torch.isfinite(logits.float()).all()
    print(f"ok: backend={model.backend} logits {tuple(logits.shape)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['auto', 'vortex', 'vllm'], default='auto')
    test_model_load(parser.parse_args().backend)
