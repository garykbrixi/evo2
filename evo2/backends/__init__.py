"""Inference backends for Evo 2.

`evo2.Evo2` dispatches to one of two engines:

- ``vortex`` (default when vLLM is absent): the reference StripedHyena 2
  implementation, in-process, multi-GPU model splitting for 40B.
- ``vllm``: the out-of-tree vLLM plugin ``vllm_evo2`` (continuous batching,
  paged Hyena state, prefix caching), one GPU per model.

Each backend lives in its own module and is imported lazily, so ``import evo2``
works in an environment that has only one of the two stacks installed.
"""
