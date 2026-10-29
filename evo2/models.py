from functools import partial
import os
import pkgutil
import torch
from typing import List, Tuple, Dict, Union
import warnings
import yaml

from evo2.scoring import score_sequences, score_sequences_rc
from evo2.utils import MODEL_NAMES, HF_MODEL_NAME_MAP, CONFIG_MAP, resolve_backend

# vortex (vtx) and vLLM pin incompatible torch versions and never share an
# environment, so each backend is imported only when it is used. `import evo2`
# therefore works in either environment.


def _has_te() -> bool:
    try:
        import transformer_engine  # noqa: F401
        return True
    except ImportError:
        return False


class Evo2:
    def __init__(
        self,
        model_name: str = MODEL_NAMES[1],
        local_path: str = None,
        use_kernels: bool = False,
        backend: str = 'auto',
        **backend_kwargs,
    ):
        """
        Load an Evo 2 checkpoint.

        Uses local_path if specified, otherwise checks if in local HuggingFace ~cache.
        Automatically downloads checkpoint from HuggingFace if it does not exist locally.

        backend selects the inference engine:
          - 'auto' (default): vLLM if `vllm` and `evo2_flash` are importable, else vortex.
            The EVO2_BACKEND environment variable overrides 'auto'.
          - 'vortex': the reference StripedHyena 2 implementation (vtx). Handles device
            placement itself and splits the model across multiple GPUs if available
            (Evo 2 40B needs several H100s). Select GPUs with CUDA_VISIBLE_DEVICES; do
            not call .to(device) manually.
          - 'vllm': the evo2_flash plugin. One GPU per model. evo2_40b and
            evo2_40b_base are not available on vLLM (use evo2_20b, or the vortex
            backend). Keyword arguments beyond those listed here go to the plugin's
            Evo2: fp8_mode, max_model_len, gpu_memory_utilization, kv_cache_dtype,
            speculative, cfg, cache_dir, and vllm.LLM arguments. The plugin downloads
            its own converted checkpoint by model name; local_path may be a vortex
            .pt (converted once) or a converted checkpoint directory.
        The two backends agree to the authors' correctness gate (see README,
        'Inference backends', for the documented semantic differences).

        use_kernels enables Vortex's opt-in Triton kernels for the Hyena short/medium/long
        convolutions (HC{S,M,L}), which can accelerate inference. Requires vtx>=1.1.0 and is
        off by default. The kernels fall back to the standard path if unavailable; always
        validate outputs when enabling them. The vLLM backend ignores it.

        Notes:
        You can change where HuggingFace downloads to by setting the HF_HOME environment
        variable.
        """
        if model_name not in MODEL_NAMES:
            raise ValueError(
                f'Invalid model name {model_name}. Should be one of: '
                f'{", ".join(MODEL_NAMES)}.'
            )
        self.model_name = model_name
        self.backend = resolve_backend(backend)
        self._vllm = None
        self._model = None

        if self.backend == 'vllm':
            from evo2.backends import vllm_backend

            self._vllm = vllm_backend.load(
                model_name, local_path=local_path, use_kernels=use_kernels, **backend_kwargs
            )
            self.tokenizer = self._vllm.tokenizer
            print(f"[evo2] {model_name}: backend=vllm ({vllm_backend.describe(self._vllm)})")
            return

        if backend_kwargs:
            raise TypeError(
                f"Unexpected keyword arguments for the vortex backend: "
                f"{sorted(backend_kwargs)}. They apply to backend='vllm' only."
            )
        from vortex.model.tokenizer import CharLevelTokenizer

        config_path = CONFIG_MAP[model_name]

        if local_path is not None:
            self._model = self.load_evo2_model(None, config_path, local_path, use_kernels=use_kernels)
        else:
            self._model = self.load_evo2_model(model_name, config_path, use_kernels=use_kernels)

        self.tokenizer = CharLevelTokenizer(512)
        import vortex
        print(f"[evo2] {model_name}: backend=vortex (vtx {getattr(vortex, '__version__', '?')}, "
              f"transformer_engine={'yes' if _has_te() else 'no'}, use_kernels={use_kernels})")

    @property
    def model(self):
        """The live model: vortex `StripedHyena`, or the plugin's `StripedHyena2ForCausalLM`.

        Module names differ between the two; pass vortex names to `forward(layer_names=...)`
        and they are mapped on either backend.
        """
        if self._vllm is not None:
            return self._vllm.model
        return self._model

    @model.setter
    def model(self, value):
        if self._vllm is not None:
            # `m.model = m.model.eval()` and similar in-place idioms are fine; a
            # different module cannot be swapped into a running vLLM engine.
            if value is self._vllm.model:
                return
            raise AttributeError("model is read-only on the vLLM backend")
        self._model = value

    def forward(
        self,
        input_ids: torch.Tensor,
        return_embeddings: bool = False,
        layer_names=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with optional embedding extraction.
        
        Args:
            input_ids: Input token IDs
            return_embeddings: If True, returns embeddings from specified layers
            layer_names: List of layer names to extract embeddings from. Required if
                return_embeddings=True
            
        Returns:
            Tuple of (outputs, embeddings_dict) if return_embeddings=True
            Tuple of (outputs, None) otherwise
            where outputs[0] is the logits tensor of shape (batch, length, vocab) on
            either backend (outputs[1] is vortex's inference-params dict, None on vLLM).
        """
        if self._vllm is not None:
            # vortex's StripedHyena.forward returns (logits, inference_params), so on the
            # vortex backend this method returns ((logits, params), embeddings) and callers
            # write `outputs, _ = model(ids); logits = outputs[0]` (README). The plugin
            # returns the logits tensor directly; wrap it so both backends unpack alike.
            logits, embeddings = self._vllm.forward(input_ids, return_embeddings, layer_names)
            return (logits, None), embeddings

        embeddings = {}
        handles = []
        
        if return_embeddings:
            if layer_names is None:
                raise ValueError(
                    "layer_names must be specified when return_embeddings=True. Look at "
                    "evo2_model.model.state_dict().keys() to see available layers."
                )
                
            def hook_fn(layer_name):
                def hook(_, __, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    embeddings[layer_name] = output.detach()
                return hook
                
            # Register hooks for requested layers
            for name in layer_names:
                layer = self.model.get_submodule(name)
                handles.append(layer.register_forward_hook(hook_fn(name)))
        
        try:
            # Original forward pass
            with torch.no_grad():
                logits = self.model.forward(input_ids)
            
            if return_embeddings:
                return logits, embeddings
            return logits, None
            
        finally:
            for handle in handles:
                handle.remove()

    def __call__(self, input_ids, return_embeddings=False, layer_names=None):
        return self.forward(input_ids, return_embeddings, layer_names)

    def score_sequences(
        self,
        seqs: List[str],
        batch_size: int = 1,
        prepend_bos: bool = False,
        reduce_method: str = 'mean',
        average_reverse_complement: bool = False,
    ) -> List[float]:
        if self._vllm is not None:
            if batch_size not in (1, None):
                warnings.warn(
                    "backend='vllm': batch_size has no effect (all sequences are scored "
                    "in one continuously batched call).", stacklevel=2)
            return self._vllm.score_sequences(
                seqs,
                batch_size=batch_size,
                prepend_bos=prepend_bos,
                reduce_method=reduce_method,
                average_reverse_complement=average_reverse_complement,
            )

        scoring_func = partial(
            score_sequences_rc if average_reverse_complement else score_sequences,
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            prepend_bos=prepend_bos,
            reduce_method=reduce_method,
        )

        with torch.no_grad():
            try:
                scores = scoring_func(seqs)
            except Exception as e:
                raise RuntimeError(f"Error during sequence scoring: {str(e)}") from e

        return scores
    
    def generate(
        self,
        prompt_seqs: List[str],
        n_tokens: int = 500,
        temperature: float = 1.0,
        top_k: int = 4,
        top_p: float = 1.0,
        batched: bool = True,
        cached_generation: bool = True,
        verbose: int = 1,
        force_prompt_threshold: int = None,
        **kwargs,
    ):
        """
        Generate sequences from a list of prompts.

        Returns a GenerationOutput with `sequences`, `logprobs_mean` and `logits`
        (vortex's dataclass, or the plugin's equivalent with the same fields).

        force_prompt_threshold: If specified, avoids OOM errors through teacher forcing if the prompt is longer than this threshold.

        If force_prompt_threshold is none, sets default assuming 1xH100 (evo2_7b) and 2xH100 (evo2_40b) to help avoid OOM errors.

        Extra keyword arguments (e.g. seed=, guidance=) are passed to the vLLM backend's
        generate; the vortex backend accepts none.
        """
        if self._vllm is not None:
            ignored = [name for name, val, default in (
                ("batched", batched, True), ("cached_generation", cached_generation, True),
                ("force_prompt_threshold", force_prompt_threshold, None)) if val != default]
            if ignored:
                warnings.warn(
                    f"backend='vllm': {ignored} have no effect (vLLM always batches, keeps "
                    "state, and bounds prefill memory with chunked prefill).", stacklevel=2)
            return self._vllm.generate(
                prompt_seqs,
                n_tokens=n_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                batched=batched,
                cached_generation=cached_generation,
                verbose=verbose,
                force_prompt_threshold=force_prompt_threshold,
                **kwargs,
            )
        if kwargs:
            raise TypeError(
                f"Unexpected keyword arguments for the vortex backend: {sorted(kwargs)}. "
                f"They apply to backend='vllm' only."
            )
        from vortex.model.generation import generate as vortex_generate

        with torch.no_grad():
            output = vortex_generate(
                prompt_seqs=prompt_seqs,
                model=self.model,
                tokenizer=self.tokenizer,
                n_tokens=n_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                batched=batched,
                cached_generation=cached_generation,
                verbose=verbose,
                force_prompt_threshold=force_prompt_threshold,
            )
            return output


    def load_evo2_model(
            self,
            model_name: str = MODEL_NAMES[1],
            config_path: str = None,
            local_path: str = None,
            remove_shards: bool = True,
            use_kernels: bool = False,
    ):
        """
        Load HuggingFace checkpoint using StripedHyena 2 (vortex backend).

        If local_path is specified, loads from local_path.
        Otherwise, downloads from HuggingFace.
        If remove_shards is True, removes HF checkpoint shards after merging to .pt file.
        """
        from huggingface_hub import snapshot_download, constants, hf_hub_download
        from vortex.model.model import StripedHyena
        from vortex.model.utils import dotdict, load_checkpoint

        HAS_TE = _has_te()

        if local_path is not None:
            print(f"Loading model from {local_path}...")
            print(f"Loading config from {config_path}...")
            config = yaml.safe_load(pkgutil.get_data(__name__, config_path))
            config = dotdict(config)

            if config.get("use_fp8_input_projections", False) and not HAS_TE:
                is_7b_model = "7b" in (model_name or "") or "7b" in (config_path or "")
                if is_7b_model:
                    warnings.warn(
                        "Transformer Engine not installed. "
                        "Falling back to bf16 projections (use_fp8_input_projections=False). "
                    )
                    config.use_fp8_input_projections = False
                else:
                    raise ImportError(
                        f"Model '{model_name or config_path}' requires FP8 via Transformer Engine, "
                        f"which is not installed.\n"
                        f"Install with: pip install transformer_engine\n"
                        f"For inference without TE, use any 7b model: Evo2('evo2_7b')"
                    )

            if use_kernels:
                # Enable Vortex's opt-in Triton HC{S,M,L} inference kernels.
                config.use_hcs_kernel = True
                config.use_hcm_kernel = True
                config.use_hcl_kernel = True

            model = StripedHyena(config)
            load_checkpoint(model, local_path)
            return model
        
        hf_model_name = HF_MODEL_NAME_MAP[model_name]
        filename = f"{model_name}.pt"
        
        final_weights_path = os.path.join(os.path.dirname(constants.HF_HUB_CACHE), filename)
        if os.path.exists(final_weights_path):
            print(f"Found existing merged file: {final_weights_path}")
            weights_path = final_weights_path
            
            hf_hub_download(
                repo_id=hf_model_name, 
                filename="config.json"
            )
        else:
            repo_dir = snapshot_download(
                repo_id=hf_model_name,
            )
            
            # Check if the complete file already exists in the repo
            repo_weights_path = os.path.join(repo_dir, filename)
            if os.path.exists(repo_weights_path):
                print(f"Found complete file in repo: {filename}")
                weights_path = repo_weights_path
            else:
                print(f"Looking for checkpoint shards for {filename}")
                parts = []
                part_num = 0

                while True:
                    part_path = os.path.join(repo_dir, f"{filename}.part{part_num}")
                    if os.path.exists(part_path):
                        parts.append(part_path)
                        part_num += 1
                    else:
                        break
                
                if parts:
                    print(f"Found {len(parts)} shards, merging them...")
                    with open(final_weights_path, 'wb') as outfile:
                        for part in parts:
                            print(f"Merging shard: {os.path.basename(part)}")
                            with open(part, 'rb') as infile:
                                while True:
                                    chunk = infile.read(8192*1024)
                                    if not chunk: 
                                        break
                                    outfile.write(chunk)
                    
                    print(f"Successfully merged all shards into {final_weights_path}")
                    weights_path = final_weights_path
                    if remove_shards and os.path.exists(final_weights_path):
                        for part in parts:
                            real_path = os.path.realpath(part)
                            if os.path.exists(real_path):
                                os.remove(real_path)
                            if os.path.exists(part):
                                os.remove(part)
                else:
                    raise FileNotFoundError(f"Could not find {filename} or any of its shards in {repo_dir}")
                
        config = yaml.safe_load(pkgutil.get_data(__name__, config_path))
        global_config = dotdict(config, Loader=yaml.FullLoader)

        if global_config.get("use_fp8_input_projections", False) and not HAS_TE:
            if "7b" in (model_name or ""):
                warnings.warn(
                    "Transformer Engine not installed. "
                    "Falling back to bf16 projections (use_fp8_input_projections=False). "
                )
                global_config.use_fp8_input_projections = False
            else:
                raise ImportError(
                    f"Model '{model_name}' requires FP8 via Transformer Engine, "
                    f"which is not installed.\n"
                    f"Install with: pip install transformer_engine\n"
                    f"For inference without TE, use any 7b model: Evo2('evo2_7b')"
                )

        if use_kernels:
            # Enable Vortex's opt-in Triton HC{S,M,L} inference kernels.
            global_config.use_hcs_kernel = True
            global_config.use_hcm_kernel = True
            global_config.use_hcl_kernel = True

        model = StripedHyena(global_config)
        load_checkpoint(model, weights_path)

        return model
