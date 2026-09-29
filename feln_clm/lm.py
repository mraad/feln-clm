"""The fine-tuned generator: a base causal LM with its LoRA adapter merged into the weights.

Two backends with one interface for the constrained decoder: ``mlx`` (Mac) and ``torch``
(CUDA host). ``start(ids)`` prefills the prompt and returns the next-token logits [1, V];
``step(parents, tokens)`` keeps KV-cache rows ``parents`` (a beam reorder, may grow the
batch) and appends one token per row, returning logits [B, V]. All rows share one length,
so no padding or masks are needed.

Adapter file: ``{module}.lora_a`` [r, in] and ``{module}.lora_b`` [out, r] per adapted
linear layer; the merged weight is W + scale · B·A (computed in float32).
"""

from __future__ import annotations

import numpy as np

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def encode(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False)


class MLX:
    def __init__(self, base: str, adapter: str | None = None, scale: float = 0.0):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm import load

        self.model, self.tok = load(base)
        if adapter:
            ab, weights = mx.load(adapter), dict(tree_flatten(self.model.parameters()))
            merged = []
            for key in [k for k in ab if k.endswith(".lora_a")]:
                name = key.removesuffix(".lora_a")
                w = weights[f"{name}.weight"]
                delta = scale * (
                    ab[f"{name}.lora_b"].astype(mx.float32) @ ab[key].astype(mx.float32)
                )
                merged.append((f"{name}.weight", (w.astype(mx.float32) + delta).astype(w.dtype)))
            self.model.load_weights(merged, strict=False)
            mx.eval(self.model.parameters())

    def _out(self, logits) -> np.ndarray:
        import mlx.core as mx

        return np.array(logits[:, -1].astype(mx.float32))

    def start(self, ids: list[int]) -> np.ndarray:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        self.cache = make_prompt_cache(self.model)
        return self._out(self.model(mx.array([ids]), cache=self.cache))

    def step(self, parents: list[int], tokens: list[int]) -> np.ndarray:
        import mlx.core as mx

        idx = mx.array(parents)
        for c in self.cache:
            c.keys, c.values = c.keys[idx], c.values[idx]
        return self._out(self.model(mx.array(tokens)[:, None], cache=self.cache))


class Torch:
    def __init__(self, base: str, adapter: str | None = None, scale: float = 0.0,
                 device: str = "cuda"):  # fmt: skip
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(base)
        self.model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
        self.model.to(device).eval()
        if adapter:
            from safetensors.torch import load_file

            ab = load_file(adapter, device=device)
            with torch.no_grad():
                for name, mod in self.model.named_modules():
                    if f"{name}.lora_a" in ab:
                        a, b = ab[f"{name}.lora_a"].float(), ab[f"{name}.lora_b"].float()
                        mod.weight.copy_(
                            (mod.weight.float() + scale * (b @ a)).to(mod.weight.dtype)
                        )

    def _run(self, ids):
        with self.torch.inference_mode():
            out = self.model(ids, past_key_values=getattr(self, "cache", None), use_cache=True)
        self.cache = out.past_key_values
        return out.logits[:, -1].float().cpu().numpy()

    def start(self, ids: list[int]) -> np.ndarray:
        self.cache = None
        return self._run(self.torch.tensor([ids], device=self.device))

    def step(self, parents: list[int], tokens: list[int]) -> np.ndarray:
        self.cache.reorder_cache(self.torch.tensor(parents, device=self.device))
        return self._run(self.torch.tensor(tokens, device=self.device)[:, None])


def load(backend: str, base: str, adapter: str | None = None, scale: float = 0.0):
    if backend == "mlx":
        return MLX(base, adapter, scale)
    if backend == "torch":
        return Torch(base, adapter, scale)
    raise ValueError(f"unknown backend {backend!r}: mlx or torch")
