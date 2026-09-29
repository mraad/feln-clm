"""Qwen3-8B encoders behind CLM's ``Embedder`` interface (its LRU cache and batching).

Both follow CLM's training recipe (``CLM/train/embed_utils.py``, vLLM ``--runner pooling``):
text tokenized without special tokens, tail kept, final-norm hidden state of the last
token, L2-normalised. ``mlx`` runs on Apple silicon; ``vllm`` in-process on CUDA (training
host); an http URL uses a running ``vllm serve ... --runner pooling`` endpoint.
"""

from __future__ import annotations

import numpy as np
from clm.embedder import Embedder, l2

MODEL = "Qwen/Qwen3-8B"


class MLXEmbedder(Embedder):
    """Misses arrive in one call; they run length-sorted in batches of at most ``tokens``
    padded tokens, so short prompts are not padded to long ones (right padding is causal,
    so each text's last-token state is unchanged by what it is batched with)."""

    def __init__(self, model: str = MODEL, max_tokens: int = 2048, tokens: int = 8192):
        super().__init__(url="mlx://" + model, model=model, max_tokens=max_tokens, batch=1 << 16)
        from mlx_lm import load

        self.lm, self.tok = load(model)
        self.tokens = tokens

    def _fetch(self, texts: list[str]) -> tuple[list[np.ndarray], int]:
        cap = (self.max_tokens or 2048) - 1
        ids = [self.tok.encode(t, add_special_tokens=False)[-cap:] or [220] for t in texts]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        out: list[np.ndarray] = [None] * len(ids)  # type: ignore[list-item]
        batch: list[int] = []
        for i in order + [-1]:
            if batch and (i < 0 or len(ids[i]) * (len(batch) + 1) > self.tokens):
                for j, v in zip(batch, self._forward([ids[j] for j in batch])):
                    out[j] = v
                batch = []
            if i >= 0:
                batch.append(i)
        return out, sum(map(len, ids))

    def _forward(self, ids: list[list[int]]) -> np.ndarray:
        import mlx.core as mx

        pad = np.zeros((len(ids), max(map(len, ids))), dtype=np.int32)  # right pad: causal, unseen
        for i, row in enumerate(ids):
            pad[i, : len(row)] = row
        hidden = self.lm.model(mx.array(pad))  # [B, T, H] after the final norm
        last = hidden[mx.arange(len(ids)), mx.array([len(r) - 1 for r in ids])]
        return l2(np.array(last.astype(mx.float32)))

    def healthy(self) -> bool:
        return True


class VLLMEmbedder(Embedder):
    def __init__(self, model: str = MODEL, max_tokens: int = 2048, gpu_mem: float = 0.5):
        super().__init__(url="vllm://" + model, model=model, max_tokens=max_tokens, batch=4096)
        from vllm import LLM

        self.llm = LLM(model=model, runner="pooling", max_model_len=max_tokens,
                       gpu_memory_utilization=gpu_mem, enable_prefix_caching=True)  # fmt: skip
        self.tok = self.llm.get_tokenizer()

    def _fetch(self, texts: list[str]) -> tuple[list[np.ndarray], int]:
        from vllm.inputs import TokensPrompt

        cap = (self.max_tokens or 2048) - 1
        ids = [self.tok.encode(t, add_special_tokens=False)[-cap:] or [220] for t in texts]
        outs = self.llm.embed([TokensPrompt(prompt_token_ids=i) for i in ids], use_tqdm=False)
        vecs = np.stack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in outs])
        return list(l2(vecs)), sum(map(len, ids))

    def healthy(self) -> bool:
        return True


def make(spec: str) -> Embedder:
    if spec == "mlx":
        return MLXEmbedder()
    if spec == "vllm":
        return VLLMEmbedder()
    if spec.startswith("http"):
        return Embedder(url=spec)
    raise ValueError(f"unknown encoder {spec!r}: use mlx, vllm or an http /v1/embeddings URL")
