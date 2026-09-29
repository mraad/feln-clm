"""Qwen3-8B encoders behind CLM's ``Embedder`` interface (its LRU cache and batching).

Both follow CLM's training recipe (``CLM/train/embed_utils.py``, vLLM ``--runner pooling``):
text tokenized without special tokens, tail kept, final-norm hidden state of the last
token, L2-normalised. ``mlx`` runs on Apple silicon; ``vllm`` in-process on CUDA (training
host); an http URL uses a running ``vllm serve ... --runner pooling`` endpoint.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from clm.embedder import Embedder, l2

from .grammar import PREFIX_END

MODEL = "Qwen/Qwen3-8B"


class MLXEmbedder(Embedder):
    """Misses arrive in one call and run length-sorted in batches of at most ``tokens``
    padded tokens (right padding is causal, so a text's last-token state does not depend on
    its batch). Texts of the ``prefix`` framing share a request-independent head: its KV
    cache is computed once, kept (LRU, ``prefixes``), and only the tail is encoded."""

    def __init__(self, model: str = MODEL, max_tokens: int = 2048, tokens: int = 8192,
                 prefixes: int = 96):  # fmt: skip
        super().__init__(url="mlx://" + model, model=model, max_tokens=max_tokens, batch=1 << 16)
        from mlx_lm import load

        self.lm, self.tok = load(model)
        self.tokens = tokens
        self.kv: OrderedDict[str, tuple[list, int]] = OrderedDict()
        self.max_prefixes = prefixes

    def _ids(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False) or [220]

    def _fetch(self, texts: list[str]) -> tuple[list[np.ndarray], int]:
        cap = (self.max_tokens or 2048) - 1
        groups: dict[str, list[tuple[int, list[int]]]] = {}  # prefix ("" = none) -> (index, ids)
        spent = 0
        for i, t in enumerate(texts):
            cut = t.find(PREFIX_END)
            head, tail = (t[: cut + 2], t[cut + 2 :]) if cut > 0 else ("", t)
            ids = self._ids(tail)
            if head and len(self._ids(head)) + len(ids) > cap:  # would be truncated: no reuse
                head, ids = "", self._ids(t)[-cap:]
            groups.setdefault(head, []).append((i, ids))
            spent += len(ids)
        out: list[np.ndarray] = [None] * len(texts)  # type: ignore[list-item]
        for head, items in groups.items():
            kv = self._prefix(head) if head else None
            items.sort(key=lambda x: len(x[1]))
            batch: list[tuple[int, list[int]]] = []
            for item in items + [None]:
                if batch and (item is None or len(item[1]) * (len(batch) + 1) > self.tokens):
                    for (j, _), v in zip(batch, self._forward([ids for _, ids in batch], kv)):
                        out[j] = v
                    batch = []
                if item is not None:
                    batch.append(item)
        return out, spent

    def _prefix(self, head: str) -> tuple[list, int]:
        """Per-layer (keys, values) of ``head``, encoded once and kept LRU."""
        from mlx_lm.models.cache import make_prompt_cache

        if head in self.kv:
            self.kv.move_to_end(head)
            return self.kv[head]
        import mlx.core as mx

        ids = self._ids(head)
        cache = make_prompt_cache(self.lm)
        self.lm.model(mx.array([ids]), cache=cache)
        entry = (
            [(c.keys[..., : c.offset, :], c.values[..., : c.offset, :]) for c in cache],
            len(ids),
        )
        mx.eval([a for kv in entry[0] for a in kv])
        self.kv[head] = entry
        while len(self.kv) > self.max_prefixes:
            self.kv.popitem(last=False)
        return entry

    def _forward(self, ids: list[list[int]], kv: tuple[list, int] | None = None) -> np.ndarray:
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        pad = np.zeros((len(ids), max(map(len, ids))), dtype=np.int32)  # right pad: causal, unseen
        for i, row in enumerate(ids):
            pad[i, : len(row)] = row
        cache = None
        if kv is not None:
            cache = []
            for k, v in kv[0]:
                c = KVCache()
                c.keys, c.values = mx.repeat(k, len(ids), axis=0), mx.repeat(v, len(ids), axis=0)
                c.offset = kv[1]
                cache.append(c)
        hidden = self.lm.model(mx.array(pad), cache=cache)  # [B, T, H] after the final norm
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
