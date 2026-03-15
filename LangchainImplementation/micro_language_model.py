"""
Micro (>=1000 parameter) language model for offline demos.

This is intentionally tiny and dependency-light: NumPy only.
Model: byte-level bigram MLP (embedding -> linear -> softmax).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


@dataclass
class MicroByteLM:
    """
    A tiny next-byte language model.

    Parameters
    ----------
    embed_dim:
        Embedding size. With vocab=256, embed_dim=4 yields 2304 parameters:
        256*4 (embedding) + 4*256 (output) + 256 (bias).
    """

    embed_dim: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.vocab_size = 256
        self.emb = (rng.standard_normal((self.vocab_size, self.embed_dim)).astype(np.float32)) * 0.02
        self.W = (rng.standard_normal((self.embed_dim, self.vocab_size)).astype(np.float32)) * 0.02
        self.b = np.zeros((self.vocab_size,), dtype=np.float32)

    def num_parameters(self) -> int:
        return int(self.emb.size + self.W.size + self.b.size)

    def _forward(self, prev_byte: int) -> np.ndarray:
        h = self.emb[prev_byte]  # (D,)
        return h @ self.W + self.b  # (256,)

    def fit(
        self,
        texts: Iterable[str],
        *,
        steps: int = 2000,
        lr: float = 0.15,
        batch_size: int = 64,
        seed: Optional[int] = None,
    ) -> float:
        """
        Train with simple SGD on random (prev_byte -> next_byte) pairs.

        Returns the final average negative log-likelihood over the last batch.
        """
        data = "\n".join(texts).encode("utf-8", errors="replace")
        if len(data) < 2:
            raise ValueError("Need at least 2 bytes of training data.")
        arr = np.frombuffer(data, dtype=np.uint8)

        rng = np.random.default_rng(self.seed if seed is None else seed)
        loss = 0.0

        for _ in range(int(steps)):
            idx = rng.integers(0, len(arr) - 1, size=(int(batch_size),), endpoint=False)
            prev = arr[idx].astype(np.int64)
            nxt = arr[idx + 1].astype(np.int64)

            d_emb = np.zeros_like(self.emb)
            dW = np.zeros_like(self.W)
            db = np.zeros_like(self.b)

            loss = 0.0
            for p, y in zip(prev, nxt):
                logits = self._forward(int(p))
                probs = _softmax(logits)
                loss += -float(np.log(probs[int(y)] + 1e-12))

                # dlogits = probs; dlogits[y] -= 1
                dlogits = probs
                dlogits[int(y)] -= 1.0

                h = self.emb[int(p)]
                dW += np.outer(h, dlogits).astype(np.float32)
                db += dlogits.astype(np.float32)
                d_emb[int(p)] += (self.W @ dlogits).astype(np.float32)

            scale = 1.0 / float(batch_size)
            self.W -= (lr * scale) * dW
            self.b -= (lr * scale) * db
            self.emb -= (lr * scale) * d_emb

            loss *= scale

        return float(loss)

    def generate(
        self,
        prompt: str,
        *,
        max_new_bytes: int = 200,
        temperature: float = 0.9,
        seed: Optional[int] = None,
    ) -> str:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        rng = np.random.default_rng(self.seed if seed is None else seed)
        out = bytearray(prompt.encode("utf-8", errors="replace"))
        if not out:
            out.append(10)  # newline as a reasonable start token

        for _ in range(int(max_new_bytes)):
            prev = int(out[-1])
            logits = self._forward(prev).astype(np.float64)
            logits = logits / float(temperature)
            probs = _softmax(logits)
            nxt = int(rng.choice(self.vocab_size, p=probs))
            out.append(nxt)

        return out.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez_compressed(
            path,
            embed_dim=np.array([self.embed_dim], dtype=np.int32),
            emb=self.emb,
            W=self.W,
            b=self.b,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MicroByteLM":
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            embed_dim = int(z["embed_dim"][0])
            obj = cls(embed_dim=embed_dim)
            obj.emb = z["emb"].astype(np.float32)
            obj.W = z["W"].astype(np.float32)
            obj.b = z["b"].astype(np.float32)
        return obj


class MicroLMTextGenerator:
    """
    Small adapter that mimics the HuggingFace pipeline return shape used in `agents.py`.
    """

    def __init__(
        self,
        *,
        weights_path: Optional[str | Path] = None,
        train_texts: Optional[Iterable[str]] = None,
        steps: int = 1200,
        lr: float = 0.15,
        seed: int = 0,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self.model = None
        self.seed = seed

        if self.weights_path and self.weights_path.exists():
            self.model = MicroByteLM.load(self.weights_path)
        else:
            self.model = MicroByteLM(seed=seed)
            if train_texts is not None:
                self.model.fit(train_texts, steps=steps, lr=lr, seed=seed)
            if self.weights_path:
                self.weights_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(self.weights_path)

    def __call__(self, prompt_text: str, *, max_new_tokens: int = 50, **_: object):
        generated = self.model.generate(
            prompt_text,
            max_new_bytes=int(max_new_tokens),
            temperature=0.9,
            seed=self.seed,
        )
        return [{"generated_text": generated}]
