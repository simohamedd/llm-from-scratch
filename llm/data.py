"""Chapter 3a — Data pipeline: raw text -> token ids on disk -> random training batches."""
from __future__ import annotations

from pathlib import Path

import torch


class TokenDataset:
    """Holds one long 1-D tensor of token ids and samples (x, y) windows from it.

    y is x shifted by one position: the model learns to predict the next token.
    """

    def __init__(self, ids: torch.Tensor, context_length: int, val_fraction: float = 0.1):
        n = int(len(ids) * (1 - val_fraction))
        self.splits = {"train": ids[:n], "val": ids[n:]}
        self.T = context_length
        for name, data in self.splits.items():
            assert len(data) > self.T + 1, f"{name} split too small for context_length={self.T}"

    @classmethod
    def from_text_file(cls, path: str | Path, tokenizer, context_length: int, cache: bool = True):
        path = Path(path)
        cache_file = path.with_suffix(".tokens.pt")
        if cache and cache_file.exists():
            ids = torch.load(cache_file)
        else:
            ids = torch.tensor(tokenizer.encode(path.read_text(encoding="utf-8")), dtype=torch.long)
            if cache:
                torch.save(ids, cache_file)
        return cls(ids, context_length)

    def batch(self, split: str, batch_size: int, device: str = "cpu"):
        data = self.splits[split]
        ix = torch.randint(len(data) - self.T - 1, (batch_size,))
        x = torch.stack([data[i:i + self.T] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + self.T] for i in ix])
        return x.to(device), y.to(device)
