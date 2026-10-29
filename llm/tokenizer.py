"""Chapter 1 — A byte-level BPE tokenizer written from scratch.

The same family of algorithm used by GPT-2/3/4, Llama 3 and most modern LLMs:

1. Split text into chunks with a regex (so merges never cross word boundaries).
2. Start from the 256 raw bytes — every string is representable, no <unk> needed.
3. Repeatedly find the most frequent adjacent pair of tokens and merge it into a
   new token, until the vocabulary reaches `vocab_size`.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# GPT-4-style pre-tokenisation pattern (simplified to the stdlib `re` module).
SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+"""


def _pair_counts(ids: list[int], counts: Counter | None = None) -> Counter:
    counts = Counter() if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, special_tokens: list[str] | None = None):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}
        self._pattern = re.compile(SPLIT_PATTERN)
        self._init_special = special_tokens or ["<|bos|>", "<|eos|>", "<|pad|>", "<|user|>", "<|assistant|>"]

    # ------------------------------------------------------------------ training
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        n_special = len(self._init_special)
        n_merges = vocab_size - 256 - n_special
        assert n_merges >= 0, "vocab_size must be >= 256 + number of special tokens"
        chunks = [list(c.encode("utf-8")) for c in self._pattern.findall(text)]
        for k in range(n_merges):
            counts: Counter = Counter()
            for ch in chunks:
                _pair_counts(ch, counts)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            new_id = 256 + k
            chunks = [_merge(ch, pair, new_id) for ch in chunks]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and k % 50 == 0:
                print(f"merge {k}/{n_merges}: {pair} -> {self.vocab[new_id]!r} ({counts[pair]}x)")
        base = 256 + len(self.merges)
        self.special_tokens = {tok: base + i for i, tok in enumerate(self._init_special)}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    def token_id(self, special: str) -> int:
        return self.special_tokens[special]

    # ------------------------------------------------------------------ encode / decode
    def _encode_chunk(self, data: bytes) -> list[int]:
        ids = list(data)
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        if allow_special and self.special_tokens:
            pat = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
            parts = re.split(pat, text)
        else:
            parts = [text]
        ids: list[int] = []
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
                continue
            for chunk in self._pattern.findall(part):
                ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def decode(self, ids: list[int]) -> str:
        inv_special = {v: k for k, v in self.special_tokens.items()}
        out = bytearray()
        for i in ids:
            if i in inv_special:
                out.extend(inv_special[i].encode("utf-8"))
            else:
                out.extend(self.vocab[i])
        return out.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        data = {
            "merges": [[a, b, i] for (a, b), i in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls(special_tokens=list(data["special_tokens"]))
        for a, b, i in data["merges"]:
            tok.merges[(a, b)] = i
            tok.vocab[i] = tok.vocab[a] + tok.vocab[b]
        tok.special_tokens = {k: int(v) for k, v in data["special_tokens"].items()}
        return tok
