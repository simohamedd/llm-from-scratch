# Chapter 1 — Tokenization

**Code:** [`llm/tokenizer.py`](../llm/tokenizer.py) · **Branch:** `chapter/01-tokenizer` · **Test:** `tests/test_01_tokenizer.py`

A neural network only sees numbers. A tokenizer maps text ↔ integer ids.

## Why byte-level BPE?
| Approach | Problem |
|---|---|
| Characters | sequences become very long → expensive attention |
| Whole words | huge vocabulary, unknown words break it |
| **Byte-level BPE** | starts from 256 bytes (any text, any language, emoji) and learns frequent chunks |

## The algorithm
1. **Pre-split** text with a regex so merges never cross word/number/punctuation boundaries.
2. Represent each chunk as UTF-8 bytes (ids 0-255).
3. Count every adjacent pair, merge the most frequent into a new id, repeat `vocab_size - 256` times.
4. **Encode** new text by replaying merges in the order they were learned.
5. Add **special tokens** (`<|bos|>`, `<|eos|>`, `<|user|>`, `<|assistant|>`, `<|pad|>`) used later for chat.

## Try it
```bash
python scripts/prepare_data.py --vocab-size 1024
```
```python
from llm.tokenizer import BPETokenizer
tok = BPETokenizer.load("checkpoints/tokenizer.json")
ids = tok.encode("To be, or not to be")
print(ids, [tok.decode([i]) for i in ids])
```

## Exercises
- Plot compression (bytes/token) vs `vocab_size`.
- Train on Arabic or French text — how does compression change?
- Compare with `tiktoken` (GPT-4's tokenizer) on the same sentence.

**Real-world equivalents:** `tiktoken`, Hugging Face `tokenizers`, SentencePiece.
