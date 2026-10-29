import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEXT = "The quick brown fox jumps over the lazy dog. " * 60 + "Ça va? 你好! 🙂 123456"


def tiny_tok():
    from llm.tokenizer import BPETokenizer
    t = BPETokenizer()
    t.train(TEXT, vocab_size=300)
    return t


def tiny_model(vocab):
    from llm.config import ModelConfig
    from llm.model import GPT
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=vocab, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, context_length=32))
