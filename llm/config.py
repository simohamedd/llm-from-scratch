"""Model hyper-parameters. Presets range from 'runs on a laptop CPU' to 'needs a GPU'."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 1024
    context_length: int = 256
    d_model: int = 256          # embedding / residual width
    n_layers: int = 4
    n_heads: int = 8            # query heads
    n_kv_heads: int = 4         # key/value heads (< n_heads = Grouped-Query Attention)
    ffn_mult: float = 8 / 3     # SwiGLU hidden size multiplier (Llama convention)
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    # Mixture of Experts (0 = dense SwiGLU). See llm/moe.py
    n_experts: int = 0
    moe_top_k: int = 2
    moe_aux_weight: float = 0.01

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    # ~0.9M params — trains in a few minutes on CPU, good for learning
    "nano": ModelConfig(d_model=128, n_layers=4, n_heads=4, n_kv_heads=2, context_length=128),
    # ~4.7M params — a solid laptop / free Colab GPU run
    "micro": ModelConfig(d_model=256, n_layers=6, n_heads=8, n_kv_heads=4, context_length=256),
    # ~27M params — needs a GPU
    "small": ModelConfig(vocab_size=8192, d_model=512, n_layers=8, n_heads=8, n_kv_heads=4, context_length=512),
    # ~100M params — GPT-2 small scale
    "base": ModelConfig(vocab_size=32000, d_model=768, n_layers=12, n_heads=12, n_kv_heads=4, context_length=1024),
}
