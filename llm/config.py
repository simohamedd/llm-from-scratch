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
    moe_shared_experts: int = 0  # always-on experts added to the routed ones (DeepSeek-MoE: 1-2)
    moe_balance: str = "aux"    # "aux" = Switch load-balancing loss, "bias" = aux-loss-free (DeepSeek-V3)
    moe_bias_rate: float = 1e-3  # bias update step for moe_balance="bias" (DeepSeek-V3: 0.001)
    # Training stability (off by default). See llm/stability.py
    qk_norm: bool = False       # RMSNorm on each query/key head (OLMo 2, Gemma 3, Qwen3)
    attn_softcap: float = 0.0   # cap*tanh(score/cap) on attention scores, e.g. 50 (Gemma 2)
    final_softcap: float = 0.0  # same on the output vocabulary logits, e.g. 30 (Gemma 2)
    # Sliding-window attention (0 = full attention). See llm/sliding_window.py
    sliding_window: int = 0     # each token sees only the last N tokens (Mistral 7B: 4096)
    global_every: int = 0       # every N-th layer stays global (Gemma 2: 2, Gemma 3: 6); 0 = none
    # Multi-head Latent Attention (0 = standard GQA). See llm/mla.py
    mla_kv_rank: int = 0        # size of the cached latent c_kv (DeepSeek-V3: 512)
    mla_rope_dim: int = 16      # decoupled RoPE key/query dims, shared key (DeepSeek-V3: 64)
    mla_q_rank: int = 0         # low-rank query bottleneck c_q; 0 = full-rank wq (DeepSeek-V3: 1536)
    # RoPE scaling for longer context ("" = off). See llm/rope_scaling.py
    rope_scaling: str = ""      # "linear" (Position Interpolation), "ntk" (NTK-aware) or "yarn"
    rope_factor: float = 1.0    # new length / original length, e.g. 4 (Qwen2.5 YaRN: 4, DeepSeek-V3: 40)
    rope_original_context: int = 0  # length RoPE was trained at; 0 = context_length
    # Gradient checkpointing (off by default): recompute block activations in backward. See llm/checkpointing.py
    grad_checkpoint: bool = False   # trade ~30 % more compute for much less activation memory
    checkpoint_every: int = 1       # checkpoint every N-th block (1 = all, 2 = half of them, ...)
    # Multi-token prediction auxiliary heads (0 = off). See llm/mtp.py
    mtp_depth: int = 0          # extra future tokens predicted per position (DeepSeek-V3: 1)
    mtp_weight: float = 0.3     # λ: loss = main + λ · mean MTP loss (DeepSeek-V3: 0.3, later 0.1)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def rope_dim(self) -> int:
        """Dimensions RoPE rotates: a full head, or only the decoupled part under MLA."""
        return self.mla_rope_dim if self.mla_kv_rank > 0 else self.head_dim

    @property
    def max_seq_len(self) -> int:
        """Longest sequence the model should see: context_length, or original × factor when scaled."""
        if not self.rope_scaling:
            return self.context_length
        original = self.rope_original_context or self.context_length
        return max(self.context_length, int(original * self.rope_factor))

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
