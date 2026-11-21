"""Pre-train a model from scratch.

    python scripts/train.py --preset nano --steps 1500        # CPU, a few minutes
    python scripts/train.py --preset micro --steps 5000       # GPU / Colab
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.config import PRESETS  # noqa: E402
from llm.data import TokenDataset  # noqa: E402
from llm.model import GPT  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402
from llm.trainer import TrainConfig, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="nano", choices=list(PRESETS))
    ap.add_argument("--data", default="data/corpus.txt")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"],
                    help="muon = Muon on hidden matrices + AdamW on the rest (see docs/advanced/muon.md)")
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "wsd"],
                    help="wsd = Warmup-Stable-Decay (see docs/advanced/wsd-schedule.md)")
    ap.add_argument("--decay-frac", type=float, default=0.2, help="wsd: fraction of steps spent decaying")
    ap.add_argument("--decay-shape", default="sqrt", choices=["linear", "cosine", "sqrt"], help="wsd decay curve")
    ap.add_argument("--min-lr-ratio", type=float, default=0.1, help="final LR as a fraction of --lr")
    ap.add_argument("--experts", type=int, default=0, help="Mixture of Experts: number of experts (0 = dense)")
    ap.add_argument("--top-k", type=int, default=2, help="experts used per token")
    ap.add_argument("--shared-experts", type=int, default=0, help="always-on shared experts (DeepSeek-MoE)")
    ap.add_argument("--moe-balance", default="aux", choices=["aux", "bias"],
                    help="aux = load-balancing loss, bias = aux-loss-free (DeepSeek-V3)")
    ap.add_argument("--moe-bias-rate", type=float, default=1e-3, help="bias step for --moe-balance bias")
    ap.add_argument("--qk-norm", action="store_true", help="RMSNorm on query/key heads (OLMo 2 / Qwen3)")
    ap.add_argument("--attn-softcap", type=float, default=0.0, help="soft-cap attention scores, e.g. 50")
    ap.add_argument("--final-softcap", type=float, default=0.0, help="soft-cap output logits, e.g. 30")
    ap.add_argument("--sliding-window", type=int, default=0, help="local attention window, e.g. 32 (Mistral / Gemma)")
    ap.add_argument("--global-every", type=int, default=0, help="every N-th layer keeps full attention (Gemma 2: 2)")
    ap.add_argument("--mla-kv-rank", type=int, default=0, help="Multi-head Latent Attention latent size, e.g. 32 (DeepSeek)")
    ap.add_argument("--mla-rope-dim", type=int, default=16, help="decoupled RoPE dims under MLA")
    ap.add_argument("--mla-q-rank", type=int, default=0, help="low-rank query compression under MLA, e.g. 64 (0 = off)")
    ap.add_argument("--rope-scaling", default="", choices=["", "linear", "ntk", "yarn"],
                    help="RoPE scaling for longer context (see docs/advanced/rope-scaling.md)")
    ap.add_argument("--rope-factor", type=float, default=1.0, help="context extension factor, e.g. 4")
    ap.add_argument("--rope-original-context", type=int, default=0, help="length RoPE was trained at (0 = preset)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute activations in backward to save memory (see docs/advanced/gradient-checkpointing.md)")
    ap.add_argument("--checkpoint-every", type=int, default=1, help="checkpoint every N-th block (with --grad-checkpoint)")
    a = ap.parse_args()

    tok = BPETokenizer.load(a.tokenizer)
    cfg = PRESETS[a.preset]
    cfg.vocab_size = tok.vocab_size
    cfg.n_experts, cfg.moe_top_k = a.experts, a.top_k
    cfg.moe_shared_experts, cfg.moe_balance, cfg.moe_bias_rate = a.shared_experts, a.moe_balance, a.moe_bias_rate
    cfg.qk_norm, cfg.attn_softcap, cfg.final_softcap = a.qk_norm, a.attn_softcap, a.final_softcap
    cfg.sliding_window, cfg.global_every = a.sliding_window, a.global_every
    cfg.mla_kv_rank, cfg.mla_rope_dim, cfg.mla_q_rank = a.mla_kv_rank, a.mla_rope_dim, a.mla_q_rank
    cfg.rope_scaling, cfg.rope_factor, cfg.rope_original_context = a.rope_scaling, a.rope_factor, a.rope_original_context
    cfg.grad_checkpoint, cfg.checkpoint_every = a.grad_checkpoint, a.checkpoint_every
    ds = TokenDataset.from_text_file(a.data, tok, cfg.context_length)
    model = GPT(cfg)
    tc = TrainConfig(max_steps=a.steps, batch_size=a.batch_size, lr=a.lr, compile=a.compile, optimizer=a.optimizer,
                     schedule=a.schedule, decay_frac=a.decay_frac, decay_shape=a.decay_shape, min_lr_ratio=a.min_lr_ratio,
                     eval_every=max(50, a.steps // 10), warmup_steps=min(100, a.steps // 10))
    model = train(model, ds, tc)
    try:  # sampling is built in chapter 4
        from llm.generate import generate
    except ImportError:
        return
    print("\n--- sample ---")
    print(generate(model, tok, "ROMEO:", max_new_tokens=120))


if __name__ == "__main__":
    main()
