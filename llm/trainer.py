"""Chapter 3b — The pre-training loop.

Techniques used by real LLM training runs, in miniature:
  * AdamW with decoupled weight decay
  * linear warm-up + cosine learning-rate decay (or Warmup-Stable-Decay, llm/schedules.py)
  * gradient accumulation (simulate big batches on small hardware)
  * gradient clipping
  * mixed precision (bf16) on GPU, torch.compile when available
  * periodic validation + checkpointing
  * data-parallel training over several processes / GPUs with `torchrun` (llm/distributed.py)
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .config import ModelConfig
from .distributed import GradReducer, all_reduce_mean, broadcast_parameters, current_env, sync_buffers
from .model import GPT
from .schedules import SCHEDULES, cosine_lr, wsd_lr


@dataclass
class TrainConfig:
    max_steps: int = 2000
    batch_size: int = 32
    grad_accum: int = 1
    lr: float = 3e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 200
    eval_iters: int = 20
    out_dir: str = "checkpoints"
    compile: bool = False
    optimizer: str = "adamw"     # or "muon" (Muon on hidden matrices, AdamW on the rest; llm/muon.py)
    seed: int = 1337
    schedule: str = "cosine"     # or "wsd" (warm-up, constant, short final decay; llm/schedules.py)
    decay_frac: float = 0.2      # wsd: fraction of max_steps spent decaying
    decay_shape: str = "sqrt"    # wsd: "linear" | "cosine" | "sqrt" (1-sqrt)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lr_at(step: int, tc: TrainConfig) -> float:
    if tc.schedule == "cosine":
        return cosine_lr(step, tc.lr, tc.warmup_steps, tc.max_steps, tc.min_lr_ratio)
    if tc.schedule == "wsd":
        return wsd_lr(step, tc.lr, tc.warmup_steps, tc.max_steps, tc.decay_frac, tc.min_lr_ratio, tc.decay_shape)
    raise ValueError(f"unknown schedule {tc.schedule!r}; choose from {SCHEDULES}")


def mtp_log(model) -> str:
    """' | mtp 3.12 3.40' — per-depth multi-token-prediction train losses of the last step (llm/mtp.py)."""
    raw = getattr(model, "_orig_mod", model)
    losses = getattr(raw, "last_mtp_losses", [])
    return " | mtp " + " ".join(f"{m.item():.3f}" for m in losses) if losses else ""


@torch.no_grad()
def estimate_loss(model, dataset, tc: TrainConfig, device, ctx) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(tc.eval_iters)
        for k in range(tc.eval_iters):
            x, y = dataset.batch(split, tc.batch_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = all_reduce_mean(losses.mean()).item()   # average over ranks (no-op single-process)
    model.train()
    return out


def save_checkpoint(path: Path, model: GPT, mcfg: ModelConfig, tc: TrainConfig, step: int, val_loss: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = getattr(model, "_orig_mod", model)
    torch.save({"model": raw.state_dict(), "model_config": mcfg.to_dict(), "train_config": asdict(tc),
                "step": step, "val_loss": val_loss}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> GPT:
    ckpt = torch.load(path, map_location=device)
    model = GPT(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device)


def train(model: GPT, dataset, tc: TrainConfig, device: str | None = None, log=print) -> GPT:
    """Train `model`. Inside an initialised process group (torchrun), runs data-parallel:
    rank 0's weights are broadcast, each rank samples its own batches (seed + rank), gradients are
    averaged with a bucketed all-reduce before clipping, and only rank 0 logs / saves checkpoints.
    `tc.batch_size` is then per rank: global batch = batch_size * grad_accum * world_size."""
    env = current_env()
    torch.manual_seed(tc.seed + env.rank)        # rank 0 = the single-process seed
    device = device or pick_device()
    model.to(device).train()
    reducer = None
    if env.enabled:
        broadcast_parameters(model)
        reducer = GradReducer(model)
    if not env.is_main:
        log = lambda *a, **k: None  # noqa: E731
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cpu", enabled=False)
    opt = model.configure_optimizer(tc.lr, tc.weight_decay, optimizer=tc.optimizer)
    run = torch.compile(model) if tc.compile and hasattr(torch, "compile") else model
    best_val, out = float("inf"), Path(tc.out_dir)
    log(f"device={device} params={model.num_params()/1e6:.2f}M bf16={use_bf16} optimizer={tc.optimizer} schedule={tc.schedule}")
    if env.enabled:
        log(f"data-parallel: world_size={env.world_size} backend={env.backend} "
            f"global batch={tc.batch_size * tc.grad_accum * env.world_size}")
    t0 = time.time()
    for step in range(tc.max_steps + 1):
        lr = lr_at(step, tc)
        for g in opt.param_groups:
            g["lr"] = lr
        if step % tc.eval_every == 0 or step == tc.max_steps:
            losses = estimate_loss(run, dataset, tc, device, ctx)
            log(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"ppl {math.exp(losses['val']):.1f} | lr {lr:.2e} | {time.time()-t0:.0f}s"
                + mtp_log(model))
            if losses["val"] < best_val:
                best_val = losses["val"]
                if env.is_main:
                    save_checkpoint(out / "best.pt", model, model.cfg, tc, step, best_val)
        if step == tc.max_steps:
            break
        opt.zero_grad(set_to_none=True)
        for _ in range(tc.grad_accum):
            x, y = dataset.batch("train", tc.batch_size, device)
            with ctx:
                _, loss = run(x, y)
            (loss / tc.grad_accum).backward()
        if reducer is not None:
            reducer.all_reduce()                 # once per step, after the last micro-batch
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        if env.enabled:
            sync_buffers(model)                  # e.g. MoE expert_bias, updated from local loads
    if env.is_main:
        save_checkpoint(out / "last.pt", model, model.cfg, tc, tc.max_steps, best_val)
    return model
