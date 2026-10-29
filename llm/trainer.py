"""Chapter 3b — The pre-training loop.

Techniques used by real LLM training runs, in miniature:
  * AdamW with decoupled weight decay
  * linear warm-up + cosine learning-rate decay
  * gradient accumulation (simulate big batches on small hardware)
  * gradient clipping
  * mixed precision (bf16) on GPU, torch.compile when available
  * periodic validation + checkpointing
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .config import ModelConfig
from .model import GPT


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
    seed: int = 1337


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lr_at(step: int, tc: TrainConfig) -> float:
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    progress = (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps)
    min_lr = tc.lr * tc.min_lr_ratio
    return min_lr + 0.5 * (tc.lr - min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


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
        out[split] = losses.mean().item()
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
    torch.manual_seed(tc.seed)
    device = device or pick_device()
    model.to(device).train()
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cpu", enabled=False)
    opt = model.configure_optimizer(tc.lr, tc.weight_decay)
    run = torch.compile(model) if tc.compile and hasattr(torch, "compile") else model
    best_val, out = float("inf"), Path(tc.out_dir)
    log(f"device={device} params={model.num_params()/1e6:.2f}M bf16={use_bf16}")
    t0 = time.time()
    for step in range(tc.max_steps + 1):
        lr = lr_at(step, tc)
        for g in opt.param_groups:
            g["lr"] = lr
        if step % tc.eval_every == 0 or step == tc.max_steps:
            losses = estimate_loss(run, dataset, tc, device, ctx)
            log(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"ppl {math.exp(losses['val']):.1f} | lr {lr:.2e} | {time.time()-t0:.0f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(out / "best.pt", model, model.cfg, tc, step, best_val)
        if step == tc.max_steps:
            break
        opt.zero_grad(set_to_none=True)
        for _ in range(tc.grad_accum):
            x, y = dataset.batch("train", tc.batch_size, device)
            with ctx:
                _, loss = run(x, y)
            (loss / tc.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
    save_checkpoint(out / "last.pt", model, model.cfg, tc, tc.max_steps, best_val)
    return model
