"""Data-parallel training across processes / GPUs, built from the collective ops up.

The idea (what `torch.nn.parallel.DistributedDataParallel` does, and what every big LLM run
starts from):

    1. launch N identical processes (`torchrun --nproc_per_node N ...`), one per GPU;
       torchrun tells each one its RANK, LOCAL_RANK and WORLD_SIZE through env variables
    2. every process holds a full copy of the model; rank 0's weights are broadcast so all
       copies start bit-identical
    3. every process draws a *different* mini-batch and runs forward + backward on it
    4. the gradients are averaged across processes with one collective, all-reduce:

           g  =  (g_0 + g_1 + ... + g_{N-1}) / N

       With equal per-rank batches, that average is exactly the gradient of the mean loss over
       the concatenated global batch of N·B sequences
    5. every process runs the same optimizer step on the same gradient -> the copies stay
       identical without ever sending weights again

So N processes with batch B behave like one process with batch N·B, at ~N× the throughput.

Two implementation details that matter for speed:
  * **bucketing** — all-reduce has a fixed latency per call, so instead of one call per
    parameter we flatten many gradients into ~25 MB buckets and reduce each bucket once
  * **gradient accumulation** — only reduce after the *last* micro-batch (DDP's `no_sync()`);
    the trainer does this naturally by calling `reducer.all_reduce()` once per optimizer step

Real DDP additionally overlaps communication with the backward pass (a bucket is reduced as
soon as its gradients are ready). We keep the reduce after backward for readability — see the
exercises in docs/advanced/distributed-training.md.

Persistent buffers that change during training (the MoE `expert_bias` for aux-loss-free
balancing) are updated from each rank's *local* load, so they are averaged after every step to
keep the copies identical (`sync_buffers`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistEnv:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: str = "cpu"
    backend: str = ""

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def setup_distributed(backend: str | None = None, device: str | None = None) -> DistEnv:
    """Read torchrun's env vars and join the process group. A no-op single-process run otherwise.

    NCCL on CUDA (one GPU per process, picked by LOCAL_RANK), gloo on CPU.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    if device is None:
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if world_size == 1:
        return DistEnv(device=device)
    backend = backend or ("nccl" if device.startswith("cuda") else "gloo")
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DistEnv(rank, local_rank, world_size, device, backend)


def current_env() -> DistEnv:
    """The DistEnv of an already-initialised process group (single-process if none)."""
    if not (dist.is_available() and dist.is_initialized()):
        return DistEnv()
    return DistEnv(dist.get_rank(), int(os.environ.get("LOCAL_RANK", "0")), dist.get_world_size(),
                   backend=dist.get_backend())


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def broadcast_parameters(model: torch.nn.Module, src: int = 0):
    """Copy rank `src`'s parameters and buffers to every rank so all replicas start identical."""
    for t in list(model.parameters()) + list(model.buffers()):
        dist.broadcast(t.data, src=src)


@torch.no_grad()
def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """Average a tensor over all ranks (in place) and return it."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return t


@torch.no_grad()
def sync_buffers(model: torch.nn.Module):
    """Average floating-point persistent buffers (e.g. MoE `expert_bias`) across ranks."""
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return
    persistent = set(model.state_dict().keys())
    for name, b in model.named_buffers():
        if name in persistent and b.is_floating_point():
            all_reduce_mean(b)


class GradReducer:
    """Average `.grad` of every trainable parameter across ranks, in flat buckets.

    reducer = GradReducer(model)       # once, after the model is on its device
    loss.backward()                    # (possibly several micro-batches)
    reducer.all_reduce()               # before clipping and opt.step()
    """

    def __init__(self, model: torch.nn.Module, bucket_mb: float = 25.0):
        self.params = [p for p in model.parameters() if p.requires_grad]
        cap = max(1, int(bucket_mb * 2**20))
        self.buckets: list[list[torch.nn.Parameter]] = []
        cur, size = [], 0
        # reverse order ~ the order gradients become ready in backward (what DDP does too)
        for p in reversed(self.params):
            nbytes = p.numel() * p.element_size()
            if cur and size + nbytes > cap:
                self.buckets.append(cur)
                cur, size = [], 0
            cur.append(p)
            size += nbytes
        if cur:
            self.buckets.append(cur)
        self.calls = 0                      # number of all-reduce calls issued (for tests / curiosity)

    @torch.no_grad()
    def all_reduce(self):
        if not (dist.is_available() and dist.is_initialized()):
            return
        world = dist.get_world_size()
        if world == 1:
            return
        for bucket in self.buckets:
            # a parameter that got no gradient on this rank still takes part, with zeros,
            # so every rank issues the same collectives in the same order
            grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in bucket]
            flat = torch.cat([g.reshape(-1) for g in grads])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat /= world
            self.calls += 1
            offset = 0
            for p in bucket:
                n = p.numel()
                chunk = flat[offset:offset + n].view_as(p)
                if p.grad is None:
                    if chunk.any():             # unused on every rank -> leave it None (no decay step)
                        p.grad = chunk.clone()
                else:
                    p.grad.copy_(chunk)
                offset += n
