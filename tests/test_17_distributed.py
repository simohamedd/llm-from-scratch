"""Data-parallel training (llm/distributed.py): 2 CPU processes over gloo, tiny models."""
import os
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import TEXT, tiny_tok  # noqa: E402
from llm.config import ModelConfig  # noqa: E402
from llm.data import TokenDataset  # noqa: E402
from llm.distributed import (DistEnv, GradReducer, all_reduce_mean, broadcast_parameters,  # noqa: E402
                             setup_distributed, sync_buffers)
from llm.model import GPT  # noqa: E402
from llm.trainer import TrainConfig, train  # noqa: E402

VOCAB, WORLD = 64, 2

pytestmark = pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    base = dict(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, context_length=32)
    return GPT(ModelConfig(**{**base, **kw}))


def _global_batch(B=4, T=24):
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, VOCAB, (B, T), generator=g), torch.randint(0, VOCAB, (B, T), generator=g)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(worker, *args):
    port = _free_port()
    mp.spawn(_entry, args=(worker, port, args), nprocs=WORLD, join=True)


def _entry(rank, worker, port, args):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD))
    torch.set_num_threads(1)
    env = setup_distributed("gloo", device="cpu")
    try:
        worker(env, *args)
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------- workers (run inside each rank)

def _w_grads_match_single_process(env: DistEnv, out_dir, bucket_mb, kw):
    model = _model(seed=env.rank, **kw).train()     # different init per rank: broadcast must fix it
    broadcast_parameters(model)
    x, y = _global_batch()
    B = x.shape[0] // env.world_size
    xs, ys = x[env.rank * B:(env.rank + 1) * B], y[env.rank * B:(env.rank + 1) * B]
    _, loss = model(xs, ys)
    loss.backward()
    reducer = GradReducer(model, bucket_mb=bucket_mb)
    reducer.all_reduce()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    torch.save({"grads": grads, "calls": reducer.calls, "buckets": len(reducer.buckets)},
               Path(out_dir) / f"rank{env.rank}.pt")


def _w_matches_torch_ddp(env: DistEnv, out_dir):
    x, y = _global_batch()
    B = x.shape[0] // env.world_size
    xs, ys = x[env.rank * B:(env.rank + 1) * B], y[env.rank * B:(env.rank + 1) * B]
    ours = _model().train()
    broadcast_parameters(ours)
    _, loss = ours(xs, ys)
    loss.backward()
    GradReducer(ours).all_reduce()
    ref = torch.nn.parallel.DistributedDataParallel(_model().train())
    _, loss = ref(xs, ys)
    loss.backward()
    g_ours = {n: p.grad for n, p in ours.named_parameters()}
    g_ref = {n: p.grad for n, p in ref.module.named_parameters()}
    for n in g_ours:
        assert torch.allclose(g_ours[n], g_ref[n], atol=1e-6, rtol=1e-5), n
    torch.save(True, Path(out_dir) / f"ok{env.rank}.pt")


def _w_train(env: DistEnv, out_dir, kw):
    tok = tiny_tok()
    ds = TokenDataset(torch.tensor(tok.encode(TEXT)), context_length=32)
    model = _model(seed=env.rank, **{**kw, "vocab_size": tok.vocab_size})
    logs = []
    tc = TrainConfig(max_steps=30, batch_size=4, grad_accum=2, lr=3e-3, warmup_steps=3, eval_every=15,
                     eval_iters=2, out_dir=str(Path(out_dir) / "ckpt"))
    train(model, ds, tc, device="cpu", log=logs.append)
    torch.save({"state": model.state_dict(), "logs": logs}, Path(out_dir) / f"rank{env.rank}.pt")


def _w_sync_buffers(env: DistEnv, out_dir):
    m = _model(n_experts=4, moe_balance="bias")
    for name, b in m.named_buffers():
        if name.endswith("expert_bias"):
            b.fill_(float(env.rank))            # ranks disagree: 0 and 1
    sync_buffers(m)
    t = all_reduce_mean(torch.tensor([float(env.rank)]))
    torch.save({"bias": [b.clone() for n, b in m.named_buffers() if n.endswith("expert_bias")], "mean": t},
               Path(out_dir) / f"rank{env.rank}.pt")


# ---------------------------------------------------------------- tests

def _single_process_grads(kw):
    model = _model(seed=0, **kw).train()
    x, y = _global_batch()
    _, loss = model(x, y)
    loss.backward()
    return {n: p.grad for n, p in model.named_parameters() if p.grad is not None}


# (MoE with the aux loss is left out on purpose: its load statistics are computed per rank, so the
#  aux-loss gradient of "mean over ranks" is not the global-batch one — see the doc)
@pytest.mark.parametrize("bucket_mb,kw", [(25.0, {}), (0.001, {}),
                                          (25.0, {"mla_kv_rank": 16, "mla_rope_dim": 4, "qk_norm": False})])
def test_all_reduced_grads_equal_single_process_on_global_batch(tmp_path, bucket_mb, kw):
    _run(_w_grads_match_single_process, str(tmp_path), bucket_mb, kw)
    r0, r1 = (torch.load(tmp_path / f"rank{r}.pt") for r in range(WORLD))
    ref = _single_process_grads(kw)
    assert r0["grads"].keys() == r1["grads"].keys()
    for n, g in ref.items():
        assert torch.equal(r0["grads"][n], r1["grads"][n]), n          # replicas identical
        assert torch.allclose(r0["grads"][n], g, atol=1e-6, rtol=1e-4), n  # = global-batch gradient
    if bucket_mb < 0.01:
        assert r0["buckets"] > 1 and r0["calls"] == r0["buckets"]      # tiny buckets -> many calls
    else:
        assert r0["calls"] == 1                                         # whole tiny model in one bucket


def test_matches_torch_ddp(tmp_path):
    _run(_w_matches_torch_ddp, str(tmp_path))
    assert all((tmp_path / f"ok{r}.pt").exists() for r in range(WORLD))


@pytest.mark.parametrize("kw", [{}, {"n_experts": 4, "moe_balance": "bias"}])
def test_ddp_training_keeps_replicas_identical_and_learns(tmp_path, kw):
    _run(_w_train, str(tmp_path), kw)
    r0, r1 = (torch.load(tmp_path / f"rank{r}.pt") for r in range(WORLD))
    for k in r0["state"]:
        assert torch.equal(r0["state"][k], r1["state"][k]), k
    assert r1["logs"] == []                                   # only rank 0 logs
    assert any("world_size=2" in line for line in r0["logs"])
    vals = [float(line.split("val ")[1].split()[0]) for line in r0["logs"] if line.startswith("step")]
    assert vals[-1] < vals[0]
    assert (tmp_path / "ckpt" / "last.pt").exists()


def test_sync_buffers_and_all_reduce_mean(tmp_path):
    _run(_w_sync_buffers, str(tmp_path))
    r0, r1 = (torch.load(tmp_path / f"rank{r}.pt") for r in range(WORLD))
    assert len(r0["bias"]) == 2
    for a, b in zip(r0["bias"], r1["bias"]):
        assert torch.allclose(a, torch.full_like(a, 0.5)) and torch.equal(a, b)
    assert r0["mean"].item() == 0.5 == r1["mean"].item()


def test_single_process_is_a_no_op(monkeypatch):
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(k, raising=False)
    env = setup_distributed(device="cpu")
    assert not env.enabled and env.is_main and not dist.is_initialized()
    m = _model()
    x, y = _global_batch()
    _, loss = m(x, y)
    loss.backward()
    before = {n: p.grad.clone() for n, p in m.named_parameters()}
    GradReducer(m).all_reduce()                               # nothing to reduce with
    sync_buffers(m)
    assert all(torch.equal(before[n], p.grad) for n, p in m.named_parameters())
    assert all_reduce_mean(torch.tensor([3.0])).item() == 3.0
