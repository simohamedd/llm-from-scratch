# Advanced — Multi-GPU data-parallel training (`torchrun`)

Code: [`llm/distributed.py`](../../llm/distributed.py) · used by [`llm/trainer.py`](../../llm/trainer.py) when a process
group is initialised · launch: `torchrun --nproc_per_node N scripts/train.py ...` ·
tests: [`tests/test_17_distributed.py`](../../tests/test_17_distributed.py)

## The idea

Data parallelism is the first thing every large training run does: put a full copy of the model on
each of N GPUs, give every copy a **different slice of the batch**, and average the gradients before
the optimizer step.

```
           rank 0                 rank 1                ...        rank N-1
  batch    x_0 (B seqs)           x_1 (B seqs)                     x_{N-1}
  fwd+bwd  g_0                    g_1                              g_{N-1}
              \_______________ all-reduce (sum) / N _______________/
  step     θ ← opt(θ, ḡ)          θ ← opt(θ, ḡ)                    θ ← opt(θ, ḡ)
```

Because the loss is a mean over tokens and every rank uses the same batch size,

    ḡ = (1/N) Σ_r ∇ mean(loss on x_r) = ∇ mean(loss on x_0 ∪ x_1 ∪ ... ∪ x_{N-1})

— the averaged gradient is **exactly** the gradient of one big batch of N·B sequences. Every rank
then applies the same optimizer step to the same gradient, so the replicas stay identical without
ever sending weights again (they only need to start identical: rank 0's weights are broadcast once).

Result: N processes with per-rank batch B train like one process with batch N·B, at up to N× the
throughput. The price is one all-reduce of the full gradient per optimizer step.

## How it works

`torchrun` starts N copies of the script and gives each its identity through environment variables
(`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`). `setup_distributed()` reads them
and joins the process group — NCCL on CUDA (one GPU per process, picked by `LOCAL_RANK`), gloo on CPU.
Without torchrun it returns a single-process `DistEnv` and nothing changes.

Inside `train()` (only when a process group exists):

| piece | what it does |
|---|---|
| `broadcast_parameters(model)` | copy rank 0's parameters + buffers to everyone |
| `torch.manual_seed(seed + rank)` | each rank samples **different** batches (rank 0 = the old single-process seed) |
| `GradReducer.all_reduce()` | after the last micro-batch: flatten gradients into ~25 MB buckets, `all_reduce(SUM)`, divide by N, copy back |
| `sync_buffers(model)` | average persistent float buffers after each step (MoE `expert_bias` is updated from the *local* load) |
| `estimate_loss` | each rank evaluates its own batches, the means are averaged → N× more eval samples |
| rank 0 only | logs and writes `best.pt` / `last.pt` |

**Why buckets?** An all-reduce has a fixed latency per call. One call per parameter (dozens of small
norm vectors) wastes most of the time on latency; one call for everything can't start until backward is
completely done. DDP's compromise — buckets of ~25 MB, filled in reverse parameter order (≈ the order
gradients become ready) — is what `GradReducer` does. Our nano model fits in a single bucket.

**Gradient accumulation** composes naturally: the reducer runs once per optimizer step, after the last
micro-batch, which is what `DistributedDataParallel.no_sync()` achieves. Global batch =
`batch_size × grad_accum × world_size`.

The tests prove the core claim directly: with 2 gloo processes, each on half of a batch, the reduced
gradients equal the single-process gradients on the full batch (dense, tiny buckets → several calls,
MLA), equal what `torch.nn.parallel.DistributedDataParallel` produces, and a 30-step `train()` ends with
bit-identical replicas.

**One honest caveat — MoE aux loss.** The load-balancing loss uses load statistics of the tokens it
sees; under data parallelism each rank computes them over its own B sequences, so the averaged aux
gradient is *not* the global-batch one (it's a per-micro-batch balance loss — which is what most real
MoE runs use too; some all-reduce the load statistics instead). The same holds for aux-loss-free bias
balancing: each rank steps its bias from its local load and `sync_buffers` averages the result.

## How to run

```bash
python scripts/prepare_data.py --vocab-size 512 --tokenizer-sample 100000   # once, before torchrun

# 2 processes on CPU (gloo) — same global batch as a single run with --batch-size 32
torchrun --nproc_per_node 2 scripts/train.py --preset nano --steps 400 --batch-size 16

# one machine, 8 GPUs (NCCL picked automatically)
torchrun --nproc_per_node 8 scripts/train.py --preset micro --steps 5000 --batch-size 32
```

`--batch-size` is **per rank**. Run `prepare_data.py` first so the token cache exists (otherwise every
rank tokenises the corpus and races to write the cache file).

## Measured result

Tiny Shakespeare, nano preset (vocab 512), 400 steps, AdamW 3e-3, cosine, a 2-core CPU container.
"Single" is the plain script (one process, PyTorch uses both cores for intra-op threads); DDP is
`torchrun --nproc_per_node 2` with gloo (torchrun sets one thread per process).

| run | processes × per-rank batch | global batch | val loss @200 | val loss @400 | training-loop time |
|---|---|---|---|---|---|
| single, seed 1337 | 1 × 32 | 32 | 3.114 | 2.851 | 101 s |
| single, seed 1338 | 1 × 32 | 32 | 3.146 | 2.849 | 99 s |
| DDP | 2 × 16 | 32 | 3.104 | 2.801 | 99 s |
| DDP | 2 × 32 | 64 | 2.958 | **2.729** | 195 s |

What this shows, honestly:

- **Same global batch, same learning curve.** 2 × 16 tracks the single 32-batch runs (step 200 within
  the seed spread; at step 400 it landed 0.05 lower on this one run — different batches are sampled, and
  DDP's eval averages 2× as many batches). The exact-equivalence claim is what the tests check, bit for bit.
- **No speed-up here, by construction.** The container has 2 cores; the single process already uses both
  through intra-op threading, so two 1-thread processes do the same work in the same ~100 s. Data
  parallelism pays off when each rank owns its own device (a GPU) — then 2 × 16 takes about the time of
  1 × 16, i.e. ~2× throughput minus the all-reduce.
- **Bigger global batch = better model per step, not per second** (on this hardware): 2 × 32 reaches
  2.729 but costs 2× the compute. On N GPUs that bigger batch would come for free in wall-clock time.

## Exercises

1. **Overlap communication with backward.** Register `p.register_post_accumulate_grad_hook` on each
   parameter; when every gradient in a bucket is ready, launch `dist.all_reduce(flat, async_op=True)` and
   `wait()` all handles before `opt.step()`. Measure on 2 GPUs how much of the all-reduce disappears.
2. **Global MoE balance.** All-reduce `last_load` across ranks before `update_bias()` so every rank takes
   the same bias step, and check the replicas stay identical *without* `sync_buffers`.
3. **Shard the data instead of random sampling.** Give each rank a disjoint, deterministic slice of the
   token stream (what `DistributedSampler` does) and compare the curves.
4. **Gradient compression.** All-reduce the gradients in bf16 (cast the flat bucket, reduce, cast back).
   How much does the val loss move?
5. **Next step: ZeRO / FSDP.** Every rank still stores the full optimizer state. Shard AdamW's `exp_avg` /
   `exp_avg_sq` across ranks (ZeRO-1): each rank updates 1/N of the parameters, then `all_gather`s them.

## Who uses it

Every large model is trained with data parallelism as the outermost layer: GPT-3, Llama 1-3, OLMo 2,
DeepSeek-V3 (combined with tensor/pipeline/expert parallelism and ZeRO-style sharding such as PyTorch FSDP
or DeepSpeed). Small open training repos — nanoGPT, nanochat, the SmolLM recipes — use exactly this
`torchrun` + all-reduce setup on a single 8-GPU node.
