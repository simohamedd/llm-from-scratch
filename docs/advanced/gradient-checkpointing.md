# Advanced — Gradient checkpointing (activation recomputation)

Code: [`llm/checkpointing.py`](../../llm/checkpointing.py) · flag: `ModelConfig(grad_checkpoint=True, checkpoint_every=1)` ·
tests: [`tests/test_16_grad_checkpoint.py`](../../tests/test_16_grad_checkpoint.py)

## The idea

When you train, the memory is mostly **not the weights** — it's the *activations*. Every op in the
forward pass saves tensors for backward (the input of each `Linear`, the SiLU input, attention
probabilities, RMSNorm statistics, ...). For one Transformer block that's about a dozen tensors of
size `batch × seq_len × d_model` (more for the FFN hidden and the attention), times every layer.
It grows linearly with batch and context — it's what runs out of memory first.

Gradient checkpointing (Chen et al. 2016, *Training Deep Nets with Sublinear Memory Cost*) makes
a trade:

- **forward:** run each checkpointed block normally, but *throw away* everything it saved for
  backward; keep only the block's input.
- **backward:** when backward reaches the block, *re-run its forward* from the kept input to get
  those tensors back, then backprop through it as usual.

| | activations kept | compute per step |
|---|---|---|
| no checkpointing | all blocks' saved tensors | 1 fwd + 1 bwd |
| every block checkpointed | L block inputs + 1 block's tensors (during its backward) | ≈ 2 fwd + 1 bwd (~+33 %) |
| every 2nd block (`checkpoint_every=2`) | about half | ≈ 1.5 fwd + 1 bwd |

The maths of training doesn't change — gradients are **identical**, only memory and time move.
That memory can then buy a bigger batch, a longer context or a bigger model on the same GPU.

## How it works

Autograd lets us intercept every tensor an op saves for backward with
`torch.autograd.graph.saved_tensors_hooks(pack, unpack)`. `checkpoint(fn, *args)` uses it like this:

```python
def pack(t):                 # forward: autograd wants to save t
    return next_ticket()     #   -> give it a small int instead; t can be freed

def unpack(ticket):          # backward: autograd needs the tensor back
    if ticket not in recomputed:
        recompute()          #   -> re-run fn(*args) once, recording each saved tensor in order
    return recomputed.pop(ticket)
```

Because the forward is deterministic, the re-run saves exactly the same tensors in the same order,
so ticket *i* maps to the *i*-th tensor again. Three details make it correct:

1. **Same randomness.** Dropout must draw the same masks twice, so we snapshot the RNG state
   (CPU and CUDA) before the forward and restore it inside `torch.random.fork_rng()` for the re-run.
2. **Same precision.** Backward runs outside `torch.autocast`, so we record the autocast state at
   forward time and re-enter it; otherwise a bf16 run would recompute in fp32 and get different tensors.
3. **No double side effects.** MoE's aux-loss-free balancing updates `expert_bias` after each training
   forward. During the re-run `is_recomputing()` is true: the update is skipped *and* routing uses the
   bias the batch was originally routed with — otherwise different experts would be picked and the
   gradients would be silently wrong (a test catches exactly this).

This is the same design as PyTorch's `torch.utils.checkpoint(..., use_reentrant=False)` (and we test
against it). The older "reentrant" design wraps the block in a custom `autograd.Function` that runs it
under `no_grad`; it's shorter, but it breaks when the block input doesn't need a gradient (LoRA with a
frozen embedding → no gradient reaches the adapters), loses gradients of side outputs (the MoE aux loss)
and doesn't support `torch.autograd.grad()`. Keeping the normal graph and only swapping the saved
tensors avoids all three.

In the model it's one `if` in `GPT.forward`:

```python
for i, block in enumerate(self.blocks):
    if should_checkpoint(self, i, cache):     # enabled, training, grad on, no KV cache
        x = checkpoint(block, x, cos, sin)
    else:
        x = block(x, cos, sin, cache)
```

Eval, `estimate_loss()` (under `no_grad`) and generation never take the checkpoint path.

## Run it

```bash
python scripts/prepare_data.py --vocab-size 512 --tokenizer-sample 100000
python scripts/train.py --preset nano --steps 400                                       # baseline
python scripts/train.py --preset nano --steps 400 --grad-checkpoint                     # every block
python scripts/train.py --preset nano --steps 400 --grad-checkpoint --checkpoint-every 2
```

```python
from llm.checkpointing import count_saved_bytes
with count_saved_bytes() as saved:      # bytes autograd keeps for backward
    _, loss = model(x, y)
print(saved[0] / 2**20, "MiB")
```

## Measured result (CPU, 2 threads)

**Same training, same numbers.** Tiny Shakespeare, nano (0.80M params, vocab 512), 400 steps,
batch 32, AdamW 3e-3, same init seed for all three runs:

| run | val @100 | val @200 | val @300 | val @400 | wall time |
|---|---|---|---|---|---|
| off (default) | 3.4840 | 3.1309 | 2.9560 | 2.8363 | 88.8 s |
| `checkpoint_every=2` | 3.4840 | 3.1309 | 2.9560 | 2.8363 | 103.2 s (+16 %) |
| every block | 3.4840 | 3.1309 | 2.9560 | 2.8363 | 113.3 s (+28 %) |

**Memory.** Peak extra process memory (RSS) for one forward + backward, micro preset (4.7M params, 6 layers,
context 256), measured in a fresh process per row:

| batch | off | every 2nd block | every block |
|---|---|---|---|
| 32 | 1914 MiB | 1250 MiB (−35 %) | 726 MiB (−62 %) |
| 64 | 3858 MiB | 2440 MiB (−37 %) | 1751 MiB (−55 %) |

Bytes autograd keeps for backward (`count_saved_bytes`, everything outside checkpointed blocks; the
kept block inputs are another `L·B·T·d·4` bytes, e.g. 6 MiB for micro B=32):

| config | off | every 2nd | every block |
|---|---|---|---|
| nano, B=32, T=128 | 193.5 MiB | 104.0 MiB | 14.3 MiB |
| micro, B=16, T=256 | 566.5 MiB | 293.5 MiB | 20.6 MiB |
| micro, B=32, T=256 | 1115.5 MiB | 578.1 MiB | 40.7 MiB |

What this shows, honestly:
- **Loss curves are identical to 4 decimals** — it's a pure memory/compute trade, never a quality change.
- **Saved activations drop ~25×** (what's left is the embedding, final norm, LM head and logits), but
  **peak memory only drops 2.2–2.6×**: the peak now comes from the parameter gradients, backward's temporary buffers,
  one block's recomputed activations during its backward, and allocator overhead. Expect a bigger win on
  GPU at long context, where attention and FFN activations dwarf everything else.
- **The time cost is +16–28 % here** (+15–35 % per isolated step, CPU timings are noisy). That's less than
  the theoretical +33 %, because the backward pass is the bigger half of a step.
- On this tiny model it's **not worth turning on** — nano fits easily. It pays when memory is the limit:
  use it to double the batch or context instead of using gradient accumulation.

## Exercises
- Find the largest batch that fits in, say, 2 GiB with and without checkpointing (watch RSS or
  `torch.cuda.max_memory_allocated()` on a GPU). Is a 2× bigger batch at +30 % time faster per token than
  gradient accumulation?
- **Selective checkpointing** (Korthikanti et al. 2022, Megatron-LM): don't recompute everything — keep
  the cheap-to-store, expensive-to-recompute tensors (the matmul outputs) and only drop the attention
  softmax and the SwiGLU hidden. Implement it by checkpointing only `attn` or only `ffn` inside a block.
- `√L` checkpointing: with `L` layers, checkpointing every `√L`-th block gives `O(√L)` memory. Try
  `checkpoint_every` = 1, 2, 3 on a 12-layer model and plot memory vs time.
- **CPU offloading**: instead of dropping a saved tensor in `pack`, copy it to CPU (`t.to("cpu", non_blocking=True)`)
  and copy it back in `unpack` — the same hooks, a different trade (bandwidth instead of compute).
- Combine with `torch.compile` (`--compile`): does the compiled model still checkpoint? (PyTorch's own
  `torch.utils.checkpoint` is understood by the compiler; plain saved-tensor hooks may cause graph breaks.)

## Who uses it
Practically every large training run: Megatron-LM (full and *selective* activation recomputation — used
for GPT-3-scale and Llama-style pre-training), DeepSpeed (`activation_checkpointing`), Hugging Face
`model.gradient_checkpointing_enable()` (the standard switch for LoRA / QLoRA fine-tuning on one GPU),
Unsloth (offloaded checkpointing), and PyTorch FSDP / torchtitan (`activation_checkpoint` with
`full` / `selective` modes) used for Llama 3 training.
