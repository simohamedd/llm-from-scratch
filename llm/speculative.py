"""Self-speculative decoding with the multi-token prediction (MTP) heads as the draft.

Plain decoding runs the whole model once per new token. On small batches that forward pass is
dominated by fixed costs (reading the weights, kernel launches, Python), so checking 3 tokens costs
almost the same as producing 1. Speculative decoding exploits that (Leviathan et al. 2023,
Chen et al. 2023): a cheap *draft* guesses the next few tokens, the big model scores all of them in
ONE forward pass, and we keep the longest prefix the big model agrees with — plus one token of its
own for free. The output distribution is exactly the big model's.

A model trained with MTP (llm/mtp.py) already contains a draft: depth k predicts the token k + 1
steps ahead from the trunk's hidden state. DeepSeek-V3 (§5.4.3) uses its single MTP module this way
(85–90 % acceptance, ≈ 1.8× tokens/s). No second model, no second tokenizer, and the draft reuses
the hidden states the main model has already computed.

One round, with committed tokens t_0..t_N where t_N ("pending") was drawn from the main model but is
not in its KV cache yet, and h_{N-1} the trunk state at the last cached position:

    depth 1:  [h_{N-1}  ; Emb(t_N)] → block → head → q_1 → draft d_1   (guess of t_{N+1})
    depth 2:  [h¹_{N-1} ; Emb(d_1)] → block → head → q_2 → draft d_2   (guess of t_{N+2})  ...
    verify:   main model on [t_N, d_1, ..., d_D] in one pass → p_0 .. p_D
    accept d_j with prob min(1, p_{j-1}(d_j) / q_j(d_j))   (greedy: iff d_j == argmax p_{j-1})
    first rejection at j: next pending ~ normalize(max(0, p_{j-1} − q_j)), discard the rest
    all accepted:         next pending ~ p_D  (the bonus token)

Every round commits between 1 and D + 1 tokens for one main-model forward pass.

Each MTP depth is a causal Transformer block, so it needs its own KV cache: depth k at position i
reads Emb(t_{i+k}), and its cache may only keep positions whose inputs are all *committed* tokens
(i ≤ N − k). Positions that used a draft token are computed, used, and truncated away again.
The main cache is truncated the same way after verification (rejected drafts were already fed).

Limits (kept simple on purpose): batch size 1, no repetition penalty, no sliding-window layers
(rolling caches cannot be truncated), and generation stops when the context window is full.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import torch

from .model import GPT, KVCache


@dataclass
class SpecStats:
    """Counters filled in by `mtp_speculative_stream` (pass one in to read them afterwards)."""
    tokens: int = 0          # tokens emitted
    main_calls: int = 0      # forward passes of the main model (prefill + one per round)
    rounds: int = 0          # verification rounds that had at least one draft
    drafted: int = 0         # draft tokens proposed
    accepted: int = 0        # draft tokens accepted
    accepted_at_depth: list[int] = field(default_factory=list)   # accepted count per depth k = 1..D
    drafts: list[tuple[int, list[int]]] = field(default_factory=list)  # (committed length, drafts) per round

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(1, self.drafted)

    @property
    def tokens_per_call(self) -> float:
        """Tokens per main-model forward pass after the prefill (plain decoding: 1.0)."""
        return self.tokens / max(1, self.main_calls - 1)


def filtered_probs(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                   top_p: float | None = None) -> torch.Tensor:
    """The distribution `generate.sample_next` samples from (temperature, top-k, top-p), as probabilities.

    temperature <= 0 gives a one-hot on the argmax (greedy). Speculative sampling must apply the
    same filtering to the draft and to the main model, otherwise p and q are not the distributions
    actually sampled from and the output would no longer match plain decoding.
    """
    if temperature <= 0:
        return torch.zeros_like(logits, dtype=torch.float).scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
    logits = logits.float() / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sp = sorted_logits.softmax(-1)
        sorted_logits = sorted_logits.masked_fill(sp.cumsum(-1) - sp > top_p, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits.softmax(-1)


def accept_or_resample(p: torch.Tensor, q: torch.Tensor, draft: int, greedy: bool) -> tuple[bool, int]:
    """Speculative-sampling step for one draft token. p, q: (V,) target / draft probabilities.

    Returns (accepted, token). If accepted, token == draft. Otherwise token is drawn from the
    residual normalize(max(0, p − q)); overall P(token = x) = p(x) exactly (Leviathan et al., Thm 1):
        q(x)·min(1, p(x)/q(x)) + P(reject)·resid(x) = min(q, p)(x) + (p − q)⁺(x) = p(x).
    Greedy (one-hot p and q): accept iff draft is the argmax, else the argmax is the correction.
    """
    if greedy:
        best = int(p.argmax())
        return best == draft, best
    if torch.rand(()) < torch.clamp(p[draft] / q[draft], max=1.0):
        return True, draft
    resid = torch.clamp(p - q, min=0.0)
    if resid.sum() <= 0:          # p == q up to rounding: any sample from p is correct
        resid = p
    return False, int(torch.multinomial(resid / resid.sum(), 1))


def truncate_cache(cache: KVCache, length: int, layers=None):
    """Forget everything after the first `length` positions (rejected drafts) in the given layers."""
    for layer in (range(len(cache.k)) if layers is None else layers):
        if cache.k[layer] is not None:
            cache.k[layer] = cache.k[layer][:, :, :length]
            cache.v[layer] = cache.v[layer][:, :, :length]
        cache.seen[layer] = min(cache.seen[layer], length)


def trunk(model: GPT, idx: torch.Tensor, cache: KVCache) -> torch.Tensor:
    """Main model up to (not including) the final norm — the hidden state the MTP heads read."""
    start, T = cache.length, idx.shape[1]
    cos, sin = model.rope_cos[start:start + T], model.rope_sin[start:start + T]
    x = model.tok_emb(idx)
    for block in model.blocks:
        x = block(x, cos, sin, cache)
    return x


def _sample(probs: torch.Tensor, greedy: bool) -> int:
    return int(probs.argmax()) if greedy else int(torch.multinomial(probs, 1))


@torch.no_grad()
def mtp_speculative_stream(model: GPT, idx: torch.Tensor, max_new_tokens: int = 100, temperature: float = 0.0,
                           top_k: int | None = None, top_p: float | None = None, n_draft: int | None = None,
                           stop_ids: set[int] | None = None, stats: SpecStats | None = None) -> Iterator[int]:
    """Yield token ids like `generate.stream`, drafting with the model's own MTP heads.

    n_draft: how many MTP depths to use per round (default: all). temperature 0 = greedy, and the
    output is then token-for-token what `stream(..., temperature=0, repetition_penalty=1)` gives.
    """
    if model.mtp is None:
        raise ValueError("model has no MTP heads (train with ModelConfig(mtp_depth >= 1))")
    D = len(model.mtp) if n_draft is None else n_draft
    if not 1 <= D <= len(model.mtp):
        raise ValueError(f"n_draft must be in 1..{len(model.mtp)}")
    if any(b.attn.window for b in model.blocks) or any(m.block.attn.window for m in model.mtp.depths):
        raise ValueError("speculative decoding needs truncatable caches: no sliding-window layers")
    if idx.shape[0] != 1:
        raise ValueError("batch size 1 only")
    model.eval()
    stats = stats if stats is not None else SpecStats()
    stats.accepted_at_depth += [0] * (D - len(stats.accepted_at_depth))
    greedy = temperature <= 0
    probs = lambda logits: filtered_probs(logits, temperature, top_k, top_p)   # noqa: E731
    ctx, L0 = model.cfg.max_seq_len, model.cfg.n_layers
    idx = idx[:, -ctx:]
    device = idx.device

    # ---- prefill: main cache holds the prompt, then draw the first pending token from the main model
    cache = model.new_cache()
    h = trunk(model, idx, cache)
    stats.main_calls += 1
    seq = idx[0].tolist()
    pending = _sample(probs(model.logits(model.norm(h[:, -1]))[0]), greedy)
    seq.append(pending)
    # per depth k: an own KV cache (layer index L0 + k - 1), how many positions it holds, and the
    # hidden states h^k still needed by depth k + 1 as (first position, tensor)
    mcache = KVCache(L0 + len(model.mtp))
    filled = [0] * (D + 1)          # filled[k] for depth k = 1..D (index 0 unused)
    hid: list[tuple[int, torch.Tensor]] = [(0, h)] + [(0, h[:, :0])] * D

    emitted = 0
    while True:
        # ---- emit the pending token (it is already exactly what plain decoding would produce)
        if stop_ids and pending in stop_ids:
            return
        yield pending
        emitted += 1
        stats.tokens += 1
        if emitted >= max_new_tokens:
            return
        L = len(seq)                # committed tokens; seq[L-1] is pending, the main cache holds L-1
        if L > ctx:                 # context full (no sliding re-prefill here, unlike stream())
            return
        d_round = D if L + D <= ctx else 0     # near the end of the context: plain decoding

        # ---- draft: run each depth over the positions it has not cached yet, ending at L - 2
        drafts, qs = [], []
        for k in range(1, d_round + 1):
            mod = model.mtp.depths[k - 1]
            start, h_prev = hid[k - 1]                      # h^{k-1} for positions [start, L-1)
            assert start <= filled[k]
            h_prev = h_prev[:, filled[k] - start:]
            ext = seq + drafts                              # depth k reads token i + k
            toks = torch.tensor([ext[filled[k] + k: L - 1 + k]], device=device)
            cos, sin = model.rope_cos[filled[k]:L - 1], model.rope_sin[filled[k]:L - 1]
            x = mod.proj(torch.cat([mod.hnorm(h_prev), mod.enorm(model.tok_emb(toks))], dim=-1))
            new = mod.block(x, cos, sin, mcache)            # cached attention over this depth's past
            logits = model.logits(mod.norm(new[:, -1]))[0]
            if greedy:                                      # fast path: no distributions needed
                drafts.append(int(logits.argmax()))
            else:
                qs.append(probs(logits))
                drafts.append(_sample(qs[-1], greedy))
            # h^k for [carry start, L-1) = carried states (positions < filled[k]) + the new ones,
            # read by depth k + 1 in the next iteration
            first, carry = hid[k]
            assert first + carry.shape[1] == filled[k]
            hid[k] = (first, torch.cat([carry, new], dim=1))
            # the cache may only keep positions whose inputs were all committed: i <= L-1-k
            filled[k] = max(0, L - k)
            truncate_cache(mcache, filled[k], [L0 + k - 1])
        for k in range(1, d_round + 1):
            # carry h^k only for the committed positions depth k+1 has not cached: [L-k-1, L-k)
            first, full = hid[k]
            s = max(0, L - k - 1)
            hid[k] = (s, full[:, s - first: filled[k] - first])

        # ---- verify: one main forward over [pending, d_1..d_D]
        inp = torch.tensor([[pending] + drafts], device=device)
        h = trunk(model, inp, cache)
        stats.main_calls += 1
        logits = model.logits(model.norm(h))[0]             # (D+1, V): p_j predicts the token after input j
        if greedy:                  # accept the longest prefix equal to the main model's argmax
            best = logits.argmax(-1).tolist()
            a = 0
            while a < len(drafts) and drafts[a] == best[a]:
                a += 1
            pending = best[a]       # the correction at the first mismatch, or the bonus token
        else:
            ps = probs(logits)
            a = 0
            for j, d in enumerate(drafts):
                ok, tok = accept_or_resample(ps[j], qs[j], d, greedy)
                if not ok:
                    pending = tok
                    break
                a += 1
            else:
                pending = _sample(ps[len(drafts)], greedy)  # bonus token
        for j in range(a):
            stats.accepted_at_depth[j] += 1
        if drafts:
            stats.rounds += 1
            stats.drafted += len(drafts)
            stats.accepted += a
            stats.drafts.append((L, drafts))
        truncate_cache(cache, L + a)                        # pending + accepted drafts stay cached
        hid[0] = (L - 1, h[:, :a + 1])                      # trunk states of positions L-1 .. L-1+a
        for d in drafts[:a]:                                # accepted drafts are emitted in order
            if stop_ids and d in stop_ids:
                return
            yield d
            emitted += 1
            stats.tokens += 1
            if emitted >= max_new_tokens:
                return
        seq.extend(drafts[:a] + [pending])
        if d_round == 0:            # plain decoding from here on: MTP state is no longer kept up to date
            D = 0


def speculative_generate(model: GPT, tokenizer, prompt: str, **kw) -> str:
    """Like `generate.generate`, but with MTP self-speculative decoding (greedy by default)."""
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt) or [tokenizer.token_id("<|bos|>")]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    stop = {tokenizer.token_id("<|eos|>")}
    return tokenizer.decode(list(mtp_speculative_stream(model, x, stop_ids=stop, **kw)))
