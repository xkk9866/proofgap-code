"""Tokenised evidence pools and evidence-state sampling."""
import json
import random
import torch

QMAX, DMAX, SMAX, LMAX = 64, 200, 100, 512
CTX_MAX = {"v": SMAX}  # per-passage cap of the context (held / other members); ProofGap v4 raises it to DMAX (set_ctx_max)


def set_ctx_max(n):
    """Context passages are truncated to n tokens each (they always share the room left by question and focus)."""
    CTX_MAX["v"] = int(n)


def load_jsonl(p, limit=None):
    out = []
    with open(p) as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def ctx_str(c):
    return f"{c['title']}: {c['text']}"


class Pool:
    """A question with its tokenised candidate pool."""

    def __init__(self, ex, tok):
        self.ex = ex
        self.q = tok(ex["question"], add_special_tokens=False)["input_ids"][:QMAX]
        self.d = [ids[:DMAX] for ids in tok([ctx_str(c) for c in ex["ctxs"]], add_special_tokens=False)["input_ids"]]
        # annotated supporting passages in hop order; ids >= len(ctxs) mark passages the pool does not contain
        # (open-corpus pools).  `gold` keeps the selectable ones; `n_out` counts requirements no set can cover.
        self.gold_all = list(ex["gold"])
        self.gold = [g for g in self.gold_all if g < len(self.d)]
        self.n_out = len(self.gold_all) - len(self.gold)

    def deficit(self, S):
        """Proof deficit r(S) = |G minus S| (annotated passages outside the pool always count as missing)."""
        return len(set(self.gold) - set(S)) + self.n_out


def build_seqs(pool, cand, state, cls_id, sep_id):
    """Sequence for each candidate d: [CLS] q [SEP] s_1 [SEP] ... s_k [SEP] d [SEP]."""
    seqs = []
    for d in cand:
        ctx = [s for s in state if s != d]
        dl = pool.d[d]
        room = LMAX - len(pool.q) - len(dl) - 3 - len(ctx)
        per = min(CTX_MAX["v"], room // max(1, len(ctx))) if ctx else 0
        ids = [cls_id] + pool.q + [sep_id]
        for s in ctx:
            ids += pool.d[s][:per] + [sep_id]
        ids += dl + [sep_id]
        seqs.append(ids[:LMAX])
    return seqs


def build_view_seqs(pool, T, cls_id, sep_id, focus=None):
    """Member-focused views of a set T (ProofGap v4): one sequence per member t,
    [CLS] q [SEP] (T minus t, truncated) [SEP] ... t (full length) [SEP].
    The view through t is exactly the sequence a state-conditioned retriever computes when it extends T minus {t}
    by t, so the child reading of the search and the verification of a candidate set are the same function.
    focus: optional list of members to read (default: all of T, in the given order)."""
    T = list(T)
    return [build_seqs(pool, [t], [s for s in T if s != t], cls_id, sep_id)[0] for t in (focus or T)]


def collate(items, cls_id, sep_id, pad_id):
    """items: list of (pool, cand list, state list). Returns padded tensors."""
    B = len(items)
    N = max(len(c) for _, c, _ in items)
    all_seqs, mask, sel = [], torch.zeros(B, N, dtype=torch.bool), torch.zeros(B, N, dtype=torch.long)
    for b, (pool, cand, state) in enumerate(items):
        seqs = build_seqs(pool, cand, state, cls_id, sep_id)
        seqs += [[cls_id, sep_id]] * (N - len(cand))
        all_seqs += seqs
        mask[b, : len(cand)] = True
        for i, d in enumerate(cand):
            if d in state:
                sel[b, i] = 1
    L = max(len(s) for s in all_seqs)
    ids = torch.full((len(all_seqs), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(all_seqs), L), dtype=torch.long)
    for i, s in enumerate(all_seqs):
        ids[i, : len(s)] = torch.tensor(s)
        attn[i, : len(s)] = 1
    return ids, attn, mask, sel


def sample_state(gold, negs, rng):
    """Evidence states covering empty, leave-one-out, partial and corrupted sets."""
    g = list(gold)
    u = rng.random()
    if u < 0.15 or len(g) == 0:
        return []
    if u < 0.60:  # leave-one-out: exactly one missing requirement witness
        miss = rng.randrange(len(g))
        return [x for i, x in enumerate(g) if i != miss]
    if u < 0.80:  # proper partial prefix in hop order
        k = rng.randrange(0, len(g))
        return g[:k]
    # corrupted state: any subset of gold (possibly complete) plus 1-2 distractors, shuffled
    k = rng.randrange(0, len(g) + 1)
    st = rng.sample(g, k) + rng.sample(negs, min(len(negs), rng.randint(1, 2)))
    rng.shuffle(st)
    return st


def make_train_item(pool, rng, n_train):
    gold = pool.gold
    negs = [i for i in range(len(pool.d)) if i not in gold]
    state = sample_state(gold, negs, rng) if gold else rng.sample(negs, min(len(negs), rng.randint(0, 2)))
    keep_neg = [x for x in negs if x in state]
    rest = [x for x in negs if x not in state]
    rng.shuffle(rest)
    cand = list(dict.fromkeys(gold + keep_neg + rest[: max(0, n_train - len(gold) - len(keep_neg))]))
    rng.shuffle(cand)
    return cand, state


def build_set_seq(pool, S, cls_id, sep_id):
    """Joint encoding of a whole evidence set for the verdict: [CLS] q [SEP] s_1 [SEP] ... s_k [SEP]."""
    per = min(DMAX, (LMAX - len(pool.q) - 2 - len(S)) // max(1, len(S)))
    ids = [cls_id] + pool.q + [sep_id]
    for s in S:
        ids += pool.d[s][:per] + [sep_id]
    return ids[:LMAX]


def collate_sets(seqs, pad_id):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        attn[i, : len(s)] = 1
    return ids, attn
