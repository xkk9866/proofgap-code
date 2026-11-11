"""Train ERC-Net or a controlled baseline head on one dataset.

python train.py --data musique --head erc --out ckpt/musique_erc
"""
import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data import Pool, build_set_seq, build_view_seqs, collate, collate_sets, load_jsonl, make_train_item
from model import NRUNG, R_DEFICIT, R_GAP, ERCNet, cum_logits, log1mexp, resolve_enc

DATA = os.environ.get("PROOFGAP_DATA_DIR", "data/proc")
DEV = "cuda" if torch.cuda.is_available() else "cpu"  # CPU only for smoke tests (bf16 autocast on CUDA, fp32 on CPU)


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16) if DEV == "cuda" else torch.autocast("cpu", enabled=False)


def bce_log(logp, y):
    """Binary cross-entropy given log-probabilities."""
    return -(y * logp + (1 - y) * log1mexp(logp))


def set_loss(out, items, device):
    """Counterfactual completeness likelihood on S and on every one-step completion S + {d}."""
    logC, logC_add = out["logC"], out["logC_add"]
    losses = []
    for b, (pool, cand, state) in enumerate(items):
        G = set(pool.gold)
        yS = float(G.issubset(state))
        idx = [i for i, d in enumerate(cand) if d not in state]
        y = torch.tensor([float(G.issubset(set(state) | {cand[i]})) for i in idx], device=device)
        l_add = bce_log(logC_add[b, idx], y).mean()
        losses.append(0.5 * (bce_log(logC[b], torch.tensor(yS, device=device)) + l_add))
    return torch.stack(losses).mean()


def assign_loss(out, items, device):
    """Complete-data likelihood of the MAP requirement-witness assignment (hard EM, Hungarian)."""
    la, lz = out["logit_a"], out["logit_z"]
    losses = []
    for b, (pool, cand, state) in enumerate(items):
        n = len(cand)
        gpos = [cand.index(g) for g in pool.gold if g in cand][: la.size(1)]
        T = torch.zeros(la.size(1), n, device=device)
        tz = torch.zeros(la.size(1), device=device)
        if gpos:
            cost = -(F.logsigmoid(la[b][:, gpos]) + F.logsigmoid(lz[b]).unsqueeze(-1))
            r, c = linear_sum_assignment(cost.detach().float().cpu().numpy())
            for j, k in zip(r, c):
                T[j, gpos[k]] = 1.0
                tz[j] = 1.0
        la_b = la[b, :, :n]
        l_neg = -F.logsigmoid(-la_b)[T == 0].mean()
        l_pos = -F.logsigmoid(la_b)[T == 1].mean() if gpos else torch.zeros((), device=device)
        l_z = F.binary_cross_entropy_with_logits(lz[b], tz)
        losses.append(l_pos + l_neg + l_z)
    return torch.stack(losses).mean()


def assign_loss2(out, items, device):
    """Factorised circuit: witness BCE (d in G), MAP requirement assignment (Hungarian) for gold, activations."""
    lp, lpi, lz = out["logit_p"], out["log_pi"], out["logit_z"]
    losses = []
    for b, (pool, cand, state) in enumerate(items):
        n = len(cand)
        y = torch.tensor([float(d in pool.gold) for d in cand], device=device)
        l_w = F.binary_cross_entropy_with_logits(lp[b, :n], y)
        gpos = [cand.index(g) for g in pool.gold if g in cand][: lpi.size(1)]
        tz = torch.zeros(lpi.size(1), device=device)
        l_pi = torch.zeros((), device=device)
        if gpos:
            cost = -(lpi[b][:, gpos] + F.logsigmoid(lz[b]).unsqueeze(-1))
            r, c = linear_sum_assignment(cost.detach().float().cpu().numpy())
            l_pi = -torch.stack([lpi[b, j, gpos[k]] for j, k in zip(r, c)]).mean()
            tz[list(r)] = 1.0
        losses.append(l_w + l_pi + F.binary_cross_entropy_with_logits(lz[b], tz))
    return torch.stack(losses).mean()


def point_loss(out, items, device):
    """Next-evidence classifier: label 1 iff d is a still-missing gold passage (Beam Retrieval style)."""
    sc = out["score"]
    losses = []
    for b, (pool, cand, state) in enumerate(items):
        G = set(pool.gold) - set(state)
        idx = [i for i, d in enumerate(cand) if d not in state]
        y = torch.tensor([float(cand[i] in G) for i in idx], device=device)
        losses.append(F.binary_cross_entropy_with_logits(sc[b, idx], y))
    return torch.stack(losses).mean()


def counterfactual_sets(out, items, rng, n_cf=2, random_cf=False, superset=False):
    """Self-counterfactual proofs for the verdict.

    For a state S with missing gold M = G minus S, the positive is the completed set S + M.  Negatives are
    (i) S itself when M is non-empty (an unfinished proof) and (ii) for the n_cf candidates x the proposer
    itself ranks highest among non-gold passages, the near-proof S + (M - {m}) + {x}: one missing
    requirement filled by the network's own most convincing wrong witness.  If S is already complete,
    a held gold passage is swapped for x.  random_cf replaces the proposer's choice by uniform sampling.
    """
    sets, ys = [], []
    for b, (pool, cand, state) in enumerate(items):
        G = set(pool.gold)
        sc = out["score"][b, : len(cand)].detach().float()
        neg = [i for i in torch.argsort(sc, descending=True).tolist() if cand[i] not in G and cand[i] not in state]
        xs = rng.sample(neg, min(n_cf, len(neg))) if random_cf else neg[:n_cf]
        xs = [cand[i] for i in xs]
        miss = [g for g in pool.gold if g not in state]
        pos = list(state) + miss
        rows = [(pos, 1.0)]
        if miss:
            rows.append((list(state), 0.0))
            for x in xs:
                m = miss[rng.randrange(len(miss))]
                rows.append((list(state) + [g for g in miss if g != m] + [x], 0.0))
        else:
            for x in xs:
                g = pool.gold[rng.randrange(len(pool.gold))]
                rows.append(([d for d in state if d != g] + [x], 0.0))
        if superset and xs:
            # supersets: completeness is monotone, so a complete set plus a distractor is complete,
            # and a near-proof plus a distractor is still incomplete
            extra = [cand[i] for i in neg if cand[i] not in xs][:1] or xs[-1:]
            rows.append((pos + extra, 1.0))
            if miss:
                m = miss[rng.randrange(len(miss))]
                rows.append((list(state) + [g for g in miss if g != m] + [xs[0]] + extra, 0.0))
        for S, y in rows:
            S = list(dict.fromkeys(S)); rng.shuffle(S)
            sets.append((pool, S)); ys.append(y)
    return sets, ys


def deficit_sets(out, items, rng, n_cf=2, random_cf=False):
    """Self-mined proof-deficit ladder for one state S (missing gold M = G minus S).

    Every row is a set S' built from the state, gold passages and the proposer's own most convincing wrong
    passages (impostors), labelled with its proof deficit r(S') = |G minus S'|: the completed proof (r=0),
    the completed proof with an impostor added (r=0, a superset), near-proofs where one or two missing
    requirements are filled by impostors (r=1, 2), a near-proof with an extra impostor (superset, r=1), and
    S itself (r=|M|).  The ladder teaches one energy to order sets by how far they are from a proof.
    """
    sets, ys = [], []
    for b, (pool, cand, state) in enumerate(items):
        G = set(pool.gold)
        sc = out["score"][b, : len(cand)].detach().float()
        neg = [i for i in torch.argsort(sc, descending=True).tolist() if cand[i] not in G and cand[i] not in state]
        neg = [cand[i] for i in (rng.sample(neg, len(neg)) if random_cf else neg)]
        xs, extra = neg[:n_cf], neg[n_cf: n_cf + 1] or neg[:1]
        miss = [g for g in pool.gold if g not in state]
        pos = list(state) + miss
        rows = [pos, pos + extra]
        if miss:
            rows.append(list(state))
        for x in xs:
            drop = rng.sample(pool.gold, 1)
            rows.append([d for d in pos if d not in drop] + [x])
        if len(pool.gold) >= 2 and len(xs) >= 2:
            drop = rng.sample(pool.gold, 2)
            rows.append([d for d in pos if d not in drop] + xs[:1])
        if xs and extra:
            drop = rng.sample(pool.gold, 1)
            rows.append([d for d in pos if d not in drop] + xs[:1] + extra)
        for S in rows:
            S = list(dict.fromkeys(S)); rng.shuffle(S)
            if not S:
                continue
            sets.append((pool, S)); ys.append(min(len(G - set(S)), R_DEFICIT - 1))
    return sets, ys


def ladder_rows(mine, items, rng, n_cf=2, pad=0, random_cf=False, R=R_DEFICIT):
    """Graded counterfactual sets for the holistic reading, each labelled with its proof deficit r(T) = |G minus T|.

    mine[b] lists the candidates of item b from the most to the least convincing completion of the state (as judged
    by the network itself); impostors are the most convincing non-gold candidates.  For a state S with missing
    in-pool requirements M and completion P = S + M the rows are
      P (r = n_out), the unfinished S (r = |M| + n_out), near-proofs P - {g} + {x_i} (r = 1 + n_out),
      a two-gap set P - {g, g'} + {x_1} (r = 2 + n_out),
      and padded supersets of P and of a near-proof (padding never changes the deficit) whose size is drawn
      uniformly up to `pad`, so that the reader is trained on every set size it reads at inference.
    n_out > 0 only for pools that miss annotated passages: then even P is incomplete, which teaches the reader that
    a proof can be unreachable in the pool at hand.  pad = 0 reproduces the single-impostor padding of v2.
    """
    sets, ys = [], []
    for b, (pool, cand, state) in enumerate(items):
        G = set(pool.gold)
        neg = [cand[i] for i in mine[b] if cand[i] not in G and cand[i] not in state]
        if random_cf:
            neg = rng.sample(neg, len(neg))
        xs, spare = neg[:n_cf], neg[n_cf:] or neg[:1]
        miss = [g for g in pool.gold if g not in state]
        P = list(state) + miss
        golds = [g for g in P if g in G]
        rows = [P] if P else []
        if miss and state:
            rows.append(list(state))
        near = []
        for x in xs:
            if golds:
                drop = rng.choice(golds)
                near.append([d for d in P if d != drop] + [x])
        rows += near
        if len(golds) >= 2 and xs:
            drop = rng.sample(golds, 2)
            rows.append([d for d in P if d not in drop] + xs[:1])
        if not golds and xs:  # nothing of the proof is in the pool: impostor sets are as far as the state
            rows.append(list(state) + xs[:1])
        for base in ([P] if P else []) + near[:1]:
            room = [x for x in spare if x not in base]
            if not room:
                continue
            if pad:
                if len(base) >= pad:
                    continue
                size = rng.randint(len(base) + 1, pad)
                rows.append(base + room[: size - len(base)])
            else:
                rows.append(base + room[:1])
        for S in rows:
            S = list(dict.fromkeys(S)); rng.shuffle(S)
            if not S:
                continue
            sets.append((pool, S)); ys.append(min(pool.deficit(S), R - 1))
    return sets, ys


# ----------------------------------------------------------------------------------------------------------------------
# ProofGap v4: one deficit energy, read through the members of a set, trained with two likelihoods
# ----------------------------------------------------------------------------------------------------------------------
def consensus(u_views, owner, n_sets):
    """Consensus deficit ladder of every set: the mean of the cumulative logits of its member-focused views.
    u_views [V, R] cumulative logits of the views, owner [V] set index of each view -> [n_sets, R].
    The mean of monotone sequences is monotone, so the consensus is again a valid ladder."""
    out = torch.zeros(n_sets, u_views.size(1), device=u_views.device, dtype=u_views.dtype)
    cnt = torch.zeros(n_sets, device=u_views.device, dtype=u_views.dtype)
    out.index_add_(0, owner, u_views)
    cnt.index_add_(0, owner, torch.ones(owner.numel(), device=u_views.device, dtype=u_views.dtype))
    return out / cnt.clamp(min=1).unsqueeze(1)


def listwise_nll(logits, y):
    """Within-question likelihood: -log P(the sets closest to a proof win | the group).
    P(T | group) = softmax over the group of `logits` (log P_theta(r(T) <= h) for the relevant rung h); the target
    mass is spread over the members with the smallest true deficit."""
    tgt = y == y.min()
    return torch.logsumexp(logits, 0) - torch.logsumexp(logits[tgt], 0)


def graded_groups(mine, items, rng, n_cf, pad, random_cf, R):
    """The graded counterfactual sets of every state (ladder_rows), kept as one group per state for the listwise term."""
    groups = []
    for b in range(len(items)):
        sets, ys = ladder_rows([mine[b]], [items[b]], rng, n_cf, pad, random_cf, R)
        groups.append((items[b][0], [S for _, S in sets], ys))
    return groups


def v4_loss(model, out, items, rng, dev, a, ids_fn):
    """ProofGap v4 training objective for one batch of (pool, candidates, state) items.

    The network reads a set only ever through one of its members, [q; T minus t; t] (data.build_view_seqs); the
    deficit ladder of the set is the consensus of its views.  Two likelihoods of the same ladder are maximised:
      absolute  - ordinal NLL of the true deficit r(T) = |G minus T| for every view read and for every consensus
                  (children of the sampled state read through the new member; graded counterfactual sets read
                  through all their members);
      relative  - listwise likelihood that, among the children of one state, those that lower the deficit win
                  (logit: log P(r(child) <= r(S) - 1)), and that, among the graded sets built from one state, the
                  ones closest to a proof win (logit: log P(r(T) = 0) of the consensus).
    A linear probe on detached features reproduces the proposal logit of chain retrievers for the controls only.
    --views holistic reads every set once as [q; t_1; ...; t_k] (v3 reading) instead of through its members.
    --set_target binary keeps only rung 0 (completeness) and, like PROVER, ranks children by a trained proposal logit.
    """
    cls_id, sep_id, pad_id = ids_fn
    h, R = out["h"], model.R
    binary = a.set_target == "binary"
    member = a.member or a.nec  # the focused passage's necessity is a second reading of every view, trained through the encoder
    l_cnll, l_clist, l_probe, mine = [], [], [], []
    for b, (pool, cand, state) in enumerate(items):
        st, n = set(state), len(cand)
        idx = [i for i, d in enumerate(cand) if d not in st]
        if not idx:
            mine.append(list(range(n)))
            continue
        hb = h[b, idx].float()
        u = model.to_cum(model.ver(hb))  # ladder of every child S + {d}, read through d
        y = torch.tensor([min(pool.deficit(st | {cand[i]}), R - 1) for i in idx], device=dev)
        yp = torch.tensor([float(cand[i] in pool.gold) for i in idx], device=dev)
        mlog = model.cls(hb if (binary or member) else hb.detach()).squeeze(-1)  # membership of the new member d
        l_probe.append(F.binary_cross_entropy_with_logits(mlog, yp))
        if binary:  # completeness only: children are ranked by the membership logit
            l_cnll.append(F.binary_cross_entropy_with_logits(u[:, 0], (y == 0).float()))
            key = -mlog.detach()
        else:
            l_cnll.append(ladder_nll(u, y))
            m = min(pool.deficit(st), R - 1)
            if m >= 1 and not a.no_list:  # which children bring the state one step closer to a proof
                l_clist.append(listwise_nll(F.logsigmoid(u[:, m - 1]), y))
            key = -mlog.detach() if a.member else torch.sigmoid(-u[:, : R - 1]).sum(1).detach()
        order = [idx[j] for j in torch.argsort(key).tolist()]  # most convincing completions first (impostor mining)
        mine.append(order + [i for i in range(n) if i not in idx])
    groups = graded_groups(mine, items, rng, a.n_cf, a.pad, a.random_cf, R)
    seqs, owner, ys, gid, mem = [], [], [], [], []
    for g, (pool, sets, yg) in enumerate(groups):
        for S, y in zip(sets, yg):
            sid = len(ys)
            ys.append(y); gid.append(g)
            if a.views == "holistic":
                views = [build_set_seq(pool, S, cls_id, sep_id)]
            else:
                views = build_view_seqs(pool, S, cls_id, sep_id)
                mem += [float(t in pool.gold) for t in S]  # membership label of the focused member of every view
            seqs += views; owner += [sid] * len(views)
    vi, va = collate_sets(seqs, pad_id)
    us, ms = [], []
    for c0 in range(0, len(seqs), a.chunk):
        with amp():
            o, ml = model.view_heads(vi[c0: c0 + a.chunk].to(dev), va[c0: c0 + a.chunk].to(dev))
        us.append(model.to_cum(o.float())); ms.append(ml.float())
    u_views = torch.cat(us)
    owner_t, y_t, gid_t = (torch.tensor(x, device=dev) for x in (owner, ys, gid))
    ubar = consensus(u_views, owner_t, len(ys))
    tgt = (lambda u, y: F.binary_cross_entropy_with_logits(u[:, 0], (y == 0).float())) if binary else ladder_nll
    # a batch whose states already contain every candidate has no children; keep a connected zero so training continues
    z = ubar.sum() * 0
    loss = (torch.stack(l_cnll).mean() if l_cnll else z) + (torch.stack(l_probe).mean() if l_probe else z) + tgt(ubar, y_t)
    if a.views != "holistic":
        loss = loss + tgt(u_views, y_t[owner_t])
        if member and mem:
            loss = loss + F.binary_cross_entropy_with_logits(torch.cat(ms), torch.tensor(mem, device=dev))
    if l_clist:
        loss = loss + torch.stack(l_clist).mean()
    if not a.no_list:
        l_slist = [listwise_nll(F.logsigmoid(ubar[gid_t == g, 0]), y_t[gid_t == g]) for g in range(len(groups)) if (gid_t == g).sum() >= 2]
        if l_slist:
            loss = loss + torch.stack(l_slist).mean()
    return loss


def set_target_loss(u, y, target, R=R_DEFICIT):
    """Loss of the deficit reading: ordinal ladder NLL (ProofGap), binary completeness BCE (rung 0 only) or an
    unordered softmax over the deficit classes (u then holds raw class logits)."""
    if target == "binary":
        return F.binary_cross_entropy_with_logits(u[:, 0], (y == 0).float())
    if target == "multiclass":
        return F.cross_entropy(u[:, :R], y)
    return ladder_nll(u, y)


def inc_loss(model, out, items, device, target):
    """Incremental reading: the ladder head on the proposer's encoding [q; S; d] predicts the deficit of the child
    S + {d}.  Labels are free (the deficit of every child follows from the annotation) and the encoding is the one
    the proposer computes anyway, so this costs no encoder pass."""
    h = out["h"]
    rows, ys = [], []
    for b, (pool, cand, state) in enumerate(items):
        st = set(state)
        for i, d in enumerate(cand):
            if d not in st:
                rows.append(h[b, i]); ys.append(min(pool.deficit(st | {d}), model.R - 1))
    o = model.ver(torch.stack(rows).float())
    y = torch.tensor(ys, device=device)
    return set_target_loss(o if target == "multiclass" else cum_logits(o), y, target, model.R)


def mine_order(model, out, items, use_inc):
    """Candidates of every item ordered from the most to the least convincing completion of its state: by the
    incremental expected deficit of the child (use_inc) or by the proposal logit."""
    order = []
    with torch.no_grad():
        for b, (pool, cand, state) in enumerate(items):
            n = len(cand)
            if use_inc:
                u = model.ladder_from_h(out["h"][b, :n].detach())
                # Binary supervision trains only rung 0; never mine with its untrained higher rungs.
                key = -u[:, 0] if model.ladder_kind == "binary" else torch.sigmoid(-u[:, : model.R - 1]).sum(1)
                order.append(torch.argsort(key).tolist())
            else:
                order.append(torch.argsort(out["score"][b, :n].detach().float(), descending=True).tolist())
    return order


def gap_rows(items, rng):
    """Training sets for the proof-gap field: the state S itself and every one-step extension S + {d}.

    Each row is a set read as one sequence [q; S; d] and labelled with its proof gap r = |G minus (S + {d})|.
    Because states include leave-one-out sets, complete sets and sets corrupted by distractors, the rows
    contain near-proofs (the one missing passage replaced by every candidate impostor), completed proofs,
    supersets of proofs and partial chains, without any hand-built negative.
    """
    sets, ys = [], []
    for pool, cand, state in items:
        G = set(pool.gold)
        st = list(state)
        rows = [st] if st else []
        for d in cand:
            if d not in state:
                base = st[:]; rng.shuffle(base)
                rows.append(base + [d])
        for S in rows:
            sets.append((pool, S)); ys.append(min(len(G - set(S)), R_GAP - 1))
    return sets, ys


def descent_rows(model, items, top, cls_id, sep_id, pad_id, dev, kmax=5):
    """On-policy rows: the sets the field's own greedy descent visits on the training pool.

    At every depth the frontier is extended by every unused candidate, the children are read once
    (no gradient), and the `top` children with the largest P(r=0) plus the E[r]-descent head become
    training rows labelled with their true gap. These are exactly the near-proofs the budget decision
    confuses with complete sets at test time.
    """
    sets, ys = [], []
    for pool, cand, _ in items:
        G = set(pool.gold)
        head = []
        for _ in range(min(kmax, len(cand))):
            ch = [head + [d] for d in cand if d not in head]
            lp, er = [], []
            with torch.no_grad():
                for c0 in range(0, len(ch), 64):
                    vi, va = collate_sets([build_set_seq(pool, S, cls_id, sep_id) for S in ch[c0: c0 + 64]], pad_id)
                    with amp():
                        a, b, _ = model.gap_field(vi.to(dev), va.to(dev))
                    lp += a.float().tolist(); er += b.float().tolist()
            pick = set(sorted(range(len(ch)), key=lambda i: -lp[i])[:top])
            hi = min(range(len(ch)), key=lambda i: er[i])
            for i in pick | {hi}:
                sets.append((pool, ch[i])); ys.append(min(len(G - set(ch[i])), R_GAP - 1))
            head = ch[hi]
    return sets, ys


def gap_loss(model, lp0, er, lpmf, raw_y, target):
    """Likelihood of the observed gap under the field: ordinal / multiclass NLL, binary BCE, or regression."""
    y = raw_y
    if target in ("ordinal", "multiclass"):
        return -lpmf.gather(1, y.unsqueeze(1)).mean()
    if target == "binary":
        return F.binary_cross_entropy_with_logits(lpmf[:, 0], (y == 0).float())
    return F.smooth_l1_loss(er, y.float())


def ladder_nll(u, y):
    """-log P(r = y) for the vector ladder with cumulative logits u [n, R]."""
    cdf = torch.sigmoid(u)
    cdf = torch.cat([cdf[:, :-1], torch.ones_like(cdf[:, :1])], 1)
    pmf = cdf - torch.cat([torch.zeros_like(cdf[:, :1]), cdf[:, :-1]], 1)
    return -torch.log(pmf.gather(1, y.unsqueeze(1)).clamp_min(1e-8)).mean()


def ordinal_nll(f, theta, y):
    """-log P(r = y) under the cumulative-link ladder P(r <= h) = sigmoid(theta_h - f)."""
    cdf = torch.sigmoid(theta.unsqueeze(0) - f.unsqueeze(1))  # [n, R]
    cdf = torch.cat([cdf[:, :-1], torch.ones_like(cdf[:, :1])], 1)  # last rung is "r >= R-1"
    pmf = cdf - torch.cat([torch.zeros_like(cdf[:, :1]), cdf[:, :-1]], 1)
    return -torch.log(pmf.gather(1, y.unsqueeze(1)).clamp_min(1e-8)).mean()


def onpolicy_beam_loss(model, items, B, cls_id, sep_id, pad_id, dev, accum):
    """Official Beam Retrieval training (Zhang et al., 2024): hop by hop over the model's OWN beam.

    Hop t scores every sequence prefix+[j] for the B prefixes kept by the model at hop t-1 (jointly encoded),
    label 1 iff set(prefix+[j]) == set(gold[:t+1]); the B best sequences by their CURRENT score are kept
    (no score accumulation), and training stops early once no kept prefix is a gold prefix.
    Each hop's loss is back-propagated immediately (prefixes are detached), so memory is one hop's batch.
    """
    total = 0.0
    for pool, cand, _ in items:
        gold = list(pool.gold_all)
        beams = [[]]
        for t in range(len(gold)):
            pairs = [(pre, d) for pre in beams for d in cand if d not in pre]
            if not pairs:
                break
            ids, attn, mask, sel = collate([(pool, [d], pre) for pre, d in pairs], cls_id, sep_id, pad_id)
            with amp():
                out = model(ids.to(dev), attn.to(dev), mask.to(dev), sel.to(dev))
            sc = out["score"][:, 0].float()
            y = torch.tensor([float(set(pre + [d]) == set(gold[: t + 1])) for pre, d in pairs], device=dev)
            loss = F.binary_cross_entropy_with_logits(sc, y)
            (loss / (accum * len(items) * len(gold))).backward()
            total += loss.item() / len(gold)
            top = torch.topk(sc.detach(), min(B, len(pairs))).indices.tolist()
            beams = [pairs[i][0] + [pairs[i][1]] for i in top]
            if not any(set(b) == set(gold[: t + 1]) for b in beams):
                break
    return total / len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--head", default="erc")
    ap.add_argument("--enc", default="/data/yaokang/models/dv3/deberta-v3-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--n_train", type=int, default=12)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--no_set", action="store_true")
    ap.add_argument("--no_assign", action="store_true")
    ap.add_argument("--no_setloss", action="store_true")
    ap.add_argument("--no_state", action="store_true", help="state enters only through selection flags")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n_cf", type=int, default=2, help="PROVER: self-counterfactual near-proofs per state")
    ap.add_argument("--random_cf", action="store_true", help="PROVER ablation: random instead of self-chosen counterfactuals")
    ap.add_argument("--no_verdict", action="store_true", help="PROVER ablation: proposer only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--superset", action="store_true", help="PROVER: also train the verdict on supersets")
    ap.add_argument("--gc", action="store_true", help="gradient checkpointing (memory)")
    ap.add_argument("--onpolicy_beam", type=int, default=0, help="official Beam Retrieval training on the model's own beam")
    ap.add_argument("--target", default="ordinal", help="gap head: ordinal | multiclass | binary | regress")
    ap.add_argument("--holdout", type=int, default=0, help="keep the last N training questions out (calibration / model selection)")
    ap.add_argument("--descent_rows", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=16, help="gap head: sequences per forward pass")
    # ---- ProofGap v3 ----
    ap.add_argument("--inc", action="store_true", help="ladder: also train the incremental reading [q;S;d] -> deficit of S+{d}")
    ap.add_argument("--prop", default="bce", help="ladder: proposal loss bce | none")
    ap.add_argument("--pad", type=int, default=0, help="ladder: pad graded rows to random sizes up to this budget (0 = v2)")
    ap.add_argument("--set_target", default="ordinal", help="ladder: ordinal | binary | multiclass (deficit reading target)")
    ap.add_argument("--open_mix", type=float, default=0.0, help="probability of presenting a question with its open-style pool")
    ap.add_argument("--hop_heads", action="store_true", help="point head: separate first-hop classifier (official Beam)")
    ap.add_argument("--rungs", type=int, default=R_DEFICIT, help="ladder: deficit classes R (v3: 5 = exact deficits 0..4)")
    # ---- ProofGap v4 ----
    ap.add_argument("--v4", action="store_true", help="ladder: member-focused consensus reading + listwise likelihoods (v4_loss)")
    ap.add_argument("--views", default="consensus", help="v4: consensus (read a set through each member) | holistic (one joint reading, v3)")
    ap.add_argument("--no_list", action="store_true", help="v4 ablation: drop the within-question listwise likelihoods")
    ap.add_argument("--member", action="store_true", help="v5: train the membership reading of every view (focused passage in the proof?) through the encoder")
    ap.add_argument("--nec", action="store_true", help="v6: every view [q; T minus t; t] also reads the necessity of t, r(T minus t) - r(T); mining unchanged")
    ap.add_argument("--init", default="", help="initialise from this checkpoint (model.pt) instead of the bare backbone")
    ap.add_argument("--save_every", type=int, default=400, help="write a resumable checkpoint every N optimiser steps")
    ap.add_argument("--save_min", type=float, default=6.0, help="... or whenever this many minutes passed since the last one")
    ap.add_argument("--ctx_max", type=int, default=0, help="v4: per-passage cap of context passages in a view (0 = v3 cap of 100; v4 default 200)")
    a = ap.parse_args()
    if a.v4:
        assert a.head == "ladder" and a.set_target in ("ordinal", "binary"), "v4 supports the ordinal ladder and its binary ablation"
        a.ctx_max = a.ctx_max or 200
    if a.ctx_max:
        import data as _data
        _data.set_ctx_max(a.ctx_max)

    log_every = int(os.environ.get("LOG_EVERY", "100"))
    if DEV == "cuda" and os.environ.get("MEM_CAP_MIB"):  # shared GPUs: never grow past this share of the card
        torch.cuda.set_per_process_memory_fraction(int(os.environ["MEM_CAP_MIB"]) * 2 ** 20 / torch.cuda.get_device_properties(0).total_memory)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = DEV
    a.enc = resolve_enc(a.enc)
    tok = AutoTokenizer.from_pretrained(a.enc)
    exs = load_jsonl(f"{DATA}/{a.data}_train.jsonl", a.limit)
    exs = [e for e in exs if len(e["gold"]) > 0]
    if a.holdout:  # the held-out questions (written by build_open_train.py as {data}_hold.jsonl) calibrate / select
        exs = exs[: len(exs) - a.holdout]
    pools = [Pool(e, tok) for e in exs]
    open_pools = {}
    if a.open_mix > 0:  # open-style pools of the SAME training questions (build_open_train.py)
        keep = {e["id"] for e in exs}
        for e in load_jsonl(f"{DATA}/{a.data}_train_open.jsonl"):
            if e["id"] in keep:
                open_pools[e["id"]] = Pool(e, tok)
    print(f"{len(pools)} training pools, {len(open_pools)} open-style pools", flush=True)

    head = "point_noset" if (a.head == "point" and a.no_set) else a.head
    model = ERCNet(a.enc, head=head, m=a.m, use_set=not a.no_set, grad_ckpt=a.gc, hop_heads=a.hop_heads, n_rung=a.rungs).to(dev)
    if a.hop_heads:
        model.sep_id = tok.sep_token_id
    if head == "ladder":
        model.ladder_kind = a.set_target
    if head == "gap":
        model.target = a.target
    if a.init:
        ck = torch.load(a.init, map_location="cpu")
        missing, unexpected = model.load_state_dict(ck["state"], strict=False)
        print(f"initialised from {a.init}: missing {len(missing)} unexpected {len(unexpected)}", flush=True)
    enc_p = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_p = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": a.lr}, {"params": head_p, "lr": a.head_lr}], weight_decay=0.01)
    steps = int(math.ceil(len(pools) * a.epochs / (a.bs * a.accum)))
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    cls_id, sep_id, pad_id = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id

    model.train()
    t0, it, run = time.time(), 0, []
    total_items = int(len(pools) * a.epochs)
    order = []
    while len(order) < total_items:
        perm = list(range(len(pools))); rng.shuffle(perm); order += perm
    order = order[:total_items]
    # ---- mid-run checkpointing: shared GPUs can be taken over by other users at any time; a killed job resumes here
    resume_path = f"{a.out}/resume.pt"
    start_b = 0
    if os.path.exists(resume_path):
        rs = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(rs["state"]); opt.load_state_dict(rs["opt"]); sch.load_state_dict(rs["sch"])
        rng.setstate(rs["rng"]); torch.set_rng_state(rs["torch_rng"])
        if DEV == "cuda" and rs.get("cuda_rng") is not None: torch.cuda.set_rng_state(rs["cuda_rng"])
        it, run, start_b = rs["it"], rs["run"], rs["bstart"]
        t0 -= rs["elapsed"]
        print(f"resumed from {resume_path} at item {start_b}/{total_items}, step {it}", flush=True)

    def save_resume(next_b):
        tmp = resume_path + ".tmp"
        torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(), "rng": rng.getstate(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state() if DEV == "cuda" else None, "it": it, "run": run[-400:],
                    "bstart": next_b, "elapsed": time.time() - t0}, tmp)
        os.replace(tmp, resume_path)

    os.makedirs(a.out, exist_ok=True)
    last_save = time.time()
    for bstart in range(start_b, total_items, a.bs):
        if it and (bstart // a.bs) % a.accum == 0 and bstart > start_b and (it % a.save_every == 0 or time.time() - last_save > 60 * a.save_min):
            save_resume(bstart); last_save = time.time()
        items = []
        for qi in order[bstart: bstart + a.bs]:
            pool = pools[qi]
            if open_pools and rng.random() < a.open_mix and pool.ex["id"] in open_pools:
                pool = open_pools[pool.ex["id"]]
            cand, state = make_train_item(pool, rng, a.n_train)
            # A pointwise CE sees no state: every in-pool gold remains positive.
            items.append((pool, cand, [] if a.no_state else state))
        if a.onpolicy_beam:
            loss = onpolicy_beam_loss(model, items, a.onpolicy_beam, cls_id, sep_id, pad_id, dev, a.accum)
            run.append(loss)
            if (bstart // a.bs + 1) % a.accum == 0 or bstart + a.bs >= total_items:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
                it += 1
                if it % 100 == 0:
                    el = time.time() - t0
                    print(f"step {it}/{steps} loss {np.mean(run[-400:]):.4f} {el/60:.1f}min eta {(steps-it)*el/it/60:.1f}min", flush=True)
            done = bstart + a.bs
            if a.epochs > 1 and done // len(pools) > (done - a.bs) // len(pools) and done < total_items:
                os.makedirs(f"{a.out}/ep{done // len(pools)}", exist_ok=True)
                torch.save({"state": model.state_dict(), "args": vars(a)}, f"{a.out}/ep{done // len(pools)}/model.pt")
            continue
        if head == "gap":
            sets, ys = gap_rows(items, rng)
            if a.descent_rows:
                s2, y2 = descent_rows(model, items, a.descent_rows, cls_id, sep_id, pad_id, dev)
                sets += s2; ys += y2
            ys = torch.tensor(ys, device=dev)
            n = len(sets)
            tot = 0.0
            for c0 in range(0, n, a.chunk):
                vi, va = collate_sets([build_set_seq(p, S, cls_id, sep_id) for p, S in sets[c0: c0 + a.chunk]], pad_id)
                with amp():
                    lp0, er, lpmf = model.gap_field(vi.to(dev), va.to(dev))
                l = gap_loss(model, lp0.float(), er.float(), lpmf.float(), ys[c0: c0 + a.chunk], a.target)
                w = (min(n, c0 + a.chunk) - c0) / n
                (l * w / a.accum).backward()
                tot += l.item() * w
            run.append(tot)
            if (bstart // a.bs + 1) % a.accum == 0 or bstart + a.bs >= total_items:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
                it += 1
                if it % 100 == 0:
                    el = time.time() - t0
                    print(f"step {it}/{steps} loss {np.mean(run[-400:]):.4f} {el/60:.1f}min eta {(steps-it)*el/it/60:.1f}min", flush=True)
            done = bstart + a.bs
            if a.epochs > 1 and done // len(pools) > (done - a.bs) // len(pools) and done < total_items:
                os.makedirs(f"{a.out}/ep{done // len(pools)}", exist_ok=True)
                torch.save({"state": model.state_dict(), "args": vars(a)}, f"{a.out}/ep{done // len(pools)}/model.pt")
                print("saved epoch", done // len(pools), flush=True)
            continue
        if a.no_state:
            enc_items = [(p, c, []) for p, c, s in items]
            ids, attn, mask, _ = collate(enc_items, cls_id, sep_id, pad_id)
            _, _, _, sel = collate(items, cls_id, sep_id, pad_id)
        else:
            ids, attn, mask, sel = collate(items, cls_id, sep_id, pad_id)
        ids, attn, mask, sel = ids.to(dev), attn.to(dev), mask.to(dev), sel.to(dev)
        with amp():
            out = model(ids, attn, mask, sel)
        if head.startswith("point"):
            loss = point_loss(out, items, dev)
        elif head == "ladder" and a.v4:
            loss = v4_loss(model, out, items, rng, dev, a, (cls_id, sep_id, pad_id))
        elif head == "ladder" and (a.inc or a.pad or a.set_target != "ordinal" or a.prop != "bce" or a.open_mix or a.rungs != R_DEFICIT):
            # ProofGap v3: one deficit head read from two serialisations of sets, summed losses without weights
            loss = point_loss(out, items, dev) if a.prop == "bce" else torch.zeros((), device=dev)
            if a.inc:
                loss = loss + inc_loss(model, out, items, dev, a.set_target)
            mine = mine_order(model, out, items, use_inc=a.inc)
            sets, ys = ladder_rows(mine, items, rng, a.n_cf, a.pad, a.random_cf, model.R)
            vi, va = collate_sets([build_set_seq(p, S, cls_id, sep_id) for p, S in sets], pad_id)
            with amp():
                o = model.set_reading(vi.to(dev), va.to(dev)).float()
            loss = loss + set_target_loss(o if a.set_target == "multiclass" else cum_logits(o), torch.tensor(ys, device=dev), a.set_target, model.R)
        elif head == "ladder":
            loss = point_loss(out, items, dev)
            sets, ys = deficit_sets(out, items, rng, a.n_cf, a.random_cf)
            vi, va = collate_sets([build_set_seq(p, S, cls_id, sep_id) for p, S in sets], pad_id)
            with amp():
                u = model.ladder_logits(vi.to(dev), va.to(dev))
            loss = loss + ladder_nll(u.float(), torch.tensor(ys, device=dev))
        elif head == "deficit":
            loss = point_loss(out, items, dev)
            sets, ys = deficit_sets(out, items, rng, a.n_cf, a.random_cf)
            vi, va = collate_sets([build_set_seq(p, S, cls_id, sep_id) for p, S in sets], pad_id)
            with amp():
                f = model.deficit_energy(vi.to(dev), va.to(dev))
            loss = loss + ordinal_nll(f.float(), model.cutpoints(), torch.tensor(ys, device=dev))
        elif head == "prover":
            loss = point_loss(out, items, dev)
            if not a.no_verdict:
                sets, ys = counterfactual_sets(out, items, rng, a.n_cf, a.random_cf, a.superset)
                vi, va = collate_sets([build_set_seq(p, S, cls_id, sep_id) for p, S in sets], pad_id)
                with amp():
                    v = model.verdict(vi.to(dev), va.to(dev))
                loss = loss + F.binary_cross_entropy_with_logits(v.float(), torch.tensor(ys, device=dev))
        else:
            loss = torch.zeros((), device=dev)
            if not a.no_setloss:
                loss = loss + set_loss(out, items, dev)
            if head == "erc" and not a.no_assign:
                loss = loss + assign_loss(out, items, dev)
            if head == "erc2" and not a.no_assign:
                loss = loss + assign_loss2(out, items, dev)
        (loss / a.accum).backward()
        run.append(loss.item())
        if (bstart // a.bs + 1) % a.accum == 0 or bstart + a.bs >= total_items:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            it += 1
            if it % log_every == 0:
                el = time.time() - t0
                peak = f" peak {torch.cuda.max_memory_allocated() / 2 ** 20:.0f}/{torch.cuda.max_memory_reserved() / 2 ** 20:.0f}MiB" if DEV == "cuda" else ""
                print(f"step {it}/{steps} loss {np.mean(run[-400:]):.4f} {el/60:.1f}min eta {(steps-it)*el/it/60:.1f}min{peak}", flush=True)
        done = bstart + a.bs
        if a.epochs > 1 and done // len(pools) > (done - a.bs) // len(pools) and done < total_items:
            os.makedirs(f"{a.out}/ep{done // len(pools)}", exist_ok=True)
            torch.save({"state": model.state_dict(), "args": vars(a)}, f"{a.out}/ep{done // len(pools)}/model.pt")
            print("saved epoch", done // len(pools), flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save({"state": model.state_dict(), "args": vars(a)}, f"{a.out}/model.pt")
    if os.path.exists(resume_path):
        os.remove(resume_path)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
