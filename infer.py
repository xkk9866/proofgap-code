"""Budgeted evidence-set selection with a trained model.

Set heads (erc/additive/pairwise): posterior-circuit greedy, each step re-evaluates the circuit
conditioned on the evidence already selected and adds argmax_d [log C(S+d) - log C(S)] / cost(d).
Point heads: greedy (beam=1) or beam search over next-evidence log-probabilities (Beam Retrieval).
Output jsonl: {id, order: [...], logC: [...]} for count budgets and optionally token budgets.
"""
import argparse
import collections
import json
import math
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from data import Pool, build_set_seq, build_view_seqs, collate, collate_sets, load_jsonl
from decide import or_decide, pi_decide
from model import NRUNG, R_DEFICIT, ERCNet, resolve_enc

DATA = os.environ.get("PROOFGAP_DATA_DIR", "data/proc")
# Inference runs on the GPU when one is visible and otherwise on the CPU (fp32; bf16 autocast only on CUDA).  The
# CPU path exists because the GPUs of the shared server are often full; the outputs are identical up to numerics.
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "32")))


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16) if DEV == "cuda" else torch.autocast("cpu", enabled=False)


@torch.no_grad()
def run_model(model, items, ids_fn, dev, no_state=False):
    cls_id, sep_id, pad_id = ids_fn
    if no_state:
        ids, attn, mask, _ = collate([(p, c, []) for p, c, s in items], cls_id, sep_id, pad_id)
        _, _, _, sel = collate(items, cls_id, sep_id, pad_id)
    else:
        ids, attn, mask, sel = collate(items, cls_id, sep_id, pad_id)
    with amp():
        return model(ids.to(dev), attn.to(dev), mask.to(dev), sel.to(dev))


def greedy(model, pools, kmax, dev, ids_fn, bs, costs=None, budget=None, no_state=False):
    """Greedy selection for a batch of pools. If budget is given, use cost-aware gain ratio."""
    res = []
    for s in range(0, len(pools), bs):
        batch = pools[s: s + bs]
        states = [[] for _ in batch]
        logcs = [[] for _ in batch]
        spent = [0 for _ in batch]
        active = list(range(len(batch)))
        for step in range(kmax):
            if not active:
                break
            items = [(batch[i], list(range(len(batch[i].d))), states[i]) for i in active]
            out = run_model(model, items, ids_fn, dev, no_state)
            new_active = []
            for bi, i in enumerate(active):
                n = len(batch[i].d)
                if "logC_add" in out:
                    gain = (out["logC_add"][bi, :n] - out["logC"][bi]).float()
                    val = out["logC_add"][bi, :n].float()
                else:
                    gain = F.logsigmoid(out["score"][bi, :n].float())
                    val = gain
                if budget is not None:
                    c = torch.tensor(costs[s + i], dtype=torch.float, device=gain.device)
                    gain = gain.clamp(min=1e-9) / c if "logC_add" in out else gain - torch.log(c)
                    ok = c <= (budget - spent[i])
                else:
                    ok = torch.ones(n, dtype=torch.bool, device=gain.device)
                for d in states[i]:
                    ok[d] = False
                if not ok.any():
                    continue
                gain = gain.masked_fill(~ok, -1e9)
                d = int(gain.argmax())
                states[i].append(d)
                logcs[i].append(float(val[d]))
                if budget is not None:
                    spent[i] += costs[s + i][d]
                if len(states[i]) < n:
                    new_active.append(i)
            active = new_active
        for i, p in enumerate(batch):
            res.append({"id": p.ex["id"], "order": states[i], "logC": logcs[i]})
    return res


def search(model, pools, kmax, width, dev, ids_fn, bs, verify=True, no_state=False, accum=True):
    """Set-level search with per-depth answers.

    Set heads (Circuit-Guided Proof Search): a node is an evidence state S.  Its posterior circuit Phi_S
    (one forward pass) yields both the *verdict* log C_{Phi_S}(S) and the *proposals* log C_{Phi_S}(S+d).
    Children are ranked by proposals (prior prediction), the kept W children are read (forward pass) and the
    budget-k answer is the depth-k node with the highest posterior verdict.  width=1 is posterior greedy.
    Point heads: standard beam search on summed next-evidence log-probabilities (Beam Retrieval).
    """
    res = []
    for s in range(0, len(pools), bs):
        batch = pools[s: s + bs]
        nodes = [[([], 0.0)] for _ in batch]
        sets = [{} for _ in batch]
        vers = [{} for _ in batch]
        preds = [{} for _ in batch]
        is_set = None
        for depth in range(kmax + 1):
            items, owner = [], []
            for qi, nd in enumerate(nodes):
                for ni, (st, _) in enumerate(nd):
                    if len(st) < len(batch[qi].d):
                        items.append((batch[qi], list(range(len(batch[qi].d))), st)); owner.append((qi, ni))
            if not items:
                break
            out = run_model(model, items, ids_fn, dev, no_state)
            is_set = "logC_add" in out
            if not is_set and depth == kmax:
                break
            children = [dict() for _ in batch]
            for row, (qi, ni) in enumerate(owner):
                st, sc = nodes[qi][ni]
                n = len(batch[qi].d)
                if is_set and depth > 0:
                    v = float(out["logC"][row])
                    if depth not in vers[qi] or (verify and v > vers[qi][depth]):
                        vers[qi][depth], sets[qi][depth], preds[qi][depth] = v, list(st), sc
                if depth == kmax:
                    continue
                if is_set:
                    val = out["logC_add"][row, :n].float()
                else:
                    val = (sc if accum else 0.0) + F.logsigmoid(out["score"][row, :n].float())
                val = val.clone()
                val[st] = -1e9
                top = torch.topk(val, min(width, n - len(st))).indices.tolist()
                for d in top:
                    key = frozenset(st + [d])
                    if key not in children[qi] or float(val[d]) > children[qi][key][1]:
                        children[qi][key] = (st + [d], float(val[d]))
            for qi in range(len(batch)):
                if not children[qi]:
                    nodes[qi] = []
                    continue
                kept = sorted(children[qi].values(), key=lambda x: -x[1])[:width]
                nodes[qi] = kept
                if not is_set:
                    sets[qi][depth + 1], preds[qi][depth + 1] = kept[0][0], kept[0][1]
        for qi, p in enumerate(batch):
            ks = sorted(sets[qi])
            res.append({"id": p.ex["id"], "sets": {str(k): sets[qi][k] for k in ks},
                        "verdict": {str(k): vers[qi][k] for k in sorted(vers[qi])},
                        "pred": {str(k): preds[qi][k] for k in ks},
                        "order": sets[qi][ks[-1]] if ks else []})
    return res


def static_circuit(model, pools, kmax, dev, ids_fn, bs, costs=None, budget=None):
    """ERC-static: one forward pass at S = {} gives (z, A); greedy maximisation of the closed-form
    log-submodular objective log C(S) (cost-benefit ratio under a token budget, plus best singleton)."""
    res = []
    for s in range(0, len(pools), bs):
        batch = pools[s: s + bs]
        out = run_model(model, [(p, list(range(len(p.d))), []) for p in batch], ids_fn, dev)
        for bi, p in enumerate(batch):
            n = len(p.d)
            l1a = F.logsigmoid(-out["logit_a"][bi, :, :n].float())  # [m,n]
            logz = F.logsigmoid(out["logit_z"][bi].float())
            logP = torch.zeros_like(logz)
            f = lambda lp: torch.log(-torch.expm1((logz + lp).clamp(max=-1e-6))).sum()
            cur, S, lc, spent = f(logP), [], [], 0
            c = torch.tensor(costs[s + bi], dtype=torch.float, device=l1a.device) if budget else torch.ones(n, device=l1a.device)
            for _ in range(kmax if budget is None else n):
                newP = logP.unsqueeze(-1) + l1a  # [m,n]
                vals = torch.log(-torch.expm1((logz.unsqueeze(-1) + newP).clamp(max=-1e-6))).sum(0)
                gain = (vals - cur) / c
                ok = torch.ones(n, dtype=torch.bool, device=gain.device)
                ok[S] = False
                if budget is not None:
                    ok &= c <= budget - spent
                if not ok.any():
                    break
                d = int(gain.masked_fill(~ok, -1e9).argmax())
                S.append(d); lc.append(float(vals[d])); spent += float(c[d]) if budget else 0
                logP, cur = newP[:, d], vals[d]
            if budget is not None:  # best-singleton safeguard of budgeted submodular greedy
                single = torch.log(-torch.expm1((logz.unsqueeze(-1) + l1a).clamp(max=-1e-6))).sum(0)
                single = single.masked_fill(c > budget, -1e9)
                if lc and float(single.max()) > lc[-1]:
                    S, lc = [int(single.argmax())], [float(single.max())]
            res.append({"id": p.ex["id"], "order": S, "logC": lc})
    return res


def beam(model, pools, kmax, width, dev, ids_fn):
    res = []
    for p in pools:
        n = len(p.d)
        beams = [([], 0.0)]
        for step in range(min(kmax, n)):
            items = [(p, list(range(n)), st) for st, _ in beams]
            out = run_model(model, items, ids_fn, dev)
            lp = F.logsigmoid(out["score"][:, :n].float())
            cand = []
            for bi, (st, sc) in enumerate(beams):
                for d in range(n):
                    if d not in st:
                        cand.append((st + [d], sc + float(lp[bi, d])))
            cand.sort(key=lambda x: -x[1])
            seen, beams = set(), []
            for st, sc in cand:
                key = frozenset(st)
                if key in seen:
                    continue
                seen.add(key); beams.append((st, sc))
                if len(beams) == width:
                    break
        res.append({"id": p.ex["id"], "order": beams[0][0], "logC": []})
    return res


@torch.no_grad()
def verdicts(model, pool, sets, ids_fn, dev, bs=32, fn=None):
    cls_id, sep_id, pad_id = ids_fn
    out = []
    for i in range(0, len(sets), bs):
        ids, attn = collate_sets([build_set_seq(pool, sorted(S), cls_id, sep_id) for S in sets[i: i + bs]], pad_id)
        with amp():
            if fn is not None:
                out.append(fn(ids.to(dev), attn.to(dev)).float())
            else:
                out.append(F.logsigmoid(model.verdict(ids.to(dev), attn.to(dev)).float()))
    return torch.cat(out).tolist()


def prover_search(model, pools, kmax, width, expand, dev, ids_fn, bs, use_verdict=True, frontier="chain", decide="or"):
    """Propose-then-verify set search.

    Proposal: a lattice over evidence *sets*.  Each frontier set S is read once by the state-conditioned
    proposer; its `expand` best completions S+{d} become depth-(|S|+1) candidates scored by the chain
    log-likelihood, and different orderings reaching the same set are merged (log-sum-exp), so the
    lattice scores sets, not sequences.  The `width` best sets form the next frontier.
    Decision: for every budget k, all depth-k candidates are read jointly by the verdict head and the
    set with the highest P(complete | q, S) is returned.  use_verdict=False returns the best-scoring set
    under the proposal score (beam-search decision, ablation).
    frontier="deficit": the frontier keeps the sets with the lowest proof deficit (highest verdict rung)
    instead of the highest chain score, so search and decision optimise the same quantity.
    decide="verdict": argmax of the verdict itself (the deficit ladder is trained on supersets), "or": noisy-OR.
    """
    res = []
    # NOTE (bug fixed 2026-11-05): this assignment used to sit inside the batch loop, after which `frontier` was
    # re-bound to a list, so every batch but the first silently fell back to the chain frontier.
    frontier_mode = frontier
    for s in range(0, len(pools), bs):
        batch = pools[s: s + bs]
        frontier = [{frozenset(): 0.0} for _ in batch]
        cands = [dict() for _ in batch]
        vcache = [dict() for _ in batch]
        for depth in range(kmax):
            items, owner = [], []
            for qi, fr in enumerate(frontier):
                for st in fr:
                    if len(st) < len(batch[qi].d):
                        items.append((batch[qi], list(range(len(batch[qi].d))), sorted(st))); owner.append((qi, st))
            if not items:
                break
            out = run_model(model, items, ids_fn, dev)
            nxt = [collections.defaultdict(list) for _ in batch]
            for row, (qi, st) in enumerate(owner):
                n = len(batch[qi].d)
                lp = F.logsigmoid(out["score"][row, :n].float()).clone()
                lp[list(st)] = -1e9
                for d in torch.topk(lp, min(expand, n - len(st))).indices.tolist():
                    nxt[qi][st | {d}].append(frontier[qi][st] + float(lp[d]))
            for qi in range(len(batch)):
                merged = {S: float(torch.logsumexp(torch.tensor(v), 0)) for S, v in nxt[qi].items()}
                cands[qi][depth + 1] = merged
                if frontier_mode == "deficit" and merged:
                    keys = list(merged)
                    v = verdicts(model, batch[qi], keys, ids_fn, dev)
                    vcache[qi].update(zip(keys, v))
                    if model.head_type == "ladder":  # least expected deficit
                        ed = dict(zip(keys, verdicts(model, batch[qi], keys, ids_fn, dev, fn=model.expected_deficit)))
                        order = sorted(keys, key=lambda S: (ed[S], -merged[S]))
                    else:
                        order = sorted(keys, key=lambda S: (-vcache[qi][S], -merged[S]))
                else:
                    order = sorted(merged, key=lambda S: -merged[S])
                frontier[qi] = {S: merged[S] for S in order[:width]}
        for qi, p in enumerate(batch):
            sets, ver, prop, allc = {}, {}, {}, {}
            for k, cd in cands[qi].items():
                keys = list(cd)
                if not keys:
                    continue
                if use_verdict:
                    todo = [S for S in keys if S not in vcache[qi]]
                    if todo:
                        vcache[qi].update(zip(todo, verdicts(model, p, todo, ids_fn, dev)))
                    v = [vcache[qi][S] for S in keys]
                else:
                    v = [0.0] * len(keys)
                best = max(range(len(keys)), key=(lambda i: (v[i], cd[keys[i]])) if use_verdict else (lambda i: cd[keys[i]]))
                sets[str(k)], ver[str(k)], prop[str(k)] = sorted(keys[best]), v[best], cd[keys[best]]
                allc[str(k)] = [[sorted(S), round(vv, 4), round(cd[S], 4)] for S, vv in zip(keys, v)]
            if use_verdict and allc and decide == "or":  # proof-disjunctive decision (decide.py)
                sets, ver, prop = or_decide(allc)
            ks = sorted(int(k) for k in sets)
            res.append({"id": p.ex["id"], "sets": sets, "verdict": ver, "pred": prop, "cands": allc,
                        "order": sets[str(ks[-1])] if ks else []})
    return res


@torch.no_grad()
def read_sets(model, reqs, ids_fn, dev, bs=32):
    """Holistic deficit ladders (cumulative logits, [n, R]) of whole sets read jointly; reqs = [(pool, set)]."""
    cls_id, sep_id, pad_id = ids_fn
    outs = []
    for i in range(0, len(reqs), bs):
        ids, attn = collate_sets([build_set_seq(p, sorted(S), cls_id, sep_id) for p, S in reqs[i: i + bs]], pad_id)
        with amp():
            outs.append(model.ladder_logits(ids.to(dev), attn.to(dev)).float().cpu())
    return torch.cat(outs) if outs else torch.zeros(0, R_DEFICIT)


def deficit_key(u, how="u1"):
    """Search priority of a set from its deficit ladder (higher = closer to a complete proof).

    er : minus the expected truncated deficit E[min(r, R-1)] (default).  All sets on one lattice depth have the same
         size t, so this ranks by t + E[r], the expected size of the smallest proof-containing extension of the set:
         the A* priority with the learned deficit as the distance to the goal;
    u1 : logit P(r(S) <= 1), one-step reachability (myopic);
    u0 : logit P(r(S) = 0), the completeness rung alone (all that a binary verdict provides).
    u holds the R cumulative logits; the last one belongs to the tail class and carries no information."""
    if how == "u1":
        return float(u[1])
    if how == "u0":
        return float(u[0])
    return -float(torch.sigmoid(-u[:-1]).sum())


def proof_search(model, pools, kmax, width, expand, dev, ids_fn, bs, cand="ladder", front="ladder", how="er", decide="or"):
    """Deficit-guided proof search (ProofGap inference).

    The set lattice is searched with the learned proof deficit as its distance-to-goal:
      depth t   every frontier set S is encoded once with every candidate d as [q; S; d]; the [CLS] vector gives the
                proposal logit and, through the same ladder head, the *incremental* deficit ladder of the child S+{d}.
      propose   the candidate budget M = width*expand is shared equally by the frontier sets; each proposes its children
                with the best incremental priority (cand="ladder": the incremental reading ranks WITHIN a parent, the
                holistic reading below compares ACROSS parents); cand="chain" ranks within the parent by the proposal logit
                (same budget); cand="global" lets all children of all parents compete on the incremental reading.
      verify    every candidate is read once *holistically* as [q; s_1; ...; s_k] (singletons: the two serialisations
                coincide, so their reading is reused).
      frontier  the `width` candidates with the highest holistic priority (front="ladder") or proposal mass ("chain").
      decide    budget k: proof-disjunctive pooling of the completeness rung over all candidates contained in a
                depth-k candidate (decide="or", decide.or_decide) or the completeness rung alone (decide="v0").
    Everything that ranks sets is a reading of the same quantity, the proof-deficit distribution.  The output keeps
    every candidate ([set, log V, proposal, holistic ladder, incremental ladder]) and the cumulative number of encoder
    sequences after every depth, so that stopping rules can be evaluated offline (stopping.py).
    """
    res = []
    M = width * expand
    for s in range(0, len(pools), bs):
        batch = pools[s: s + bs]
        fronts = [[frozenset()] for _ in batch]
        lat = [dict() for _ in batch]
        cost = [dict() for _ in batch]
        nseq = [0 for _ in batch]
        for depth in range(kmax):
            items, owner = [], []
            for qi, fr in enumerate(fronts):
                for st in fr:
                    if len(st) < len(batch[qi].d):
                        items.append((batch[qi], list(range(len(batch[qi].d))), sorted(st))); owner.append((qi, st))
                        nseq[qi] += len(batch[qi].d)
            if not items:
                break
            out = run_model(model, items, ids_fn, dev)
            lps = F.logsigmoid(out["score"].float()).cpu()
            uI = model.ladder_from_h(out["h"]).float().cpu()
            kids = [dict() for _ in batch]
            nfr = collections.Counter(qi for qi, _ in owner)
            for row, (qi, st) in enumerate(owner):
                n = len(batch[qi].d)
                pc = lat[qi][st]["chain"] if st else 0.0
                lp = lps[row, :n]
                # each frontier set proposes an equal share of the M candidates (identical budgets for both signals)
                share = -(-M // nfr[qi])
                if cand == "chain":  # ranked within the parent by the proposal logit (chain retrievers)
                    l2 = lp.clone(); l2[list(st)] = -1e9
                    ds = torch.topk(l2, min(share, n - len(st))).indices.tolist()
                elif cand == "ladder":
                    # ranked WITHIN the parent by the incremental deficit of its children; comparisons ACROSS parents
                    # are left to the holistic reading
                    key = torch.tensor([deficit_key(uI[row, d], how) if d not in st else -1e9 for d in range(n)])
                    ds = torch.topk(key, min(share, n - len(st))).indices.tolist()
                else:  # "global": all children of all parents compete on the incremental reading (ablation)
                    ds = [d for d in range(n) if d not in st]
                for d in ds:
                    e = kids[qi].setdefault(st | {d}, ([], []))
                    e[0].append(pc + float(lp[d])); e[1].append(uI[row, d])
            hreq, cands_q = [], []
            for qi in range(len(batch)):
                merged = {T: (float(torch.logsumexp(torch.tensor(ch), 0)), torch.stack(us).mean(0)) for T, (ch, us) in kids[qi].items()}
                if cand in ("chain", "ladder", "member"):
                    C = list(merged)
                else:
                    C = sorted(merged, key=lambda T: (deficit_key(merged[T][1], how), merged[T][0]), reverse=True)[:M]
                for T in C:
                    lat[qi][T] = {"chain": merged[T][0], "uI": merged[T][1], "depth": depth + 1}
                    if depth == 0:
                        lat[qi][T]["uH"] = merged[T][1]
                    else:
                        hreq.append((qi, T))
                cands_q.append(C)
            if hreq:
                U = read_sets(model, [(batch[qi], T) for qi, T in hreq], ids_fn, dev)
                for (qi, T), u in zip(hreq, U):
                    lat[qi][T]["uH"] = u
                    nseq[qi] += 1
            for qi in range(len(batch)):
                C = cands_q[qi]
                if front == "chain":
                    order = sorted(C, key=lambda T: lat[qi][T]["chain"], reverse=True)
                else:
                    order = sorted(C, key=lambda T: (deficit_key(lat[qi][T]["uH"], how), lat[qi][T]["chain"]), reverse=True)
                fronts[qi] = order[:width]
                cost[qi][str(depth + 1)] = nseq[qi]
        for qi, p in enumerate(batch):
            allc = {}
            for T, v in lat[qi].items():
                allc.setdefault(str(v["depth"]), []).append(
                    [sorted(T), round(float(F.logsigmoid(v["uH"][0])), 4), round(v["chain"], 4),
                     [round(float(x), 3) for x in v["uH"][:-1]], [round(float(x), 3) for x in v["uI"][:-1]]])
            if decide == "or":
                sets, ver, prop = or_decide(allc)
            else:
                sets, ver, prop = {}, {}, {}
                for k, L in allc.items():
                    b = max(L, key=lambda x: (x[1], x[2]))
                    sets[k], ver[k], prop[k] = b[0], b[1], b[2]
            lad = {k: next(x[3] for x in allc[k] if x[0] == sets[k]) for k in sets}
            ks = sorted(int(k) for k in sets)
            res.append({"id": p.ex["id"], "sets": sets, "verdict": ver, "pred": prop, "ladder": lad, "cands": allc,
                        "cost": cost[qi], "order": sets[str(ks[-1])] if ks else []})
    return res


@torch.no_grad()
def read_views(model, reqs, ids_fn, dev, bs=32):
    """Member-focused views [q; T minus t; t] -> (cumulative deficit logits [n, R], necessity logits [n]).
    reqs = [(pool, T, t)].  The necessity logit is the second reading of the same view: is t one of the annotated
    passages, i.e. does removing t from T raise the deficit by one?"""
    cls_id, sep_id, pad_id = ids_fn
    outs, necs = [], []
    for i in range(0, len(reqs), bs):
        seqs = [build_view_seqs(p, sorted(T), cls_id, sep_id, focus=[t])[0] for p, T, t in reqs[i: i + bs]]
        ids, attn = collate_sets(seqs, pad_id)
        with amp():
            o, ml = model.view_heads(ids.to(dev), attn.to(dev))
        outs.append(model.to_cum(o.float()).cpu()); necs.append(ml.float().cpu())
    if not outs:
        return torch.zeros(0, model.R), torch.zeros(0)
    return torch.cat(outs), torch.cat(necs)


def focus_search(model, pools, kmax, width, expand, dev, ids_fn, bs, cand="ladder", front="ladder", views="consensus",
                 decide="or", how="er", exchange=False):
    """ProofGap inference: proof search in which expanding a set and verifying a set are the same reading.

      extend    every frontier set S is read once with every candidate d as [q; S; d] (data.build_seqs): this is the
                view of the child S + {d} through its new member; its [CLS] gives the child's deficit ladder and the
                necessity of d.  The candidate budget M = width*expand is shared equally by the extended sets; each
                proposes the children with the smallest expected deficit; children reached twice are merged.
      complete  every new candidate T is completed to its full set of member views [q; T minus t; t]; its ladder is the
                consensus (mean cumulative logits) of the |T| views, and every view also gives the necessity of its
                member.  views="holistic" reads T once as [q; t_1; ...; t_k] instead (control).
      frontier  the `width` candidates with the smallest consensus expected deficit (front="pooled": deficit scaled by the
                completeness pooled over the candidate proofs the set contains).
      exchange  (exchange=True) each frontier set T of size >= 2 gives up its least necessary member t*: the context
                T minus t* is extended exactly like a frontier set, so the search can replace a member it chose earlier.
                The siblings it produces compete with every other candidate of the same size for the frontier.
      decide    budget k: the completeness rung P(r = 0) of the consensus, pooled over the candidate proofs contained
                in a depth-k candidate (decide="or"), or of the set alone ("v0").
    Output per question: decided sets, their scores, the ladders of the decided sets, every candidate
    ([set, log P0, chain, consensus ladder, mean child-view ladder, rung-0 logit of every view, necessity of every
    member]) and the cumulative encoder sequences per depth.
    """
    res = []
    M, R = width * expand, model.R
    for s in range(0, len(pools), bs):
        batch = pools[s: s + bs]
        fronts = [[frozenset()] for _ in batch]
        lat = [dict() for _ in batch]
        cost = [dict() for _ in batch]
        nseq = [0 for _ in batch]
        single = [None for _ in batch]  # depth-0 reading of every passage alone, [N, R] (cand="pair")
        expanded = [set() for _ in batch]
        cmass = [dict() for _ in batch]  # proposal mass of exchange contexts that were never candidates themselves

        def grow(ctxs):
            """Extend every context set of ctxs[qi] by every unused passage, complete the new candidates' views."""
            items, owner = [], []
            for qi, cl in enumerate(ctxs):
                for st in cl:
                    if st in expanded[qi] or len(st) >= len(batch[qi].d):
                        continue
                    expanded[qi].add(st)
                    items.append((batch[qi], list(range(len(batch[qi].d))), sorted(st))); owner.append((qi, st))
                    nseq[qi] += len(batch[qi].d)
            if not items:
                return False
            out = run_model(model, items, ids_fn, dev)
            sc = out["score"].float().cpu()
            lps = F.logsigmoid(sc)
            uI = model.ladder_from_h(out["h"]).float().cpu()  # [rows, N, R]: ladder of every child through its new member
            kids = [dict() for _ in batch]
            nfr = collections.Counter(qi for qi, _ in owner)
            for row, (qi, st) in enumerate(owner):
                n = len(batch[qi].d)
                if not st:
                    single[qi] = uI[row, :n].clone()
                pc = lat[qi][st]["chain"] if st in lat[qi] else cmass[qi].get(st, 0.0)
                lp = lps[row, :n]
                share = -(-M // nfr[qi])
                if cand in ("chain", "member"):
                    l2 = lp.clone(); l2[list(st)] = -1e9
                    ds = torch.topk(l2, min(share, n - len(st))).indices.tolist()
                else:
                    u = uI[row, :n, : R - 1]
                    if cand == "pair" and st:
                        u = 0.5 * (u + single[qi][:, : R - 1])
                    er = torch.sigmoid(-u).sum(1)
                    if cand == "edit":  # expected distance of the child: passages still missing + is the new member superfluous
                        er = er + torch.sigmoid(-sc[row, :n])
                    key = torch.tensor([-float(er[d]) if d not in st else -1e9 for d in range(n)])
                    ds = torch.topk(key, min(share, n - len(st))).indices.tolist()
                for d in ds:
                    e = kids[qi].setdefault(st | {d}, ([], {}, {}))
                    e[0].append(pc + float(lp[d])); e[1][d] = uI[row, d]; e[2][d] = float(sc[row, d])
            vreq, new = [], [[] for _ in batch]
            for qi in range(len(batch)):
                for T, (ch, vw, nc) in kids[qi].items():
                    if T in lat[qi]:  # reached again through an exchange: merge the proposal mass
                        lat[qi][T]["chain"] = float(torch.logsumexp(torch.tensor([lat[qi][T]["chain"]] + ch), 0))
                        continue
                    lat[qi][T] = {"chain": float(torch.logsumexp(torch.tensor(ch), 0)), "views": dict(vw), "nec": dict(nc),
                                  "kid": torch.stack(list(vw.values())).mean(0), "depth": len(T)}
                    new[qi].append(T)
                    if views == "consensus":
                        vreq += [(qi, T, t) for t in sorted(T) if t not in vw]
                    elif len(T) > 1:
                        vreq.append((qi, T, None))
            if vreq:
                if views == "consensus":
                    U, Nc = read_views(model, [(batch[qi], T, t) for qi, T, t in vreq], ids_fn, dev)
                else:
                    U, Nc = read_sets(model, [(batch[qi], T) for qi, T, _ in vreq], ids_fn, dev), None
                for i, (qi, T, t) in enumerate(vreq):
                    nseq[qi] += 1
                    if views == "consensus":
                        lat[qi][T]["views"][t] = U[i]; lat[qi][T]["nec"][t] = float(Nc[i])
                    else:
                        lat[qi][T]["holo"] = U[i]
            for qi in range(len(batch)):
                for T in new[qi]:
                    e = lat[qi][T]
                    e["u"] = e["holo"] if "holo" in e else torch.stack(list(e["views"].values())).mean(0)
            return True

        def frontier(qi, size):
            C = [T for T, v in lat[qi].items() if v["depth"] == size]
            if front == "chain":
                order = sorted(C, key=lambda T: lat[qi][T]["chain"], reverse=True)
            elif front == "pooled":
                # the completeness a set inherits from the candidate proofs it contains also drives the frontier:
                # E~[r] = E[r] (1 - P~) / (1 - P0); partial sets (no contained proof) keep the consensus E[r]
                logp0 = {S: float(F.logsigmoid(v["u"][0])) for S, v in lat[qi].items()}

                def pooled_key(T):
                    l1m = sum(math.log1p(-min(math.exp(logp0[S]), 1 - 1e-6)) for S in logp0 if S <= T)
                    p_tilde, p0 = -math.expm1(l1m), min(math.exp(logp0[T]), 1 - 1e-6)
                    er = float(torch.sigmoid(-lat[qi][T]["u"][:-1]).sum())
                    return -er * (1 - p_tilde) / (1 - p0)
                order = sorted(C, key=lambda T: (pooled_key(T), lat[qi][T]["chain"]), reverse=True)
            elif front == "edit":
                # expected edit distance to the proof: E[missing passages] (consensus) + E[superfluous members] (one per view)
                def edit_key(T):
                    v = lat[qi][T]
                    return -(float(torch.sigmoid(-v["u"][:-1]).sum())
                             + float(torch.sigmoid(-torch.tensor(list(v["nec"].values()))).sum()))
                order = sorted(C, key=lambda T: (edit_key(T), lat[qi][T]["chain"]), reverse=True)
            else:
                order = sorted(C, key=lambda T: (deficit_key(lat[qi][T]["u"], how), lat[qi][T]["chain"]), reverse=True)
            return order[:width]

        for depth in range(kmax):
            if not grow(fronts):
                break
            fronts = [frontier(qi, depth + 1) for qi in range(len(batch))]
            if exchange and depth >= 1:
                ctxs = []
                for qi in range(len(batch)):
                    cl = []
                    for T in fronts[qi]:
                        nec = lat[qi][T]["nec"]
                        t_star = min(T, key=lambda t: nec.get(t, 0.0))
                        cl.append(T - {t_star})
                        cmass[qi].setdefault(T - {t_star}, lat[qi][T]["chain"])
                    ctxs.append(cl)
                if grow(ctxs):
                    fronts = [frontier(qi, depth + 1) for qi in range(len(batch))]
            for qi in range(len(batch)):
                cost[qi][str(depth + 1)] = nseq[qi]
        for qi, p in enumerate(batch):
            allc = {}
            for T, v in lat[qi].items():
                allc.setdefault(str(v["depth"]), []).append(
                    [sorted(T), round(float(F.logsigmoid(v["u"][0])), 4), round(v["chain"], 4),
                     [round(float(x), 3) for x in v["u"][:-1]], [round(float(x), 3) for x in v["kid"][:-1]],
                     {str(t): round(float(u[0]), 3) for t, u in v["views"].items()},
                     {str(t): round(x, 3) for t, x in v["nec"].items()}])
            extra = {}
            if decide == "or":
                sets, ver, prop = or_decide(allc)
            elif decide == "pi":
                sets, ver, prop, extra["map"], extra["map_logpi"] = pi_decide(allc)
            else:
                sets, ver, prop = {}, {}, {}
                for k, L in allc.items():
                    b = max(L, key=lambda x: (x[1], x[2]))
                    sets[k], ver[k], prop[k] = b[0], b[1], b[2]
            lad = {k: next(x[3] for x in allc[k] if x[0] == sets[k]) for k in sets}
            ks = sorted(int(k) for k in sets)
            res.append({"id": p.ex["id"], "sets": sets, "verdict": ver, "pred": prop, "ladder": lad, "cands": allc,
                        "cost": cost[qi], "order": sets[str(ks[-1])] if ks else [], **extra})
    return res


@torch.no_grad()
def gap_read(model, pool, seqs_sets, ids_fn, dev, chunk=64):
    """Read sets (lists in insertion order) with the proof-gap field -> (log P(r=0), E[r]) lists."""
    cls_id, sep_id, pad_id = ids_fn
    lp, er = [], []
    for i in range(0, len(seqs_sets), chunk):
        ids, attn = collate_sets([build_set_seq(pool, S, cls_id, sep_id) for S in seqs_sets[i: i + chunk]], pad_id)
        with amp():
            a, b, _ = model.gap_field(ids.to(dev), attn.to(dev))
        lp += a.float().tolist(); er += b.float().tolist()
    return lp, er


def gap_search(model, pools, kmax, width, dev, ids_fn, frontier="gap", decide="p0", keep=48):
    """Proof-gap descent: one field both proposes and decides.

    Depth t: every frontier set S is extended by every unused passage d and the field reads each S + {d} once,
    giving P(r=0 | S+d) and the expected gap E[r | S+d].  The `width` children with the smallest expected gap
    form the next frontier (descent on the learned proof distance).  The budget-k answer is the depth-k set with
    the highest P(r=0) among *all* children read at depth k.  No separate proposer, verdict pass or aggregation.
    frontier="p0": rank children by P(r=0) instead (ablation); decide="gap": answer by smallest E[r] (ablation).
    """
    res = []
    for p in pools:
        n = len(p.d)
        fr = [[]]
        sets, ver, pred, allc = {}, {}, {}, {}
        for depth in range(min(kmax, n)):
            child = {}
            todo = []
            for st in fr:
                for d in range(n):
                    if d not in st:
                        key = frozenset(st + [d])
                        if key not in child:
                            child[key] = None; todo.append(st + [d])
            lp, er = gap_read(model, p, todo, ids_fn, dev)
            for S, a, b in zip(todo, lp, er):
                child[frozenset(S)] = (S, a, b)
            vals = list(child.values())
            fkey = (lambda x: (x[2], -x[1])) if frontier == "gap" else (lambda x: (-x[1], x[2]))
            dkey = (lambda x: (-x[1], x[2])) if decide == "p0" else (lambda x: (x[2], -x[1]))
            fr = [x[0] for x in sorted(vals, key=fkey)[:width]]
            best = sorted(vals, key=dkey)[0]
            k = str(depth + 1)
            sets[k], ver[k], pred[k] = sorted(best[0]), best[1], -best[2]
            top = sorted(vals, key=lambda x: -x[1])[:keep]
            allc[k] = [[sorted(S), round(a, 4), round(-b, 4)] for S, a, b in top]
        ks = sorted(int(k) for k in sets)
        res.append({"id": p.ex["id"], "sets": sets, "verdict": ver, "pred": pred, "cands": allc,
                    "order": sets[str(ks[-1])] if ks else []})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True)  # e.g. musique_dev or open_musique
    ap.add_argument("--out", required=True)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--beam", type=int, default=0)
    ap.add_argument("--budgets", default="")  # token budgets, e.g. 256,512,1024
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--static", action="store_true", help="ERC-static: single circuit from S={}")
    ap.add_argument("--search", type=int, default=0, help="set-level search width (CGPS for set heads, beam for point)")
    ap.add_argument("--noverify", action="store_true", help="CGPS ablation: pick depth-k answer by prior prediction")
    ap.add_argument("--no_accum", action="store_true", help="official Beam Retrieval: rank sequences by their current score")
    ap.add_argument("--prover", type=int, default=0, help="propose-then-verify lattice width W (0 = off)")
    ap.add_argument("--expand", type=int, default=4, help="completions proposed per frontier set")
    ap.add_argument("--chain", action="store_true", help="lattice decision by proposal score instead of the verdict")
    ap.add_argument("--frontier", default="", help="chain | deficit (default: deficit for the deficit head)")
    ap.add_argument("--decide", default="", help="or | verdict (default: verdict for the deficit head); gap head: p0 | gap")
    ap.add_argument("--gap", type=int, default=0, help="proof-gap descent frontier width W (gap head)")
    ap.add_argument("--pg", type=int, default=0, help="ProofGap deficit-guided proof search, lattice width W (ladder head)")
    ap.add_argument("--cand", default="ladder", help="pg: candidates ladder (incremental deficit within each parent) | chain (proposal mass) | global")
    ap.add_argument("--front", default="ladder", help="pg: frontier ladder (holistic deficit) | chain (proposal mass)")
    ap.add_argument("--how", default="er", help="pg: deficit priority er (A*: expected deficit) | u1 | u0")
    ap.add_argument("--pg4", type=int, default=0, help="ProofGap v4 search (member-focused consensus reading), lattice width W")
    ap.add_argument("--views", default="consensus", help="pg4: consensus (read every candidate through each member) | holistic (v3 reading)")
    ap.add_argument("--exchange", action="store_true", help="pg4: each frontier set also re-extends the context left by its least necessary member")
    a = ap.parse_args()
    # Training-target ablations share the source-selected candidate rule with the main model.
    selection = os.path.join(os.path.dirname(a.out), 'g1_selection.json')
    controlled = ('pg_noopen', 'pg_mc', 'pg_bin', 'pg_w4')
    if a.pg and any(a.out.endswith(f'.{key}.jsonl') for key in controlled) and os.path.exists(selection):
        a.cand = json.load(open(selection))['candidate']
    ck = torch.load(f"{a.ckpt}/model.pt", map_location="cpu")
    ar = ck["args"]
    ar["enc"] = resolve_enc(ar["enc"])
    if ar.get("ctx_max"):  # v4 models: context passages of a view use the room up to this cap (as in training)
        import data as _data
        _data.set_ctx_max(ar["ctx_max"])
    tok = AutoTokenizer.from_pretrained(ar["enc"])
    head = "point_noset" if (ar["head"] == "point" and ar.get("no_set")) else ar["head"]
    model = ERCNet(ar["enc"], head=head, m=ar["m"], use_set=not ar.get("no_set"), hop_heads=ar.get("hop_heads", False),
                   n_rung=ar.get("rungs", R_DEFICIT)).to(DEV).eval()
    model.load_state_dict(ck["state"])
    model.sep_id = tok.sep_token_id
    model.ladder_kind = ar.get("set_target", "ordinal")
    if head == "gap":
        model.target = ar.get("target", "ordinal")
    a.kmax = min(a.kmax, 5) if (a.prover or a.gap or a.pg or a.pg4) else a.kmax
    pools = [Pool(e, tok) for e in load_jsonl(f"{DATA}/{a.split}.jsonl", a.limit)]
    ids_fn = (tok.cls_token_id, tok.sep_token_id, tok.pad_token_id)
    no_state = ar.get("no_state", False)
    rows = {}
    if a.pg4:
        for r in focus_search(model, pools, a.kmax, a.pg4, a.expand, DEV, ids_fn, a.bs, cand=a.cand, front=a.front,
                              views=a.views, decide=a.decide or "or", how=a.how, exchange=a.exchange):
            r['search_config'] = {'candidate': a.cand, 'frontier': a.front, 'views': a.views, 'width': a.pg4, 'decide': a.decide or 'or',
                                  'exchange': a.exchange}
            rows[r["id"]] = r
    elif a.pg:
        for r in proof_search(model, pools, a.kmax, a.pg, a.expand, DEV, ids_fn, a.bs, cand=a.cand, front=a.front,
                              how=a.how, decide=a.decide or "or"):
            r['search_config'] = {'candidate': a.cand, 'frontier': a.front, 'how': a.how, 'width': a.pg}
            rows[r["id"]] = r
    elif a.gap:
        for r in gap_search(model, pools, a.kmax, a.gap, DEV, ids_fn, frontier=a.frontier or "gap", decide=a.decide or "p0"):
            rows[r["id"]] = r
    elif a.prover:
        dfc = head in ("deficit", "ladder")
        fr = a.frontier or ("deficit" if dfc else "chain")
        dc = a.decide or "or"
        for r in prover_search(model, pools, a.kmax, a.prover, a.expand, DEV, ids_fn, a.bs, use_verdict=not a.chain,
                               frontier=fr, decide=dc):
            rows[r["id"]] = r
    elif a.search:
        for r in search(model, pools, a.kmax, a.search, DEV, ids_fn, a.bs, verify=not a.noverify, no_state=no_state, accum=not a.no_accum):
            rows[r["id"]] = r
    elif a.beam > 1:
        for r in beam(model, pools, a.kmax, a.beam, DEV, ids_fn):
            rows[r["id"]] = {"id": r["id"], "order": r["order"], "logC": r["logC"]}
    elif a.static:
        for r in static_circuit(model, pools, a.kmax, DEV, ids_fn, a.bs):
            rows[r["id"]] = r
    else:
        for r in greedy(model, pools, a.kmax, DEV, ids_fn, a.bs, no_state=no_state):
            rows[r["id"]] = r
    if a.budgets:
        costs = [[len(x) for x in p.d] for p in pools]
        for B in [int(x) for x in a.budgets.split(",")]:
            sel = (static_circuit(model, pools, 50, DEV, ids_fn, a.bs, costs=costs, budget=B) if a.static else
                   greedy(model, pools, 50, DEV, ids_fn, a.bs, costs=costs, budget=B, no_state=no_state))
            for r in sel:
                rows[r["id"]][f"tok{B}"] = r["order"]
                rows[r["id"]][f"tok{B}_logC"] = r["logC"]
    with open(a.out, "w") as f:
        for r in rows.values():
            f.write(json.dumps(r) + "\n")
    print("wrote", a.out, len(rows))


if __name__ == "__main__":
    main()
