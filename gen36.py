"""LaTeX tables of the paper from collect36.json (+ the few analysis numbers kept in MANUAL below).

python gen36.py results_pg4_collect36.json paper/erc/tab_auto.tex

Each table is written as a macro (\\tabmain, \\tabopen, ...).  A row whose result is missing is reported on stdout and
left out, so the paper never shows an empty cell.
"""
import json
import os
import sys

BEAM = r"Beam Retrieval$^\dagger$"  # retrained under the shared small budget, not the published 16-epoch schedule
DAGGER_FULL = r" $^\dagger$Retrained under the shared protocol of \S\ref{sec:setup}; with its published 16-epoch schedule its paper reports 75.47 Set-EM on MuSiQue."
DAGGER = r" $^\dagger$Retrained at the budget of \S\ref{sec:setup}; see the note of Table~\ref{tab:main}."  # every other table that shows the row
NAMES = [("dense", "bge-large-en-v1.5"), ("rerank", "bge-reranker-v2-m3"), ("qwen3rr", "Qwen3-Reranker-4B"), ("rankgpt", "RankGPT"),
         ("ircot", "IRCoT"), ("llmsetfill", "SetR"), ("beam", BEAM), ("PG", r"\textbf{\method{}}")]
MISSING = []


def fmt(v, best, second, d=1):
    s = f"{v:.{d}f}"
    if abs(v - best) < 1e-9:
        return rf"\textbf{{{s}}}"
    if second is not None and abs(v - second) < 1e-9:
        return rf"\underline{{{s}}}"
    return s


def block(J, split, cols, keys=None, get=None):
    """rows of one dataset block: [(name, [values])], with the best / second-best of every column marked."""
    rows = []
    for key, name in (keys or NAMES):
        m = J["ret"].get(f"{split}|{key}")
        if m is None:
            MISSING.append(f"{split}|{key}")
            continue
        rows.append((name, [get(m, c) if get else m[c] for c in cols]))
    out = []
    for name, vals in rows:
        cells = []
        for j, v in enumerate(vals):
            col = sorted({round(r[1][j], 1) for r in rows}, reverse=True)
            cells.append(fmt(round(v, 1), col[0], col[1] if len(col) > 1 else None))
        out.append((name, cells))
    return out


def wide_table(J, splits, titles, cols_per, caption, label, star=True):
    blocks = [block(J, sp, cols) for sp, cols in zip(splits, cols_per)]
    names = [n for n, _ in blocks[0]]
    spec = "l" + "".join("|" + "c" * len(c) if i else "c" * len(c) for i, c in enumerate(cols_per)).replace("|", "@{\\hspace{8pt}}")
    head1 = " & ".join([""] + [rf"\multicolumn{{{len(c)}}}{{c}}{{{t}}}" for t, c in zip(titles, cols_per)]) + r" \\"
    cm, a = [], 2
    for c in cols_per:
        cm.append(rf"\cmidrule(lr){{{a}-{a + len(c) - 1}}}")
        a += len(c)
    short = {"FSR@2": "@2", "FSR@3": "@3", "FSR@4": "@4", "FSR@5": "@5", "SetEM@|G|": "Set-EM"}
    head2 = " & ".join(["Method"] + [short[x] for c in cols_per for x in c]) + r" \\"
    body = []
    for i, n in enumerate(names):
        cells = []
        for b in blocks:
            row = dict(b).get(n)
            if row is None:
                return None
            cells += row
        if n.startswith(r"\textbf"):
            body.append(r"\midrule")
        body.append(" & ".join([n] + cells) + r" \\")
    env = "table*" if star else "table"
    return "\n".join([rf"\begin{{{env}}}[t]", rf"\caption{{{caption}}}\label{{{label}}}", r"\centering\small\setlength{\tabcolsep}{3.6pt}",
                      rf"\begin{{tabular}}{{{spec}}}", r"\toprule", head1, " ".join(cm), head2, r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"])


def simple(rows, header, caption, label, spec, star=False, size=r"\small", sep="4pt", mid_before=()):
    if not rows:  # no table without data
        return None
    env = "table*" if star else "table"
    L = [rf"\begin{{{env}}}[t]", rf"\caption{{{caption}}}\label{{{label}}}", rf"\centering{size}\setlength{{\tabcolsep}}{{{sep}}}",
         rf"\begin{{tabular}}{{{spec}}}", r"\toprule", header + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i in mid_before:
            L.append(r"\midrule")
        L.append(" & ".join(r) + r" \\")
    return "\n".join(L + [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"])


def mark(cols):
    """cols: list of columns (lists of floats) -> formatted strings with best bold, second underlined."""
    out = []
    for col in cols:
        s = sorted({round(v, 1) for v in col}, reverse=True)
        out.append([fmt(round(v, 1), s[0], s[1] if len(s) > 1 else None) for v in col])
    return out


def main():
    J = json.load(open(sys.argv[1]))
    M = {}  # optional extra result files next to the main one: qaboot.json, latency2.json
    for name in ("qaboot", "latency3", "stopcmp_musique_dev"):
        p = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), f"{name}.json")
        if os.path.exists(p):
            M[name] = json.load(open(p))
    T = {}
    F5 = ["FSR@2", "FSR@3", "FSR@4", "FSR@5", "SetEM@|G|"]
    F4 = ["FSR@2", "FSR@3", "FSR@4", "FSR@5"]
    T["tabmain"] = wide_table(J, ["musique_dev", "2wiki_dev", "hotpotqa_dev"], ["MuSiQue (2--4 hops)", "2WikiMultiHopQA", "HotpotQA"], [F5, F5, F4],
                              r"Distractor setting. Columns @$k$ are full-support recall with $k$ passages; Set-EM is full-support recall at a budget equal to the number of annotated supports. On HotpotQA every question has two supports, so Set-EM equals @2. The seven comparison systems are published retrievers, run on the same pools. Beam Retrieval and \method{} share the backbone, the training questions and the training protocol, and each row is run 0 of that model; Table~\ref{tab:seeds} gives three runs of both on MuSiQue. Best in bold, second underlined." + DAGGER_FULL, "tab:main")
    T["tabopen"] = wide_table(J, ["open_musique", "open_2wiki", "open_hotpotqa"], ["Open MuSiQue (52.9)", "Open 2Wiki (58.4)", "Open HotpotQA (94.4)"], [F5, F5, F4],
                              r"Open-corpus setting: each pool is the top 30 of \texttt{bge-large} over the corpus of the benchmark. The number after each name is the percentage of questions whose annotated support is fully inside the pool, which bounds every entry of that block." + DAGGER, "tab:open")
    T["tabwide"] = wide_table(J, ["open100_musique", "open100_2wiki"], ["Wide MuSiQue (68.4)", "Wide 2Wiki (67.7)"], [F5, F5],
                              r"Wide-pool setting: the first stage returns 100 passages. The number after each name bounds every entry of that block. SetR reads the 30 highest-ranked candidates in its single prompt; every other method reads all 100." + DAGGER, "tab:wide")
    # by proof length (MuSiQue)
    rows, cols = [], []
    keys = []
    for key, name in NAMES:
        m = J["ret"].get(f"musique_dev|{key}"); mo = J["ret"].get(f"open_musique|{key}")
        if m is None or mo is None:
            continue
        keys.append(name)
        h, ho = m["by_hops"], mo["by_hops"]
        cols.append([h["2hop@2"], h["3hop@3"], h["4hop@4"], ho["2hop@2"], ho["3hop@3"], ho["4hop@4"]])
    mk = mark(list(map(list, zip(*cols))))
    rows = [[n] + [mk[j][i] for j in range(6)] for i, n in enumerate(keys)]
    T["tabhops"] = simple(rows, r" & \multicolumn{3}{c}{MuSiQue} & \multicolumn{3}{c}{Open MuSiQue} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7} Method & 2-hop & 3-hop & 4-hop & 2-hop & 3-hop & 4-hop",
                          r"Set-EM by the length of the proof. Each entry is full-support recall at a budget equal to the number of hops." + DAGGER, "tab:hops", "lcccccc", size=r"\footnotesize", sep="3pt", mid_before=(len(rows) - 1,))
    # SR / FSR / Set-EM
    rows, cols, keys = [], [], []
    for key, name in NAMES:
        m = J["ret"].get(f"musique_dev|{key}")
        if m:
            keys.append(name); cols.append([m["SR@4"], m["FSR@4"], m["SetEM@|G|"]])
    mk = mark(list(map(list, zip(*cols))))
    T["tabsr"] = simple([[n] + [mk[j][i] for j in range(3)] for i, n in enumerate(keys)], "Method & SR@4 & FSR@4 & Set-EM",
                        r"MuSiQue under three definitions. Support recall averages annotated passages. FSR@4 requires every annotated passage inside four. Set-EM requires every annotated passage inside a budget equal to the length of the proof." + DAGGER, "tab:sr", "lccc", mid_before=(len(keys) - 1,))
    # stop rule
    rows = []
    for sp, nm in [("musique_dev", "MuSiQue"), ("2wiki_dev", "2Wiki"), ("hotpotqa_dev", "HotpotQA"), ("open_musique", "Open MuSiQue"), ("open_2wiki", "Open 2Wiki"), ("open_hotpotqa", "Open HotpotQA")]:
        s = J["stop"].get(f"{sp}|PG"); b = J["ret"].get(f"{sp}|beam")
        if s is None or b is None:
            MISSING.append(f"stop {sp}")
            continue
        rows.append([nm, f"{s['complete']:.1f}", f"{s['exact']:.1f}", f"{s['passages']:.2f}", f"{b['FSR@2']:.1f}", f"{b['FSR@3']:.1f}", f"{b['FSR@4']:.1f}"])
    T["tabstop"] = simple(rows, r" & \multicolumn{3}{c}{\method{}, stop rule} & \multicolumn{3}{c}{Beam$^\dagger$ at budget} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7} Pool & complete & exact & size & 2 & 3 & 4",
                          r"Proof length withheld. \method{} returns the shortest set whose completeness reaches $1/2$. ``Complete'' is the percentage of questions whose annotated support is inside the returned set, ``exact'' the percentage whose returned set is the annotated support, and ``size'' the mean number of passages returned. Beam Retrieval is given a fixed number of passages, so the two sides are different operating points and nothing is marked in bold." + DAGGER, "tab:stop", "lcccccc", size=r"\footnotesize", sep="3pt", mid_before=(3,))
    # proof length withheld on MuSiQue: every way of choosing the length, means over the training runs (ana_stopcmp.py)
    SC = M.get("stopcmp_musique_dev")
    if SC:
        import statistics as st_
        pg = [SC[k] for k in ("pg0", "pg1", "pg2") if k in SC]
        bm = [SC[k] for k in ("beam0", "beam1", "beam2") if k in SC]
        MET = ("complete", "exact", "f1", "size")

        def clf(o):  # one run: mean over the classifier seeds
            vs = [o[f"lenclf{c}"] for c in (0, 1, 2) if f"lenclf{c}" in o]
            return {m: st_.mean(v[m] for v in vs) for m in MET}
        B_ = r"Beam$^\dagger$"
        spec = [(B_, "fixed at three", [o["fixed"]["3"] for o in bm]),
                ("", "score $\\geq 1/2$", [o["rule"] for o in bm]),
                ("", "largest score drop", [o["cliff"] for o in bm]),
                ("", "highest score", [o["peak"] for o in bm]),
                ("", "question classifier", [clf(o) for o in bm]),
                (r"\method{}", "question classifier", [clf(o) for o in pg]),
                ("", "deficit (stop rule)", [o["rule"] for o in pg]),
                (B_, "given", [o["oracle_length"] for o in bm]),
                (r"\method{}", "given", [o["oracle_length"] for o in pg])]
        mean = [[st_.mean(v[m] for v in vals) for m in MET] for _, _, vals in spec]
        nw = 7  # the rows in which the length is withheld; the best of each column among them is in bold
        best = [max(round(mean[i][j], 1) for i in range(nw)) for j in range(3)]
        rows = []
        for i, (a_, b_, vals) in enumerate(spec):
            cells = [rf"\textbf{{{mean[i][j]:.1f}}}" if i < nw and round(mean[i][j], 1) == best[j] else f"{mean[i][j]:.1f}" for j in range(3)]
            rows.append([a_, b_] + cells + [f"{mean[i][3]:.2f}"])
            print(f"   stopcmp {a_ or '  '} | {b_}: " + "  ".join(f"{m} {st_.mean(v[m] for v in vals):.1f} [{min(v[m] for v in vals):.1f}, {max(v[m] for v in vals):.1f}]" for m in MET[:3])
                  + f"  size {mean[i][3]:.2f}")
        acc = [SC["pg0"][f"lenclf{c}_acc"] for c in (0, 1, 2)]
        print(f"   stopcmp: classifier accuracy {acc}; runs PG {len(pg)} Beam {len(bm)}")
        T["tabstopcmp"] = simple(rows, "Sets of & Length from & complete & exact & F1 & size",
                                 r"Proof length withheld on MuSiQue, means of three training runs. ``Complete'' is the percentage of questions whose annotated support is inside the returned set, ``exact'' the percentage whose returned set is the annotated support, F1 compares the returned set with the support, and ``size'' is the mean number of passages returned. Beam Retrieval is stopped by a fixed budget or by its own sequence score: while the next sequence scores at least $1/2$, before the largest drop, or at the highest score. The question classifier is a separate model with the same backbone that predicts the length from the question, " + f"{st_.mean(acc):.0f}" + r"\% accurate; its rows also average three classifier seeds. The last two rows are told the length. Best of the first seven rows in bold." + DAGGER,
                                 "tab:stopcmp", "llcccc", size=r"\footnotesize", sep="3pt", mid_before=(5, 7))
        for h in ("2", "3", "4"):
            f_ = lambda xs: st_.mean(xs)
            print(f"   stopcmp {h}-hop exact: deficit rule {f_([o['by_hops']['rule'][h]['exact'] for o in pg]):.1f}; Beam peak {f_([o['by_hops']['peak'][h]['exact'] for o in bm]):.1f}; "
                  f"PG + question length {f_([st_.mean(o[f'lenclf{c}_by_hops'][h]['exact'] for c in (0, 1, 2)) for o in pg]):.1f}; Beam + question length {f_([st_.mean(o[f'lenclf{c}_by_hops'][h]['exact'] for c in (0, 1, 2)) for o in bm]):.1f}")
    else:
        MISSING.append("stopcmp_musique_dev")
    # QA
    rows, cols, keys = [], [], []
    for key, name in NAMES:
        k = "pg_nolist_q" if key == "PG" else f"{key}_q"
        a, b = J["qa"].get(f"musique_dev|{k}"), J["qa"].get(f"open_musique|{k}")
        if a is None or b is None:
            MISSING.append(f"qa {key}")
            continue
        keys.append(name); cols.append([a["EM"], a["F1"], b["EM"], b["F1"]])
    mk = mark(list(map(list, zip(*cols))))
    rows = [[n] + [mk[j][i] for j in range(4)] for i, n in enumerate(keys)] if cols else []
    nmain = len(rows)
    for k, nm in (("pg_nolist_adapt_q", r"\method{}, stop rule"), ("gold_q", r"\emph{Annotated support}")):
        g1, g2 = J["qa"].get(f"musique_dev|{k}"), J["qa"].get(f"open_musique|{k}")
        if rows and g1 and g2:
            rows.append([nm, f"{g1['EM']:.1f}", f"{g1['F1']:.1f}", f"{g2['EM']:.1f}", f"{g2['F1']:.1f}"])
    T["tabqa"] = simple(rows, r" & \multicolumn{2}{c}{MuSiQue} & \multicolumn{2}{c}{Open MuSiQue} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5} Passages from & EM & F1 & EM & F1",
                        r"Answers of one fixed reader (Qwen3-8B) on 1,000 questions per pool. The upper block gives the reader four passages from each retriever, and best and second best are marked within it. The stop-rule row gives it the set \method{} returns when the proof length is withheld (2.94 and 3.33 passages on average). The last row gives it the annotated support." + DAGGER, "tab:qa", "lcccc", mid_before=(nmain - 1, nmain))
    # transfer
    rows = []
    for sp, nm in [("hotpotqa_dev", "HotpotQA"), ("2wiki_dev", "2Wiki")]:
        a, b = J["ret"].get(f"{sp}|pg_nolist_xmu"), J["ret"].get(f"{sp}|beam_xmu")
        if a is None or b is None:
            MISSING.append(f"transfer {sp}")
            continue
        z = J["ret"].get(f"{sp}|llmsetfill")  # the strongest retriever of the main table that is trained on no benchmark
        if z is not None:
            rows.append([nm, "SetR"] + [f"{z[c]:.1f}" for c in F5])
        for nmm, m, o in ((BEAM, b, a), (r"\method{}", a, b)):  # the better of the two transferred models in bold
            rows.append(["" if z is not None or m is a else nm, nmm] + [rf"\textbf{{{m[c]:.1f}}}" if round(m[c], 1) > round(o[c], 1) else f"{m[c]:.1f}" for c in F5])
        print(f"transfer {sp}: PG hops", {k: round(v, 1) for k, v in a.get("by_hops", {}).items()}, "Beam hops", {k: round(v, 1) for k, v in b.get("by_hops", {}).items()})
    if rows:
        per = len(rows) // 2
        T["tabtransfer"] = simple(rows, "Target & Model & @2 & @3 & @4 & @5 & Set-EM",
                                  r"Transfer without target-domain training. Beam Retrieval and \method{} are trained on MuSiQue only and applied to the distractor pools of another benchmark; the better of the two is in bold. SetR is trained on no benchmark." + DAGGER, "tab:transfer", "llccccc",
                                  size=r"\footnotesize", sep="3pt", mid_before=(per,) if len(rows) > per else ())
    # seeds
    def seedruns(key0, keyfn, split):  # the metric dicts of seeds 0, 1, 2 that exist
        runs = [J["ret"].get(f"{split}|{key0}")] + [keyfn(s, split) for s in (1, 2)]
        return [m for m in runs if m]
    import statistics as st
    sm = {}
    for sp in ("musique_dev", "open_musique"):
        sm[("PG", sp)] = seedruns("PG", lambda s, sp_: J["ret"].get(f"{sp_}|musique_nolist_s{s}"), sp)
        sm[("beam", sp)] = seedruns("beam", lambda s, sp_: J["seeds"].get(f"{sp_}|beam|s{s}"), sp)
    sd = {k: [m["SetEM@|G|"] for m in v] for k, v in sm.items()}
    HOPS = ("2hop@2", "3hop@3", "4hop@4")
    hopmean = {k: [st.mean(m["by_hops"][h] for m in v) for h in HOPS] for k, v in sm.items() if k[1] == "musique_dev" and v}
    print("seeds", {k: [round(x, 1) for x in v] for k, v in sd.items()})
    for k, v in sd.items():
        if len(v) > 1:
            print(f"   {k}: mean {st.mean(v):.2f} sd {st.stdev(v):.2f} min {min(v):.1f} max {max(v):.1f}")
    print("   hop means", {k[0]: [round(x, 1) for x in v] for k, v in hopmean.items()})
    if "PG" in {k[0] for k in hopmean} and "beam" in {k[0] for k in hopmean}:
        print("   hop margins of the seed means", [round(a - b, 1) for a, b in zip(hopmean[("PG", "musique_dev")], hopmean[("beam", "musique_dev")])])
    for c in F4:
        print(f"   {c} seed means: PG {st.mean(m[c] for m in sm[('PG', 'musique_dev')]):.1f} Beam {st.mean(m[c] for m in sm[('beam', 'musique_dev')]):.1f}")
    stops = [J["stop"].get("musique_dev|PG")] + [J["stop"].get(f"musique_dev|musique_nolist_s{s}") for s in (1, 2)]
    stops = [s for s in stops if s]
    print("   stop rule per seed (complete, exact, passages):", [(round(s["complete"], 1), round(s["exact"], 1), round(s["passages"], 2)) for s in stops])
    if len(stops) > 1:
        print(f"   stop rule seed means: complete {st.mean(s['complete'] for s in stops):.1f} exact {st.mean(s['exact'] for s in stops):.1f} passages {st.mean(s['passages'] for s in stops):.2f}")
    # the two-epoch run (the epochs and updates of Beam Retrieval on MuSiQue).  ONE run, seed 1: it was the first of three
    # started runs to finish and was chosen for that reason before any two-epoch result existed; the other two were stopped.
    TWO = {sp: J["ret"].get(f"{sp}|musique_nolist2_s1") for sp in ("musique_dev", "open_musique")}
    for sp, m in TWO.items():
        if m:
            print(f"   2-EPOCH RUN (seed 1) {sp}: Set-EM {m['SetEM@|G|']:.1f} FSR@2-5 " + "/".join(f"{m[c]:.1f}" for c in F4)
                  + f" | Beam runs {[round(x, 1) for x in sd[('beam', sp)]]} | one-epoch runs {[round(x, 1) for x in sd[('PG', sp)]]}"
                  + (" | hops " + str([round(m['by_hops'][h], 1) for h in HOPS]) + f" SR@4 {m.get('SR@4', float('nan')):.1f}" if sp == "musique_dev" else ""))

    def listing(v):  # "33.3, 32.2 and 30.1"
        s = [f"{x:.1f}" for x in v]
        return s[0] if len(s) == 1 else ", ".join(s[:-1]) + " and " + s[-1]
    po, bo = sd[("PG", "open_musique")], sd[("beam", "open_musique")]
    T["seedsopen"] = (rf"Only open MuSiQue has been repeated: the {['', 'one run', 'two runs', 'three runs'][len(po)]} of \method{{}} evaluated there reach {listing(po)}, "
                      rf"{st.mean(po):.1f} on average, and the {['', 'one run', 'two runs', 'three runs'][len(bo)]} of Beam Retrieval {listing(bo)}, {st.mean(bo):.1f} on average.")
    three = [nm for sp, nm in (("musique_dev", "MuSiQue"), ("open_musique", "open MuSiQue")) if len(sd[("PG", sp)]) == 3 and len(sd[("beam", sp)]) == 3]
    T["teaserruns"] = (rf"On {' and '.join(three)} the bars of the two trained retrievers are means of three training runs; the other pools show run 0." if three
                       else "The two trained retrievers are shown with run 0.")  # the rule make_teaser.py applies to panel (a)
    if len(sd[("PG", "musique_dev")]) == 3 and len(sd[("beam", "musique_dev")]) == 3:  # three runs of both models on MuSiQue
        quant = [("Set-EM", "musique_dev", lambda m: m["SetEM@|G|"])] + \
                [(f"\\quad {h[0]}-hop questions", "musique_dev", (lambda m, h=h: m["by_hops"][h])) for h in HOPS] + \
                [(f"Full support in {c[-1]} passages", "musique_dev", (lambda m, c=c: m[c])) for c in F4]
        mids = [4]
        if len(sd[("PG", "open_musique")]) == 3 and len(sd[("beam", "open_musique")]) == 3:
            mids.append(len(quant))
            quant.append(("Open MuSiQue, Set-EM", "open_musique", lambda m: m["SetEM@|G|"]))
        else:
            MISSING.append("seeds on open_musique (need 3 per model): " + str({k: len(v) for k, v in sd.items() if k[1] == "open_musique"}))
        # the single two-epoch run (seed 1) is one more column once it is evaluated on every split the table shows
        # (no row with an empty cell: without its open-pool result the column is left out and reported)
        two = all(TWO[sp] for sp in {sp for _, sp, _ in quant})
        if TWO["musique_dev"] and not two:
            MISSING.append("two-epoch run on open_musique (its column is left out of the seed table until it is in)")
        ms = lambda v: f"{st.mean(v):.1f} $\\pm$ {st.stdev(v):.1f}"
        rows = []
        for lab, sp, get in quant:
            vb, vp = [get(m) for m in sm[("beam", sp)]], [get(m) for m in sm[("PG", sp)]]
            print(f"   seeds table {lab}: Beam {[round(x, 1) for x in vb]} PG {[round(x, 1) for x in vp]} diff of means {st.mean(vp) - st.mean(vb):+.2f}; PG min - Beam max {min(vp) - max(vb):+.1f}")
            rows.append([lab] + [f"{x:.1f}" for x in vb] + [ms(vb)] + [f"{x:.1f}" for x in vp] + [ms(vp), f"${st.mean(vp) - st.mean(vb):+.1f}$"])
            if two:
                x2 = get(TWO[sp])
                rows[-1].append(f"{x2:.1f}")
                print(f"      two-epoch run {x2:.1f}: minus Beam seed 1 {x2 - vb[1]:+.1f}, minus Beam max {x2 - max(vb):+.1f}, minus Beam mean {x2 - st.mean(vb):+.1f}; "
                      f"minus one-epoch seed 1 {x2 - vp[1]:+.1f}, minus one-epoch max {x2 - max(vp):+.1f}, minus one-epoch mean {x2 - st.mean(vp):+.1f}")
        if two:
            T["tabseeds"] = simple(rows, r" & \multicolumn{4}{c}{Beam Retrieval$^\dagger$, two epochs} & \multicolumn{4}{c}{\method{}, one epoch} & & \method{}, two epochs \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){11-11} MuSiQue, run & 0 & 1 & 2 & mean $\pm$ sd & 0 & 1 & 2 & mean $\pm$ sd & $\Delta$ & 1",
                                   r"Training runs on MuSiQue. A run is one seed, which fixes the initialisation of the head, the order of the questions, and the sampled states and pools. The two models of the other tables have three runs each; $\Delta$ is the difference of their means, and sd is the sample standard deviation of three runs. The last column is a single run of \method{} trained for two epochs, the schedule of Beam Retrieval on this benchmark (4,735 updates against 2,368), with seed 1. Every other table reports run 0 of Beam Retrieval and run 0 of the one-epoch \method{}." + DAGGER,
                                   "tab:seeds", "lcccccccccc", star=True, mid_before=tuple(mids))
        else:
            T["tabseeds"] = simple(rows, r" & \multicolumn{4}{c}{Beam Retrieval$^\dagger$} & \multicolumn{4}{c}{\method{}} & \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9} MuSiQue, run & 0 & 1 & 2 & mean $\pm$ sd & 0 & 1 & 2 & mean $\pm$ sd & $\Delta$",
                                   r"Three training runs of each model on MuSiQue. A run is one seed, which fixes the initialisation of the head, the order of the questions, and the sampled states and pools. $\Delta$ is the difference of the two means, and sd is the sample standard deviation of three runs. Every other table reports run 0 of each model." + DAGGER, "tab:seeds", "lccccccccc", star=True, mid_before=tuple(mids))
    else:
        MISSING.append("seeds (need 3 per model): " + str({k: len(v) for k, v in sd.items()}))
    # controls
    have = lambda k: all(J["ret"].get(f"{sp}|{k}") for sp in ("musique_dev", "open_musique"))
    final = all(have(k) for k in ("musique_nl_bin", "musique_nl_holo"))
    if final:  # every control is the reported recipe with one change (a row whose open-pool result is not in yet is left out and reported)
        ctl = [("PG", r"\textbf{\method{}}"), ("musique_nl_bin", r"\quad binary target"), ("musique_nl_holo", r"\quad joint reading"),
               ("musique_nl_noopen", r"\quad no open-style pools"), ("pg_ep1", r"\quad with listwise loss"), ("musique_v6", r"\quad with membership loss"),
               ("beam", BEAM)]
        cap = r"Variants of the training, Set-EM, one run each with seed 0. Each indented row is the training of \method{} with one change. The binary row ranks children by its completeness logit, the only reading it has; every other row uses the search of \method{}." + (rf" The {len(sd[('PG', 'musique_dev')])} runs of the unchanged training span {min(sd[('PG', 'musique_dev')]):.1f} to {max(sd[('PG', 'musique_dev')]):.1f} on MuSiQue (Table~\ref{{tab:seeds}}).")
        T["ablfinal"] = "1"
        # the two mechanism controls repeated with a second seed (table only when both runs are in)
        rows = []
        for key0, key1, name in (("PG", "musique_nolist_s1", r"\method{}"), ("musique_nl_bin", "musique_nl_bin_s1", r"\quad binary target"),
                                 ("musique_nl_holo", "musique_nl_holo_s1", r"\quad joint reading")):
            r = [name]
            for sp in ("musique_dev", "open_musique"):
                a, b = J["ret"].get(f"{sp}|{key0}"), J["ret"].get(f"{sp}|{key1}")
                if a is None or b is None:
                    r = None
                    break
                r += [f"{a['SetEM@|G|']:.1f}", f"{b['SetEM@|G|']:.1f}", f"{(a['SetEM@|G|'] + b['SetEM@|G|']) / 2:.1f}"]
            if r is None:
                rows = []
                MISSING.append(f"control seed 1: {key1}")
                break
            rows.append(r)
        T["tabablseeds"] = simple(rows, r" & \multicolumn{3}{c}{MuSiQue} & \multicolumn{3}{c}{Open MuSiQue} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7} Seed & 0 & 1 & mean & 0 & 1 & mean",
                                  r"The two mechanism controls with a second training seed, Set-EM. Rows of one column share the seed.", "tab:ablseeds", "lcccccc", size=r"\footnotesize", sep="4pt")
    else:  # until those runs are in: the controls that exist were trained with the listwise term
        ctl = [("PG", r"\textbf{\method{}}"), ("pg_ep1", r"\ \ with listwise loss"), ("pg_bin", r"\quad binary target"),
               ("pg_holo", r"\quad joint reading"), ("pg_noopen", r"\quad no open-style pools"), ("beam", BEAM)]
        cap = r"Controls, Set-EM. The second row adds a listwise ranking loss to the training of \method{}. The next three rows keep that loss and change one more thing: a binary target in place of the graded one, one joint reading of the set in place of the member views, or training without the open-style pools. They are compared with the second row."
        for k in ("musique_nl_bin", "musique_nl_holo", "musique_nl_noopen"):
            if not have(k):
                MISSING.append(f"control {k}")
    rows = []
    for key, name in ctl:
        a, b = J["ret"].get(f"musique_dev|{key}"), J["ret"].get(f"open_musique|{key}")
        if a is None or b is None:
            MISSING.append(f"control {key}")
            continue
        h = a["by_hops"]
        f = (lambda v: rf"\textbf{{{v:.1f}}}") if key == "PG" else (lambda v: f"{v:.1f}")
        rows.append([name, f(a["SetEM@|G|"]), f(h["2hop@2"]), f(h["3hop@3"]), f(h["4hop@4"]), f(b["SetEM@|G|"])])
    T["tababl"] = simple(rows, r" & \multicolumn{4}{c}{MuSiQue} & Open \\ \cmidrule(lr){2-5} Variant & Set-EM & 2-hop & 3-hop & 4-hop & Set-EM",
                         cap + DAGGER, "tab:abl", "lccccc", size=r"\footnotesize", sep="3pt", mid_before=(len(rows) - 1,))
    # controls at shared budgets (appendix)
    rows = []
    for key, name in ctl[:-1]:
        a = J["ret"].get(f"musique_dev|{key}")
        b = J["ret"].get(f"open_musique|{key}")
        if a and b:
            rows.append([name] + [f"{a[c]:.1f}" for c in F4] + [f"{b[c]:.1f}" for c in F4])
    T["tabcurve"] = simple(rows, r" & \multicolumn{4}{c}{MuSiQue} & \multicolumn{4}{c}{Open MuSiQue} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9} Variant & @2 & @3 & @4 & @5 & @2 & @3 & @4 & @5",
                           r"Full-support recall at shared budgets for the rows of Table~\ref{tab:abl}.", "tab:curve", "lcccccccc", star=True)
    # search settings (appendix): width, proposals, frontier
    sens = [("PG", r"$W{=}2$, $E{=}4$ (\method{})"), ("pg_nolist_fl", r"plain frontier"), ("pg_nolist_w1", r"$W{=}1$"),
            ("pg_nolist_w4", r"$W{=}4$"), ("pg_nolist_e2", r"$E{=}2$"), ("pg_nolist_e8", r"$E{=}8$")]
    rows = []
    for key, name in sens:
        a, s = J["ret"].get(f"musique_dev|{key}"), J["stop"].get(f"musique_dev|{key}")
        if a is None or s is None:
            MISSING.append(f"sens {key}")
            continue
        rows.append([name] + [f"{a[c]:.1f}" for c in F5] + [f"{s['enc_to4']:.0f}"])
    T["tabsens"] = simple(rows, r"Search & @2 & @3 & @4 & @5 & Set-EM & seq.",
                          r"Search settings on MuSiQue with the same weights; each row changes one setting of the first. ``Seq.'' is the mean number of encoder inputs per question for a search to four passages.", "tab:sens", "lcccccc",
                          size=r"\footnotesize", sep="3pt")
    # bootstrap
    rows = []
    for sp, nm in [("musique_dev", "MuSiQue"), ("2wiki_dev", "2Wiki"), ("hotpotqa_dev", "HotpotQA"), ("open_musique", "Open MuSiQue"), ("open_2wiki", "Open 2Wiki"), ("open_hotpotqa", "Open HotpotQA"),
                   ("open100_musique", "Wide MuSiQue"), ("open100_2wiki", "Wide 2Wiki")]:
        b = J["boot"].get(sp)
        if not b:
            MISSING.append(f"boot {sp}")
            continue
        x = b["SetEM"]
        p = "$<$0.001" if x["p"] < 0.001 else f"{x['p']:.3f}"
        rows.append([nm, f"${x['diff']:+.1f}$", f"[${x['lo']:+.1f}$, ${x['hi']:+.1f}$]", p, str(x["wins"]), str(x["losses"])])
    T["tabboot"] = simple(rows, r"Pool & $\Delta$ & 95\% interval & $p$ & wins & losses",
                          r"Paired bootstrap of the Set-EM difference $\Delta$ between run 0 of \method{} and run 0 of Beam Retrieval, 10,000 resamples of the questions. Wins and losses count questions on which exactly one of the two returns the proof. The intervals cover the sampling of questions for this pair of models and not the variation between training runs, which Table~\ref{tab:seeds} reports.", "tab:boot", "lccccc",
                          size=r"\footnotesize", sep="3pt")
    # answer bootstrap (appendix), from qaboot.py
    QB = M.get("qaboot", {})
    rows = []
    for sp, nm in (("musique_dev", "MuSiQue"), ("open_musique", "Open MuSiQue")):
        for key, name in (("beam_q", r"Beam$^\dagger$"), ("llmsetfill_q", "SetR")):
            o = QB.get(f"{sp}|{key}")
            if o is None:
                MISSING.append(f"qaboot {sp} {key}")
                continue
            rows.append([nm, name] + [f"${o[m]['diff']:+.1f}$ [${o[m]['lo']:+.1f}$, ${o[m]['hi']:+.1f}$]" for m in ("EM", "F1")])
    T["tabqaboot"] = simple(rows, r"Pool & vs. & $\Delta$ EM & $\Delta$ F1",
                            r"Paired bootstrap of the answer differences of Table~\ref{tab:qa} at four passages: \method{} minus the named retriever, with 95\% intervals over 1,000 questions." + DAGGER, "tab:qaboot", "llcc",
                            size=r"\footnotesize", sep="3pt", mid_before=(2,))
    # measured cost, from latency2.py
    LAT = M.get("latency3")
    if LAT:
        rows = []
        r = LAT["bge-reranker-v2-m3"]
        rows.append(["bge-reranker-v2-m3", "pool", f"{r['seqs_per_q']:.0f}", f"{r['ms_per_q']:.0f}"])
        for name, lab in (("Beam Retrieval", BEAM), ("ProofGap", r"\method{}")):
            for K in (2, 3, 4):
                r = LAT[f"{name} K={K}"]
                rows.append([lab if K == 2 else "", str(K), f"{r['seqs_per_q']:.0f}", f"{r['ms_per_q']:.0f}"])
        r = LAT["ProofGap stop"]
        rows.append([r"\method{}, stop rule", f"{r['mean_depth']:.2f}", f"{r['seqs_per_q']:.0f}", f"{r['ms_per_q']:.0f}"])
        T["tabcost"] = simple(rows, "Method & Passages & Sequences & ms per question",
                              rf"Measured cost on {LAT['n']} MuSiQue questions drawn at random, pools of 20 passages. Each question is processed alone, and the systems take turns on it in a shuffled order on one RTX 4090 shared with other jobs, so the times include that load, the same for every system. ``Sequences'' is the mean number of encoder inputs per question; the reranker scores every candidate once." + DAGGER,
                              "tab:cost", "lccc", size=r"\footnotesize", sep="4pt", mid_before=(1, 4, 7))
        print("latency3:", {k: v for k, v in LAT.items()})
    else:
        MISSING.append("latency3")
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        f.write("% generated by code/erc/gen36.py -- do not edit\n")
        for k, v in T.items():
            if v is None:
                print("TABLE NOT WRITTEN:", k)
                continue
            f.write(f"\\newcommand{{\\{k}}}{{%\n{v}\n}}\n\n")
    print("missing:", MISSING)
    # numbers quoted in the text
    for sp in ("musique_dev", "2wiki_dev", "hotpotqa_dev", "open_musique", "open_2wiki", "open_hotpotqa", "open100_musique", "open100_2wiki"):
        a, b = J["ret"].get(f"{sp}|PG"), J["ret"].get(f"{sp}|beam")
        if a and b:
            print(f"{sp}: PG SE {a['SetEM@|G|']:.1f} Beam {b['SetEM@|G|']:.1f} diff {a['SetEM@|G|'] - b['SetEM@|G|']:+.1f}; FSR PG " + "/".join(f"{a[c]:.1f}" for c in F4) + " Beam " + "/".join(f"{b[c]:.1f}" for c in F4)
                  + f"; SR@4 {a['SR@4']:.1f} vs {b['SR@4']:.1f}")


if __name__ == "__main__":
    main()
