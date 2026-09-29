"""Pooled (model-stratified) analysis of arm contrasts across all models.

Unit = (model, item) pair; strata = models. Each contrast X vs A uses the items where
both arms are complete for that model. Outcome = majority vote over answer clusters,
with ties broken at random, scored as the EXPECTED correctness (1/|tied| if the gold
cluster is among the tied leaders). No tie-break rule can favour an arm.

Tests
  - stratified sign-flip permutation test (exact under H0 of exchangeable arms
    within each item), two-sided;
  - stratified bootstrap CI of the item-weighted pooled difference;
  - DerSimonian-Laird random-effects estimate across models (allows heterogeneity);
  - stratified McNemar (exact binomial on pooled discordant pairs) as sensitivity,
    with deterministic lowest-slot tie-break.
Family for Holm (planned): S-A, C-A, G-A. S4 (S minus SymPy) is exploratory: it was
chosen after seeing gemma, so it is also reported on the other models only.
"""
import json, math
from collections import defaultdict

import numpy as np

import aggregate as AG
import arms_v2 as AR

SOURCES = {
    "google/gemma-4-31b-it": ["results/records_gemma_partial.jsonl", "results/records_gemmaX_partial.jsonl",
                              "results/records_v10_gemma31b.jsonl", "results/records_vs4_gemma31b.jsonl"],
    "gemini-3.5-flash-lite": ["results/records_final_gemflashlite.jsonl"],
    "gemini-3.1-flash-lite": ["results/records_final_gem31flashlite.jsonl"],
    "openai/gpt-oss-20b": ["results/records_final_gptoss20b.jsonl", "results/records_final_gptoss20bgroq.jsonl"],
}
SYMPY_SLOT = 2
ITEMS90 = json.load(open("data/items90.json"))     # the fixed item set: every analysis uses it
# same open weights served by two providers on DISJOINT items -> one model stratum
MERGE = {"groq:openai/gpt-oss-20b": "openai/gpt-oss-20b"}


def load_all():
    recs = []
    for m, paths in SOURCES.items():
        for p in paths:
            try:
                for r in (json.loads(l) for l in open(p)):
                    r["model"] = MERGE.get(r["model"], r["model"])
                    if r["model"] == m:
                        recs.append(r)
            except FileNotFoundError:
                pass
    tmp = "/tmp/_pooled.jsonl"
    with open(tmp, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    _, agents, verif, gold, _ = AG.load(tmp)
    cl = AG.build_clusters(recs, gold, "results/clusters_cache.json")
    return agents, verif, gold, cl


def labels(agents, cl, m, arm, it, slots=range(5)):
    blk = agents.get((m, arm), {}).get(it)
    if not blk or any(s not in blk or blk[s].get("error") for s in slots):
        return None
    return np.array([AG.label(cl, it, blk[s].get("answer")) for s in slots])


def mv_expected(L):
    votes = defaultdict(int)
    for l in L:
        if l != -1:
            votes[l] += 1
    if not votes:
        return 0.0
    top = max(votes.values())
    tied = [l for l, v in votes.items() if v == top]
    return (1.0 / len(tied)) if 0 in tied else 0.0


def mv_lowest(L):
    return float(AG.weighted_vote(L, np.ones(len(L))) == 0)


ARMS = {
    "A": ("A", range(5)),
    "S": ("S", range(5)),
    "S4": ("S", [s for s in range(5) if s != SYMPY_SLOT]),
    "C": ("C", range(5)),
    "G": ("G", range(5)),
}


def arm_labels(agents, cl, m, name, it):
    arm, slots = ARMS[name]
    return labels(agents, cl, m, arm, it, slots)


def paired(agents, cl, gold, m, x, y="A"):
    rows = []
    for it in ITEMS90:
        if it not in gold:
            continue
        Lx, Ly = arm_labels(agents, cl, m, x, it), arm_labels(agents, cl, m, y, it)
        if Lx is None or Ly is None:
            continue
        rows.append(dict(it=it, dx=mv_expected(Lx), dy=mv_expected(Ly),
                         lx=mv_lowest(Lx), ly=mv_lowest(Ly),
                         ix=float((Lx == 0).mean()), iy=float((Ly == 0).mean())))
    return rows


def stratified_tests(strata, key_x="dx", key_y="dy", reps=20000, seed=5):
    rng = np.random.RandomState(seed)
    ds = {m: np.array([r[key_x] - r[key_y] for r in rows]) for m, rows in strata.items() if rows}
    allm = list(ds)
    N = sum(len(d) for d in ds.values())
    obs = sum(d.sum() for d in ds.values()) / N
    # sign-flip permutation (flip each item's difference independently)
    perm = np.zeros(reps)
    for m in allm:
        d = ds[m]
        perm += (d[None, :] * rng.choice([-1, 1], size=(reps, len(d)))).sum(1)
    perm /= N
    p_perm = (np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (reps + 1)
    # stratified bootstrap
    boots = np.zeros(reps)
    for m in allm:
        d = ds[m]
        boots += d[rng.randint(0, len(d), size=(reps, len(d)))].sum(1)
    boots /= N
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # DerSimonian-Laird random effects over per-model mean differences
    est, var = [], []
    for m in allm:
        d = ds[m]
        v = d.var(ddof=1) / len(d) if len(d) > 1 else 0.25
        est.append(d.mean()); var.append(max(v, 1e-4))
    est, var = np.array(est), np.array(var)
    w = 1 / var
    fe = (w * est).sum() / w.sum()
    Q = (w * (est - fe) ** 2).sum()
    tau2 = max(0.0, (Q - (len(est) - 1)) / (w.sum() - (w ** 2).sum() / w.sum())) if len(est) > 1 else 0.0
    wr = 1 / (var + tau2)
    re = (wr * est).sum() / wr.sum()
    se = math.sqrt(1 / wr.sum())
    return dict(n=N, diff=obs, lo=lo, hi=hi, p_perm=p_perm, re=re, re_lo=re - 1.96 * se,
                re_hi=re + 1.96 * se, tau2=tau2, per_model={m: (len(ds[m]), ds[m].mean()) for m in allm})


def mcnemar(strata):
    b = sum(1 for rows in strata.values() for r in rows if r["lx"] > r["ly"])
    c = sum(1 for rows in strata.values() for r in rows if r["lx"] < r["ly"])
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return b, c, min(1.0, p)


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main():
    agents, verif, gold, cl = load_all()
    models = [m for m in SOURCES if any(k[0] == m for k in agents)]
    short = lambda m: AR.SHORT.get(m, m)
    out = {}

    print("POOLED, MODEL-STRATIFIED ANALYSIS  (majority vote, random tie-break as expectation)\n")
    fam = ["S", "C", "G"]
    res = {}
    for x in fam + ["S4"]:
        strata = {m: paired(agents, cl, gold, m, x) for m in models}
        strata = {m: r for m, r in strata.items() if r}
        if not strata:
            continue
        res[x] = (strata, stratified_tests(strata), mcnemar(strata))
    adj = holm(np.array([res[x][1]["p_perm"] for x in fam if x in res]))
    padj = dict(zip([x for x in fam if x in res], adj))

    print(f"{'contrast':<10}{'n':>5}{'pooled diff':>13}{'95% CI (strat. boot)':>24}{'p perm':>9}{'p Holm':>9}"
          f"{'random-effects [95% CI]':>30}{'McNemar b/c p':>18}")
    for x, (strata, t, (b, c, pm)) in res.items():
        ph = f"{padj[x]:.4f}" if x in padj else "  expl."
        print(f"{x+' - A':<10}{t['n']:>5}{t['diff']:>+13.3f}   [{t['lo']:+.3f}, {t['hi']:+.3f}]"
              f"{t['p_perm']:>11.4f}{ph:>9}   {t['re']:+.3f} [{t['re_lo']:+.3f}, {t['re_hi']:+.3f}]"
              f"{b:>7}/{c:<3}{pm:.4f}")
        print("          per model: " + "   ".join(f"{short(m)} n={n} {d:+.3f}" for m, (n, d) in t["per_model"].items()))
        out[f"{x}-A"] = dict(t, holm=float(padj.get(x, float('nan'))), mcnemar=(b, c, pm))

    # S4 out-of-sample: only models NOT used to choose it
    if "S4" in res:
        strata = {m: r for m, r in res["S4"][0].items() if m != "google/gemma-4-31b-it"}
        if strata:
            t = stratified_tests(strata)
            print(f"\nS4 - A on models not used to choose S4 (gemini, gpt-oss): n={t['n']}  "
                  f"{t['diff']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}]  p={t['p_perm']:.4f}")
            out["S4-A_oos"] = t
        s4s = {m: paired(agents, cl, gold, m, "S4", "S") for m in models}
        s4s = {m: r for m, r in s4s.items() if r and m != "google/gemma-4-31b-it"}
        if s4s:
            t = stratified_tests(s4s)
            print(f"S4 - S  on models not used to choose S4:            n={t['n']}  "
                  f"{t['diff']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}]  p={t['p_perm']:.4f}")
            out["S4-S_oos"] = t

    # mechanism: per-agent accuracy (augmentation) and error correlation (decorrelation)
    print("\nMECHANISM  (S vs A): per-agent accuracy gain and error correlation, per model and pooled")
    strata = res["S"][0]
    t = stratified_tests(strata, "ix", "iy")
    for m, rows in strata.items():
        its = [r["it"] for r in rows]
        phis = {}
        for arm in ("A", "S"):
            L = np.array([arm_labels(agents, cl, m, arm, it) for it in its])
            phis[arm] = AG.diversity(L == 0, L)["phi"]
        ia, isx = np.mean([r["iy"] for r in rows]), np.mean([r["ix"] for r in rows])
        print(f"   {short(m):<14} n={len(rows):<3} per-agent acc A {ia:.3f} -> S {isx:.3f} ({isx-ia:+.3f})   "
              f"error corr phi A {phis['A']:.2f} -> S {phis['S']:.2f}")
    print(f"   pooled per-agent accuracy gain S-A: {t['diff']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}]  p={t['p_perm']:.4f}")
    out["mechanism_indiv"] = t

    # compute-matched, gemma only
    m = "google/gemma-4-31b-it"
    va10, vs, vs4 = verif.get((m, "A10"), {}), verif.get((m, "S"), {}), verif.get((m, "S4"), {})
    its = [it for it in ITEMS90 if it in va10 and it in vs and it in vs4]
    if its:
        f = lambda d: np.array([AG.label(cl, it, d[it].get("answer")) == 0 for it in its], float)
        a10, s5, s4 = f(va10), f(vs), f(vs4)
        print(f"\nCOMPUTE-MATCHED (gemma, n={len(its)}):  A10+V {a10.mean():.3f} | S+V {s5.mean():.3f} | S4+V {s4.mean():.3f}")
        for name, x in (("S+V  - A10+V", s5), ("S4+V - A10+V", s4)):
            d, lo, hi = AG.boot(x, a10)
            print(f"   {name}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        out["compute_matched_gemma"] = dict(n=len(its), A10V=a10.mean(), SV=s5.mean(), S4V=s4.mean(),
                                        items=its, SV_per_item=s5.tolist())

    json.dump(out, open("results/pooled_summary.json", "w"), indent=2, default=float)
    print("\nwrote results/pooled_summary.json")


if __name__ == "__main__":
    main()
