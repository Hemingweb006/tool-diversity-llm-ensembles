"""Reviewer control: is heterogeneity itself useful, or just the best tool?
Compares, on gemma-4-31B and identical items:
  S  (5 different substrates)  vs  HB (brute-force x5, the best single substrate)  vs  HP (python x5)  vs  A.
"""
import json
import numpy as np

import aggregate as AG
import pooled_analysis as P

P.SOURCES = {"google/gemma-4-31b-it": ["results/records_gemma_partial.jsonl", "results/records_final_homog.jsonl",
                                        "results/records_gemmaX_partial.jsonl", "results/records_final_s2.jsonl"]}
P.ARMS.update({"HB": ("HB", range(5)), "HP": ("HP", range(5)), "S2": ("S2", range(5)), "X": ("X", range(5))})
agents, verif, gold, cl = P.load_all()
m = "google/gemma-4-31b-it"
arms = ["A", "S", "HB", "HP"]
items = [it for it in P.ITEMS90 if it in gold and all(P.arm_labels(agents, cl, m, a, it) is not None for a in arms)]
res = {"n": len(items), "arms": {}, "contrasts": {}}
L = {a: np.array([P.arm_labels(agents, cl, m, a, it) for it in items]) for a in arms}
mv = {a: np.array([P.mv_expected(r) for r in L[a]]) for a in arms}
for a in arms:
    d = AG.diversity(L[a] == 0, L[a])
    res["arms"][a] = dict(indiv=float((L[a] == 0).mean()), mv=float(mv[a].mean()),
                          oracle=float((L[a] == 0).any(1).mean()), phi=d["phi"], ambiguity=d["ambiguity"])
for x, y in (("S", "HB"), ("S", "HP"), ("HB", "A"), ("HP", "A"), ("S", "A")):
    diff, lo, hi = AG.boot(mv[x], mv[y])
    t = P.stratified_tests({m: [dict(dx=a, dy=b) for a, b in zip(mv[x], mv[y])]})
    res["contrasts"][f"{x}-{y}"] = dict(diff=diff, lo=lo, hi=hi, p=t["p_perm"])
# independent replicate: S2 (second draw of S) vs X (second draw of A) -> a second, independent S-A estimate
rep_items = [it for it in P.ITEMS90 if it in gold and all(P.arm_labels(agents, cl, m, a, it) is not None for a in ("A", "S", "X", "S2"))]
if rep_items:
    g = lambda a: np.array([P.mv_expected(P.arm_labels(agents, cl, m, a, it)) for it in rep_items])
    A_, S_, X_, S2_ = g("A"), g("S"), g("X"), g("S2")
    d1 = AG.boot(S_, A_); d2 = AG.boot(S2_, X_); d3 = AG.boot(S_, S2_)
    both = AG.boot((S_ + S2_) / 2, (A_ + X_) / 2)
    res["replicate"] = dict(n=len(rep_items), SA_run1=d1, SA_run2=d2, S_vs_S2=d3, SA_avg_2runs=both,
                            S_flip=float(np.mean(np.abs(S_ - S2_) >= 0.5)), A_flip=float(np.mean(np.abs(A_ - X_) >= 0.5)))
    print(f"  replicate (n={len(rep_items)}): S-A run1 {d1[0]:+.3f} [{d1[1]:+.3f},{d1[2]:+.3f}] | "
          f"S2-X run2 {d2[0]:+.3f} [{d2[1]:+.3f},{d2[2]:+.3f}] | averaged over 2 runs {both[0]:+.3f} [{both[1]:+.3f},{both[2]:+.3f}] | "
          f"outcome flips S {res['replicate']['S_flip']:.0%} A {res['replicate']['A_flip']:.0%}")
json.dump(res, open("results/homog_summary.json", "w"), indent=1, default=float)
print(f"gemma-4-31B homogeneous controls, n={len(items)} identical items")
for a in arms:
    r = res["arms"][a]
    print(f"  {a:<3} indiv {r['indiv']:.3f}  MV {r['mv']:.3f}  oracle {r['oracle']:.3f}  phi {r['phi']:.2f}")
for k, r in res["contrasts"].items():
    print(f"  {k:<6} {r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]  p={r['p']:.3f}")
