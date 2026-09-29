"""Figure 1 of the paper: per-model effect of substrate agents (S) vs self-consistency (A).

Three small multiples sharing one model axis (ordered by A's per-agent accuracy):
 (a) majority-vote accuracy A -> S
 (b) per-agent accuracy A -> S         (augmentation)
 (c) mean pairwise error correlation   (decorrelation)
No dual axes; two categorical hues (reference palette slots 1 and 2, validated).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import aggregate as AG
import arms_v2 as AR
import pooled_analysis as P

A_COL, S_COL = "#2a78d6", "#eb6834"          # palette slots 1 (blue), 2 (orange)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"google/gemma-4-31b-it": "gemma-4-31B", "gemini-3.5-flash-lite": "gemini-3.5-flash-lite",
         "gemini-3.1-flash-lite": "gemini-3.1-flash-lite", "openai/gpt-oss-20b": "gpt-oss-20b",
         "meta/muse-glimmer-30b": "muse-glimmer-30B", "groq:qwen/qwen3.8-27b": "qwen3.8-27B"}

agents, verif, gold, cl = P.load_all()
rows = []
for m in P.SOURCES:
    pairs = P.paired(agents, cl, gold, m, "S")
    if not pairs:
        continue
    its = [r["it"] for r in pairs]
    phi = {}
    for arm in ("A", "S"):
        L = np.array([P.arm_labels(agents, cl, m, arm, it) for it in its])
        phi[arm] = AG.diversity(L == 0, L)["phi"]
    rows.append(dict(m=m, n=len(its),
                     mvA=np.mean([r["dy"] for r in pairs]), mvS=np.mean([r["dx"] for r in pairs]),
                     iA=np.mean([r["iy"] for r in pairs]), iS=np.mean([r["ix"] for r in pairs]),
                     phiA=phi["A"], phiS=phi["S"]))
# output tokens per item (A = 5 agents; S = 5 agents + code repairs), from every cache file
import glob, collections
SH = {"google/gemma-4-31b-it": ["gemma31b"], "gemini-3.5-flash-lite": ["gemflashlite"],
      "gemini-3.1-flash-lite": ["gem31flashlite"], "openai/gpt-oss-20b": ["gptoss20b", "gptoss20bgroq"]}
tok = collections.defaultdict(float)
for p in glob.glob("results/cache*.jsonl"):
    for l in open(p):
        try:
            d = json.loads(l)
        except Exception:
            continue
        k = d["key"].split("|")
        if len(k) >= 4 and k[1] in ("A", "S") and k[3].isdigit():
            tok[(k[0], k[1], k[2])] += d.get("usage", {}).get("completion_tokens", 0)
for r in rows:   # tokens per item, on this model's paired items
    its = [x["it"] for x in P.paired(agents, cl, gold, r["m"], "S")]
    for arm in ("A", "S"):
        r["tok" + arm] = float(np.mean([sum(tok[(s, arm, it)] for s in SH[r["m"]]) for it in its]))
rows.sort(key=lambda r: r["iA"])

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=True)
y = np.arange(len(rows))
panels = [("mvA", "mvS", "(a) Majority-vote accuracy", (0, 1)),
          ("iA", "iS", "(b) Per-agent accuracy", (0, 1)),
          ("phiA", "phiS", "(c) Error correlation $\\bar\\phi$ (lower = more diverse)", (0, 0.7))]
for ax, (ka, ks, title, xlim) in zip(axes, panels):
    for i, r in enumerate(rows):
        ax.plot([r[ka], r[ks]], [i, i], color=GRID, lw=2, zorder=1, solid_capstyle="round")
        ax.scatter(r[ka], i, s=64, color=A_COL, zorder=3, edgecolor="white", linewidth=1.5)
        ax.scatter(r[ks], i, s=64, color=S_COL, zorder=3, edgecolor="white", linewidth=1.5)
        if ka != "phiA":
            d = r[ks] - r[ka]
            ax.annotate(f"{d*100:+.1f}", (max(r[ka], r[ks]), i), xytext=(7, -3),
                        textcoords="offset points", color=INK2, fontsize=8)
    ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(y)
axes[0].set_yticklabels([f"{NAMES[r['m']]}  (n={r['n']})" for r in rows])
handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=A_COL, label="A: self-consistency (5 identical agents)"),
           plt.Line2D([], [], marker="o", ls="", ms=8, color=S_COL, label="S: substrate agents (text, Python, SymPy, brute force, solve-then-check)")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig("paper/figure1.pdf", bbox_inches="tight")
fig.savefig("paper/figure1.png", dpi=200, bbox_inches="tight")
json.dump(rows, open("results/figure1_data.json", "w"), indent=1, default=float)
print("wrote paper/figure1.pdf/.png")
for r in rows:
    print(f"  {NAMES[r['m']]:<24} n={r['n']:<3} MV {r['mvA']:.3f}->{r['mvS']:.3f}  indiv {r['iA']:.3f}->{r['iS']:.3f}  phi {r['phiA']:.2f}->{r['phiS']:.2f}  tokens A {r['tokA']:.0f} S {r['tokS']:.0f}")
