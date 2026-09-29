"""Compare our best gemma system (substrate agents + verifier) with the MathArena leaderboard.

MathArena reports single-attempt accuracy (mean over 4 runs per problem) of single models.
For each leaderboard model we use only the problems it shares with our item set, and report
our accuracy on exactly those problems. Needs results/pooled_summary.json (run pooled_analysis.py first).
Offline: reads data/matharena_leaderboard/*.csv (public per-problem results from MathArena).
"""
import csv, glob, json
from collections import defaultdict

S = json.load(open("results/pooled_summary.json"))["compute_matched_gemma"]
ours = dict(zip(S["items"], S["SV_per_item"]))          # item id -> 1.0 / 0.0 for gemma S x5 + verifier

runs = defaultdict(lambda: defaultdict(list))            # model -> item id -> [correct per run]
names = {}
for p in sorted(glob.glob("data/matharena_leaderboard/*.csv")):
    for r in csv.DictReader(open(p)):
        it = f"{r['comp']}_{r['problem_idx']}"
        if it in ours:
            runs[r["model_config"]][it].append(r["correct"].strip().lower() == "true")
            names[r["model_config"]] = r["model_name"]

rows = []
for cfg, per in runs.items():
    its = sorted(per)
    if len(its) < 40:
        continue
    theirs = sum(sum(v) / len(v) for v in (per[i] for i in its)) / len(its)
    mine = sum(ours[i] for i in its) / len(its)
    rows.append((theirs, names[cfg], len(its), mine))
rows.sort(reverse=True)

print(f"Our system: gemma-4-31B, 5 substrate agents + verifier, {len(ours)} items, accuracy {S['SV']:.3f}")
print(f"{'MathArena model':<34}{'n':>4}{'theirs':>9}{'ours':>8}{'diff':>8}")
for theirs, name, n, mine in rows:
    print(f"{name:<34}{n:>4}{theirs:>9.3f}{mine:>8.3f}{mine - theirs:>+8.3f}")
beaten = sum(m > t for t, _, _, m in rows)
print(f"\nours is higher than {beaten}/{len(rows)} leaderboard models (>= 40 shared problems)")
json.dump([dict(model=n, n=k, theirs=t, ours=m) for t, n, k, m in rows],
          open("results/leaderboard_comparison.json", "w"), indent=1)
