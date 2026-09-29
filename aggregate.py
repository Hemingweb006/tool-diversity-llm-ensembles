"""v2 analysis: diversity, aggregation and the capability ladder.

For every (model, arm) with 5 agents:
  accuracy (mean individual, best slot, oracle), error decorrelation (phi, Q,
  effective number of independent agents), Krogh-Vedelsby/Brier ambiguity,
  and six aggregators:
    MV        majority vote over answer-equivalence clusters (abstentions don't vote)
    NB        log-odds weighted vote, weights = logit(slot accuracy)         [cross-fitted]
    CORR      correlation-adjusted weights = (R + lam I)^-1 logit(p), >= 0   [cross-fitted]
    VERIFIER  the base model reads the 5 candidates and picks one
    STOP      adaptive stopping: consult agents best-first, stop once 2 agree
    ORACLE    any agent correct (upper bound for any selector)
Exchange rate per model: d(ambiguity) / (2 * d(individual accuracy lost)) vs A;
the arm pays for itself iff rate > 1 (or it loses no accuracy at all).
"""
import argparse, itertools, json, math, os
from collections import defaultdict

import numpy as np

import arms_v2 as AR
import grading as GR

K = 5
FOLDS = 5
LAM = 0.5


# ------------------------------------------------------------------ loading + clustering
def load(path):
    recs = [json.loads(l) for l in open(path)]
    agents = defaultdict(lambda: defaultdict(dict))     # (model,arm) -> item -> slot -> rec
    verif = defaultdict(dict)                           # (model,arm) -> item -> rec
    gold, meta = {}, {}
    for r in recs:
        gold[r["item_id"]] = r["gold"]
        meta[r["item_id"]] = (r.get("level"), r.get("subject"))
        if r["arm"].startswith("V"):
            verif[(r["model"], r["arm"][1:])][r["item_id"]] = r
        else:
            agents[(r["model"], r["arm"])][r["item_id"]][r["slot"]] = r
    return recs, agents, verif, gold, meta


def build_clusters(recs, gold, cache_path):
    """Global answer-equivalence clusters per item across ALL models/arms, so pooled
    ensembles vote over a common label space. gold -> cluster 0; None -> -1."""
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    by_item = defaultdict(set)
    for r in recs:
        if r.get("answer"):
            by_item[r["item_id"]].add(r["answer"])
    out = {}
    for it, answers in by_item.items():
        key = it + "::" + str(hash(tuple(sorted(answers))))
        if key in cache:
            out[it] = cache[key]
            continue
        uniq = sorted(answers)
        ids = GR.cluster(uniq, gold=gold[it])
        out[it] = dict(zip(uniq, ids))
        cache[key] = out[it]
    json.dump(cache, open(cache_path, "w"))
    return out


def label(clusters, it, ans):
    if not ans:
        return -1
    return clusters.get(it, {}).get(ans, -1)


# ------------------------------------------------------------------ core matrices
def matrix(agents, key, items, clusters):
    """-> L (n x K) cluster labels, C (n x K) correctness (label == 0)."""
    L = np.full((len(items), K), -1, dtype=int)
    for i, it in enumerate(items):
        for s in range(K):
            r = agents[key][it].get(s)
            if r is not None:
                L[i, s] = label(clusters, it, r.get("answer"))
    return L, (L == 0)


def weighted_vote(Lrow, w):
    score = defaultdict(float)
    for s, lab in enumerate(Lrow):
        if lab != -1:
            score[lab] += w[s]
    if not score:
        return -1
    best = max(score.values())
    for s, lab in enumerate(Lrow):                      # tie -> lowest slot
        if lab != -1 and abs(score[lab] - best) < 1e-12:
            return lab
    return -1


def logit(p):
    p = np.clip(p, 0.05, 0.95)
    return np.log(p / (1 - p))


def fit_weights(C_train, kind):
    n = len(C_train)
    p = (C_train.sum(0) + 1) / (n + 2)                  # Laplace-smoothed slot accuracy
    if kind == "nb":
        return np.maximum(logit(p), 0.0)
    E = (~C_train).astype(float)                        # error indicators
    if E.std(0).min() < 1e-9:
        R = np.eye(K)
    else:
        R = np.corrcoef(E.T)
        R = np.nan_to_num(R, nan=0.0)
        np.fill_diagonal(R, 1.0)
    w = np.linalg.solve(R + LAM * np.eye(K), np.maximum(logit(p), 0.0))
    return np.maximum(w, 0.0) if w.max() > 0 else np.ones(K)


def crossfit_predictions(L, C, kind, seed=0):
    n = len(L)
    idx = np.random.RandomState(seed).permutation(n)
    pred = np.full(n, -1)
    for f in range(FOLDS):
        test = idx[f::FOLDS]
        train = np.setdiff1d(idx, test)
        w = fit_weights(C[train], kind) if kind != "mv" else np.ones(K)
        for i in test:
            pred[i] = weighted_vote(L[i], w)
    return pred


def adaptive_stop(L, C, seed=0):
    """Best-first order learned on training folds; stop once 2 agents agree,
    or once a cluster reaches 3 votes; else MV over all 5. -> (pred, calls)"""
    n = len(L)
    idx = np.random.RandomState(seed).permutation(n)
    pred, calls = np.full(n, -1), np.zeros(n)
    for f in range(FOLDS):
        test = idx[f::FOLDS]
        train = np.setdiff1d(idx, test)
        order = np.argsort(-C[train].mean(0), kind="stable")
        for i in test:
            votes, used, chosen = defaultdict(int), 0, None
            for s in order:
                used += 1
                lab = L[i, s]
                if lab != -1:
                    votes[lab] += 1
                    if votes[lab] >= 2 and (used == 2 or votes[lab] >= 3):
                        chosen = lab
                        break
            if chosen is None:
                chosen = weighted_vote(L[i], np.ones(K))
            pred[i], calls[i] = chosen, used
    return pred, calls


def diversity(C, L):
    E = ~C
    phis, qs = [], []
    for a, b in itertools.combinations(range(K), 2):
        x, y = C[:, a], C[:, b]
        n11, n00 = int((x & y).sum()), int((~x & ~y).sum())
        n10, n01 = int((x & ~y).sum()), int((~x & y).sum())
        den = n11 * n00 + n01 * n10
        qs.append((n11 * n00 - n01 * n10) / den if den else np.nan)
        d = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
        phis.append((n11 * n00 - n10 * n01) / d if d else np.nan)
    phi = float(np.nanmean(phis)) if not all(np.isnan(phis)) else float("nan")
    q = float(np.nanmean(qs)) if not all(np.isnan(qs)) else float("nan")
    # Brier / Krogh-Vedelsby on the cluster simplex (abstain = its own wrong answer)
    amb_items, ind_items, ens_items = [], [], []
    for row in L:
        labs = sorted(set(row.tolist()) | {0})
        pos = {l: j for j, l in enumerate(labs)}
        P = np.zeros((K, len(labs)))
        for s, l in enumerate(row):
            P[s, pos[l]] = 1
        y = np.zeros(len(labs)); y[pos[0]] = 1
        pbar = P.mean(0)
        ind_items.append(((P - y) ** 2).sum(1).mean())
        ens_items.append(((pbar - y) ** 2).sum())
        amb_items.append(((P - pbar) ** 2).sum(1).mean())
    eff_n = K / (1 + (K - 1) * phi) if not math.isnan(phi) else float("nan")
    return dict(phi=phi, Q=q, eff_n=eff_n, ambiguity=float(np.mean(amb_items)),
                ind_err=float(np.mean(ind_items)), ens_err=float(np.mean(ens_items)),
                identity_residual=float(abs(np.mean(ind_items) - np.mean(ens_items) - np.mean(amb_items))),
                amb_items=np.array(amb_items))


def arm_block(agents, verif, key, items, clusters):
    L, C = matrix(agents, key, items, clusters)
    mv = crossfit_predictions(L, C, "mv")
    nb = crossfit_predictions(L, C, "nb")
    co = crossfit_predictions(L, C, "corr")
    st, calls = adaptive_stop(L, C)
    ver = np.array([label(clusters, it, verif.get(key, {}).get(it, {}).get("answer")) == 0
                    for it in items])
    d = diversity(C, L)
    recs = [agents[key][it][s] for it in items for s in range(K) if s in agents[key][it]]
    kinds = [r for r in recs if r["kind"] != "text"]
    return dict(
        L=L, C=C, per_item=dict(mv=mv == 0, nb=nb == 0, corr=co == 0, stop=st == 0,
                                verifier=ver, oracle=C.any(1), amb=d.pop("amb_items")),
        n=len(items), slot_acc=C.mean(0).tolist(),
        mean_ind=float(C.mean()), best_slot=float(C.mean(0).max()),
        mv=float((mv == 0).mean()), nb=float((nb == 0).mean()), corr=float((co == 0).mean()),
        verifier=float(ver.mean()), stop=float((st == 0).mean()), stop_calls=float(calls.mean()),
        oracle=float(C.any(1).mean()),
        abstain=float((L == -1).mean()),
        truncated=float(np.mean([r.get("truncated", False) for r in recs])),
        code_ok=float(np.mean([bool(r.get("code_ok")) for r in kinds])) if kinds else None,
        repaired=float(np.mean([bool(r.get("repaired")) for r in kinds])) if kinds else None,
        plan_fallback=float(np.mean([bool(r.get("plan_fallback")) for r in recs])),
        tokens=float(np.mean([r.get("tokens", 0) for r in recs])) * K,
        **d)


def pooled(agents, keys, items, clusters, kind="mv"):
    """Ensemble over several (model, arm) blocks: MV or cross-fitted NB weights."""
    Ls, Cs = zip(*[matrix(agents, k, items, clusters) for k in keys])
    L, C = np.hstack(Ls), np.hstack(Cs)
    n, m = L.shape
    idx = np.random.RandomState(0).permutation(n)
    pred = np.full(n, -1)
    for f in range(FOLDS):
        test = idx[f::FOLDS]; train = np.setdiff1d(idx, test)
        if kind == "mv":
            w = np.ones(m)
        else:
            p = (C[train].sum(0) + 1) / (len(train) + 2)
            w = np.maximum(logit(p), 0.0)
        for i in test:
            score = defaultdict(float)
            for s, lab in enumerate(L[i]):
                if lab != -1:
                    score[lab] += w[s]
            pred[i] = max(score, key=score.get) if score else -1
    return pred == 0


def boot(x, y=None, reps=10000, seed=7):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, float)
    d = x - (np.asarray(y, float) if y is not None else 0)
    n = len(d)
    bs = d[rng.randint(0, n, size=(reps, n))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


# ------------------------------------------------------------------ report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="results/records.jsonl")
    ap.add_argument("--json-out", default="results/summary_v2.json")
    args = ap.parse_args()

    recs, agents, verif, gold, meta = load(args.records)
    clusters = build_clusters(recs, gold, "results/clusters_cache.json")
    present = {k[0] for k in agents}
    models = [m for m in AR.MODELS if m in present] + sorted(present - set(AR.MODELS))
    arms = [a for a in AR.ARM_ORDER if any(k[1] == a for k in agents)]
    AR.SHORT.setdefault("x", "x")

    # common items: complete (5 agents, no errors) for every (model, arm) present
    items = sorted(gold)
    for k in agents:
        items = [it for it in items if it in agents[k] and len(agents[k][it]) == K
                 and not any(r.get("error") for r in agents[k][it].values())]
    print(f"\n2025-26 competitions | {len(items)} complete items | models: "
          + ", ".join(AR.SHORT[m] for m in models))

    S = {}
    for m in models:
        for a in arms:
            if (m, a) in agents:
                S[(m, a)] = arm_block(agents, verif, (m, a), items, clusters)

    hdr = (f"{'model':<10}{'arm':<20}{'indiv':>7}{'best':>7}{'MV':>7}{'NB':>7}{'CORR':>7}"
           f"{'VERIF':>7}{'STOP':>7}{'calls':>6}{'orac':>7}{'ambig':>7}{'phi':>7}{'effN':>6}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for m in models:
        for a in arms:
            s = S.get((m, a))
            if not s:
                continue
            print(f"{AR.SHORT[m]:<10}{AR.LABELS[a]:<20}{s['mean_ind']:>7.3f}{s['best_slot']:>7.3f}"
                  f"{s['mv']:>7.3f}{s['nb']:>7.3f}{s['corr']:>7.3f}{s['verifier']:>7.3f}"
                  f"{s['stop']:>7.3f}{s['stop_calls']:>6.2f}{s['oracle']:>7.3f}{s['ambiguity']:>7.3f}"
                  f"{s['phi']:>7.3f}{s['eff_n']:>6.2f}")
        print()

    print("operational: abstain / truncated / code ran OK / repaired / planner fallback / tokens per item")
    for (m, a), s in S.items():
        f = lambda v: "   -  " if v is None else f"{v:6.2f}"
        print(f"  {AR.SHORT[m]:<10}{a}  {s['abstain']:6.2f} {s['truncated']:6.2f} {f(s['code_ok'])} "
              f"{f(s['repaired'])} {s['plan_fallback']:6.2f} {s['tokens']:9.0f}")

    print("\nper-slot accuracy")
    for (m, a), s in S.items():
        tags = [agents[(m, a)][items[0]][k]["tag"].split(":")[-1][:14] if a != "G" else f"g{k}"
                for k in range(K)]
        print(f"  {AR.SHORT[m]:<10}{a}  " + "  ".join(f"{t}={v:.2f}" for t, v in zip(tags, s["slot_acc"])))

    print("\nexchange rate vs A  (pays iff rate > 1, or no accuracy lost)")
    rates = {}
    for m in models:
        A = S.get((m, "A"))
        if not A:
            continue
        for a in [x for x in arms if x != "A"]:
            X = S.get((m, a))
            if not X:
                continue
            loss = A["mean_ind"] - X["mean_ind"]
            gain = X["ambiguity"] - A["ambiguity"]
            rate = gain / (2 * loss) if loss > 1e-9 else float("inf")
            rates[f"{AR.SHORT[m]}:{a}"] = dict(acc_loss=loss, amb_gain=gain, rate=rate)
            verdict = "FREE diversity" if loss <= 1e-9 and gain > 0 else (
                "pays" if rate > 1 else "does not pay")
            print(f"  {AR.SHORT[m]:<10}{a}: acc loss {loss:+.3f}  ambiguity {gain:+.3f}  "
                  f"rate {rate:6.2f}  -> {verdict}")

    print("\nbootstrap vs A, same model (10k resamples, 95% CI)   * = excludes 0")
    boots = {}
    for m in models:
        if (m, "A") not in S:
            continue
        A = S[(m, "A")]["per_item"]
        for a in [x for x in arms if x != "A"]:
            if (m, a) not in S:
                continue
            X = S[(m, a)]["per_item"]
            for stat in ("mv", "corr", "verifier", "amb"):
                d, lo, hi = boot(X[stat], A[stat] if stat != "amb" else A["amb"])
                boots[f"{AR.SHORT[m]}:{a}-A:{stat}"] = (d, lo, hi)
                star = " *" if lo > 0 or hi < 0 else ""
                print(f"  {AR.SHORT[m]:<10}{a}-A  {stat:<9}{d:+.3f}  [{lo:+.3f}, {hi:+.3f}]{star}")

    print("\npooled ensembles (cross-fitted NB weights; MV in brackets)")
    pools = {}
    for m in models:
        combos = [("A+S", ["A", "S"]), ("A+G", ["A", "G"]), ("A+C", ["A", "C"]),
                  ("G+S", ["G", "S"]), ("all 4 arms", arms)]
        for name, ar in combos:
            ks = [(m, a) for a in ar if (m, a) in agents]
            if len(ks) != len(ar):
                continue
            w = pooled(agents, ks, items, clusters, "nb")
            mv = pooled(agents, ks, items, clusters, "mv")
            pools[f"{AR.SHORT[m]}:{name}"] = dict(nb=float(w.mean()), mv=float(mv.mean()),
                                                  n_agents=K * len(ks))
            print(f"  {AR.SHORT[m]:<10}{name:<12}{K*len(ks):>3} agents  {w.mean():.3f}  [{mv.mean():.3f}]")
    for a in arms:
        ks = [(m, a) for m in models if (m, a) in agents]
        if len(ks) == len(models) > 1:
            w = pooled(agents, ks, items, clusters, "nb")
            mv = pooled(agents, ks, items, clusters, "mv")
            pools[f"cross-model:{a}"] = dict(nb=float(w.mean()), mv=float(mv.mean()),
                                             n_agents=K * len(ks))
            print(f"  cross-model {a:<9}{K*len(ks):>3} agents  {w.mean():.3f}  [{mv.mean():.3f}]")

    out = dict(n_items=len(items), models=[AR.SHORT[m] for m in models], arms=arms,
               stats={f"{AR.SHORT[m]}:{a}": {k: v for k, v in s.items() if k not in ("L", "C", "per_item")}
                      for (m, a), s in S.items()},
               exchange_rates=rates, bootstrap=boots, pooled=pools)
    json.dump(out, open(args.json_out, "w"), indent=2, default=float)
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
