"""Offline reviewer checks (no API calls):
 (1) run-to-run stability: gemma A and X are two independent 5-agent self-consistency ensembles;
 (2) full compute accounting per arm: LLM calls, input tokens, output tokens, code executions, sandbox time."""
import json, glob, time, collections, numpy as np
import pooled_analysis as P, aggregate as AG, sandbox as SB
agents, verif, gold, cl = P.load_all()
items = [it for it in P.ITEMS90 if it in gold]
G = "google/gemma-4-31b-it"
out = {}
# (1) replicate stability
a = np.array([P.mv_expected(P.labels(agents, cl, G, "A", it)) for it in items if P.labels(agents, cl, G, "X", it) is not None])
x = np.array([P.mv_expected(P.labels(agents, cl, G, "X", it)) for it in items if P.labels(agents, cl, G, "X", it) is not None])
d, lo, hi = AG.boot(a, x)
flip = float(np.mean(np.abs(a - x) >= 0.5))
print(f"(1) gemma replicate A vs X (two independent self-consistency runs, n={len(a)}): "
      f"MV {a.mean():.3f} vs {x.mean():.3f}, diff {d:+.3f} [{lo:+.3f},{hi:+.3f}], items flipping outcome {flip:.1%}")
out["replicate"] = dict(n=len(a), A=a.mean(), X=x.mean(), diff=d, lo=lo, hi=hi, flip=flip)
# (2) compute accounting from every cache file
calls = collections.defaultdict(lambda: collections.Counter())
SH = {"gemma31b": G, "gemflashlite": "gemini-3.5-flash-lite", "gem31flashlite": "gemini-3.1-flash-lite",
      "gptoss20b": "openai/gpt-oss-20b", "gptoss20bgroq": "openai/gpt-oss-20b"}
seen = set()
PAIRED = {mm: {r["it"] for r in P.paired(agents, cl, gold, mm, "S")} for mm in set(SH.values())}
for p in glob.glob("results/cache*.jsonl"):
    for l in open(p):
        try: dd = json.loads(l)
        except Exception: continue
        k = dd["key"].split("|")
        if dd["key"] in seen or len(k) < 3 or k[0] not in SH or k[2] not in set(items): continue
        seen.add(dd["key"])
        arm = k[1]
        if arm in ("A", "S") and k[2] not in PAIRED.get(SH[k[0]], set()): continue
        if arm not in ("A", "S", "X", "VA10", "VS", "HB", "HP"): continue
        u = dd.get("usage", {})
        c = calls[(SH[k[0]], arm)]
        c["calls"] += 1; c["in"] += u.get("prompt_tokens", 0); c["out"] += u.get("completion_tokens", 0)
        c["items"] = len(items)
# code executions + sandbox time for gemma S (re-executed deterministically, offline)
import run_v2 as RV
t_exec, n_exec = 0.0, 0
llm = RV.llm; llm.load_cache(); llm.CACHE_ONLY = True
for it in items[:90]:
    for slot in (1, 2, 3, 4):
        rec = llm._cache.get(f"gemma31b|S|{it}|{slot}")
        if not rec: continue
        code = SB.extract_code(rec.get("text", ""))
        t = time.time(); SB.run(code); t_exec += time.time() - t; n_exec += 1
        rr = llm._cache.get(f"gemma31b|S|{it}|{slot}|repair")
        if rr:
            t = time.time(); SB.run(SB.extract_code(rr.get("text", ""))); t_exec += time.time() - t; n_exec += 1
print(f"(2) gemma S: {n_exec} code executions, total sandbox time {t_exec:.0f}s = {t_exec/len(items):.1f}s per problem")
out["sandbox_gemma_S"] = dict(executions=n_exec, seconds=t_exec, per_item=t_exec / len(items))
print(f"    per problem (mean over the {len(items)} items): LLM calls | input tokens | output tokens")
# paired items per model: average only over items where both A and S are complete for that model
paired_n = {mm: len(P.paired(agents, cl, gold, mm, "S")) for mm in set(SH.values())}
for (m, arm), c in sorted(calls.items()):
    n = max(paired_n.get(m, len(items)), 1)
    print(f"    {m:<24}{arm:<6}{c['calls']/n:6.2f} |{c['in']/n:9.0f} |{c['out']/n:9.0f}")
    out.setdefault("compute", {})[f"{m}|{arm}"] = dict(calls=c["calls"]/n, input=c["in"]/n, output=c["out"]/n)
json.dump(out, open("results/extra_checks.json", "w"), indent=1, default=float)
