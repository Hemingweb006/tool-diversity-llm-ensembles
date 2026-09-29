"""Verifier over S minus SymPy (4 substrate candidates) = recipe 'S4 + verifier'.

Agents are rebuilt from the cache (code is re-executed deterministically in the
sandbox, no new solver calls); only the verifier call is new.
"""
import argparse, json
from concurrent.futures import ThreadPoolExecutor, as_completed

import arms_v2 as AR
import grading as GR
import llm
import run_v2 as RV

SYMPY_SLOT = 2          # arms_v2.ARM_S order: text-cot, python, sympy-symbolic, brute-force, solve-then-check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items-file", default="data/items90.json")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--cache-only", action="store_true")
    args = ap.parse_args()
    items = {r["item_id"]: r for r in json.load(open("data/matharena_2025_2026.json"))}
    wanted = json.load(open(args.items_file))
    llm.load_cache()
    keys = llm.load_keys()

    def run(it):
        item = dict(items[it])
        outs, texts = [], []
        llm.CACHE_ONLY = args.cache_only
        for slot, cfg in enumerate(AR.ARM_S):
            if slot == SYMPY_SLOT:
                continue
            if f"{AR.SHORT[args.model]}|S|{it}|{slot}" not in llm._cache:
                return None
            try:
                o, t = RV.run_agent(args.model, "S", item, slot, cfg, keys)
            except llm.NotCached:
                return None
            if o.get("error"):
                return None
            outs.append(o); texts.append(t)
        try:
            v = RV.verify_stage(args.model, "S4", item, outs, texts, keys)
        except llm.NotCached:
            return None
        v["gold"] = item["answer"]
        return v

    recs, missing = [], 0
    with ThreadPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(run, it) for it in wanted]):
            r = f.result()
            if r is None:
                missing += 1
                continue
            r["correct"] = GR.correct(r["answer"], r["gold"]) if r["answer"] else False
            recs.append(r)
    out = f"results/records_vs4_{AR.SHORT[args.model]}.jsonl"
    with open(out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(f"{out}: {len(recs)} verified | {missing} missing | "
          f"{sum(1 for r in recs if r.get('error'))} errors | acc {sum(r['correct'] for r in recs)/max(len(recs),1):.3f}")


if __name__ == "__main__":
    main()
