"""Compute-matched baseline: a verifier over the 10 neutral samples A+X (= A10 + V).

This answers the obvious reviewer objection: does S+verifier win only because it
spends more compute? A10+V spends MORE than S+V (10 solver calls + a longer verifier
prompt), so if S+V still matches or beats it, the gain is not bought with compute.
"""
import argparse, json, random
from concurrent.futures import ThreadPoolExecutor, as_completed

import arms_v2 as AR
import grading as GR
import llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items-file", default="data/items90.json")
    ap.add_argument("--workers", type=int, default=40)
    args = ap.parse_args()

    short = AR.SHORT[args.model]
    items = {r["item_id"]: r for r in json.load(open("data/matharena_2025_2026.json"))}
    wanted = json.load(open(args.items_file))
    llm.load_cache()
    keys = llm.load_keys()

    def cand_texts(it):
        out = []
        for arm in ("A", "X"):
            for s in range(5):
                rec = llm._cache.get(f"{short}|{arm}|{it}|{s}")
                if rec is None:
                    return None
                out.append(rec.get("text", ""))
        return out

    def run(it):
        texts = cand_texts(it)
        if texts is None:
            return None
        order = list(range(10))
        random.Random(f"{it}|A10|{args.model}").shuffle(order)
        blocks = [f"=== CANDIDATE {n} ===\nFinal answer: {GR.last_boxed(texts[i])}\n"
                  f"Solution (excerpt):\n{texts[i][-1800:]}" for n, i in enumerate(order, 1)]
        prompt = AR.VERIFIER.format(k=10, problem=items[it]["problem"], candidates="\n\n".join(blocks))
        rec = llm.call(f"{short}|VA10|{it}", args.model, [{"role": "user", "content": prompt}],
                       0.3, max_tokens=AR.MAX_TOKENS, keys=keys)
        return dict(model=args.model, arm="VA10", item_id=it, slot=0, tag="verifier10",
                    kind="verifier", error=rec.get("error"), gold=items[it]["answer"],
                    answer=None if rec.get("error") else GR.last_boxed(rec.get("text", "")),
                    tokens=rec.get("usage", {}).get("total_tokens", 0))

    recs, missing = [], 0
    with ThreadPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(run, it) for it in wanted]):
            r = f.result()
            if r is None:
                missing += 1
                continue
            r["correct"] = GR.correct(r["answer"], r["gold"]) if r["answer"] else False   # main thread
            recs.append(r)
    out = f"results/records_v10_{short}.jsonl"
    with open(out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    errs = sum(1 for r in recs if r.get("error"))
    print(f"{out}: {len(recs)} verified | {missing} items missing A/X samples | {errs} errors | "
          f"acc {sum(r['correct'] for r in recs)/max(len(recs),1):.3f}")


if __name__ == "__main__":
    main()
