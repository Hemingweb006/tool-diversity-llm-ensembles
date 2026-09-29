"""Run the v2 ladder: MATH-500 L4-5 x {A, C, G, S} x 3 base models, + verifier stage.

Resumable: every LLM call is cached; rerunning the same command only makes the calls
that are missing or previously failed. Code is re-executed deterministically.
"""
import argparse, json, os, random, re, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import arms_v2 as AR
import grading as GR
import llm
import sandbox as SB

DATA = "data/matharena_2025_2026.json"   # AIME/HMMT/SMT/CMIMC/BRUMO 2025-26
OUT = "results/records.jsonl"
NO_VERIFY = False
_wlock = threading.Lock()


def load_items(n, offset=0, seed=2026):
    """2025-26 competition problems (post-cutoff for most models; older AIME/MATH were
    at ceiling or memorised). APEX is excluded: unsolvable by design, all-wrong items
    carry no diversity signal."""
    rows = [r for r in json.load(open(DATA)) if r["source"] != "apex_2025"]
    rows.sort(key=lambda r: r["item_id"])
    random.Random(seed).shuffle(rows)
    for r in rows:
        r["level"], r["subject"] = None, r["source"]
    return rows[offset:offset + n]


def _msgs(prompt, problem):
    return [{"role": "user", "content": f"{prompt}\n\nPROBLEM:\n{problem}"}]


def _call(key, model, msgs, keys, temperature=AR.T):
    return llm.call(key, model, msgs, temperature, max_tokens=AR.MAX_TOKENS, keys=keys)


def run_agent(model, arm, item, slot, cfg, keys):
    base = f"{AR.SHORT[model]}|{arm}|{item['item_id']}|{slot}"
    msgs = _msgs(cfg["prompt"], item["problem"])
    rec = _call(base, model, msgs, keys)
    out = dict(model=model, arm=arm, item_id=item["item_id"], slot=slot, tag=cfg["tag"],
               kind=cfg["kind"], error=rec.get("error"), truncated=rec.get("finish") == "length",
               tokens=rec.get("usage", {}).get("total_tokens", 0), repaired=False,
               code_ok=None, text_answer=None, answer=None, source=None)
    if rec.get("error"):
        return out, ""
    text = rec.get("text", "")

    if cfg["kind"] == "text":
        out["answer"] = GR.last_boxed(text)
        out["source"] = "boxed"
        return out, text

    # code / check: execute, one repair round on failure
    out["text_answer"] = GR.last_boxed(text) if cfg["kind"] == "check" else None
    res = SB.run(SB.extract_code(text))
    ans = GR.answer_line(res["stdout"])
    if not res["ok"] or ans is None:
        err = (res["stderr"] or "")[-1500:] or "The program printed no 'ANSWER:' line."
        if res["stdout"] and ans is None:
            err += f"\n\nstdout was:\n{res['stdout'][-800:]}"
        m2 = msgs + [{"role": "assistant", "content": text},
                     {"role": "user", "content": AR.REPAIR.format(err=err)}]
        rec2 = _call(base + "|repair", model, m2, keys)
        out["repaired"] = True
        out["tokens"] += rec2.get("usage", {}).get("total_tokens", 0)
        if not rec2.get("error"):
            text2 = rec2.get("text", "")
            res = SB.run(SB.extract_code(text2))
            ans = GR.answer_line(res["stdout"])
            text = text + "\n\n[REPAIRED]\n" + text2
    out["code_ok"] = bool(res["ok"] and ans is not None)
    out["exec_stdout"] = (res["stdout"] or "")[-600:]
    if ans is not None:
        out["answer"], out["source"] = ans, "code"
    elif out["text_answer"] is not None:
        out["answer"], out["source"] = out["text_answer"], "boxed-fallback"
    return out, text


def _parse_plans(text):
    """Planners write LaTeX (\\(, \\frac) inside JSON strings, which strict JSON rejects
    as invalid escapes. Treat every backslash not escaping a quote as literal; fall back
    to a regex over name/plan pairs."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return []
    raw = m.group(0)
    for candidate in (raw, re.sub(r'\\(?!")', r"\\\\", raw)):
        try:
            plans = json.loads(candidate)["plans"]
            plans = [p for p in plans if isinstance(p, dict) and p.get("plan")]
            if plans:
                return plans[:5]
        except Exception:
            pass
    pairs = re.findall(r'"name"\s*:\s*"(.*?)"\s*,\s*"plan"\s*:\s*"(.*?)"\s*[}\]]', raw, re.S)
    return [dict(name=n, plan=p) for n, p in pairs][:5]


def plan_duals(model, item, keys):
    """Planner -> 5 problem-specific solver configs. Falls back to C templates."""
    key = f"{AR.SHORT[model]}|G|{item['item_id']}|plan"
    for attempt in range(2):
        rec = _call(key + ("" if attempt == 0 else f"|retry{attempt}"), model,
                    [{"role": "user", "content": AR.PLANNER + item["problem"]}], keys)
        text = rec.get("text", "") or ""
        plans = _parse_plans(text)
        if len(plans) == 5:
            return [dict(tag=f"gen:{p.get('name', 'plan')[:40]}", kind="text",
                         prompt=AR.g_solver_prompt(p.get("name", "plan"), p["plan"]))
                    for p in plans], False, plans
    return [dict(c, tag="fallback:" + c["tag"]) for c in AR.ARM_C], True, []


def verify_stage(model, arm, item, agent_outs, texts, keys):
    order = list(range(len(agent_outs)))
    random.Random(f"{item['item_id']}|{arm}|{model}").shuffle(order)   # no position bias
    blocks = []
    for n, i in enumerate(order, 1):
        o, t = agent_outs[i], texts[i]
        body = t[-1800:] if t else "(no solution text)"
        extra = f"\nProgram output:\n{o.get('exec_stdout', '')[-300:]}" if o["kind"] != "text" else ""
        blocks.append(f"=== CANDIDATE {n} ===\nFinal answer: {o['answer']}\n"
                      f"Solution (excerpt):\n{body}{extra}")
    prompt = AR.VERIFIER.format(k=len(blocks), problem=item["problem"],
                                candidates="\n\n".join(blocks))
    rec = _call(f"{AR.SHORT[model]}|V{arm}|{item['item_id']}", model,
                [{"role": "user", "content": prompt}], keys, temperature=0.3)
    return dict(model=model, arm=f"V{arm}", item_id=item["item_id"], slot=0, tag="verifier",
                kind="verifier", error=rec.get("error"),
                answer=GR.last_boxed(rec.get("text", "")) if not rec.get("error") else None,
                truncated=rec.get("finish") == "length",
                tokens=rec.get("usage", {}).get("total_tokens", 0))


def run_task(model, arm, item, keys):
    fallback, plans = False, None
    cfgs = AR.ARMS.get(arm)
    if arm == "G":
        cfgs, fallback, plans = plan_duals(model, item, keys)
    outs, texts = [], []
    for slot, cfg in enumerate(cfgs):
        o, t = run_agent(model, arm, item, slot, cfg, keys)
        o["plan_fallback"] = fallback
        outs.append(o); texts.append(t)
    recs = outs[:]
    if not NO_VERIFY and all(not o.get("error") for o in outs):
        try:
            recs.append(verify_stage(model, arm, item, outs, texts, keys))
        except llm.NotCached:
            pass            # cache-only rebuild: keep the agents even if the verifier never ran
    for r in recs:                       # grading happens in the main thread (math-verify)
        r["gold"] = item["answer"]
        r["level"], r["subject"] = item["level"], item["subject"]
    if arm == "G" and plans:
        recs[0]["plans"] = plans
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=150)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--models", default=",".join(AR.MODELS))
    ap.add_argument("--arms", default="A,C,G,S")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--order", choices=["random", "item", "cached"], default="random",
                    help="item: finish all blocks of one item before the next -> complete "
                         "items accumulate early (useful when the run may be cut short)")
    ap.add_argument("--items-file", default=None,
                    help="JSON list of item_ids: run exactly these items (e.g. the 90 gemma items)")
    ap.add_argument("--no-verify", action="store_true", help="skip the verifier stage")
    ap.add_argument("--cache-only", action="store_true",
                    help="no API calls: rebuild records for tasks fully present in the cache")
    args = ap.parse_args()
    llm.CACHE_ONLY = args.cache_only

    models = [m for m in args.models.split(",") if m]
    arms = [a for a in args.arms.split(",") if a]
    items = load_items(args.n_items, args.offset)
    if args.items_file:
        wanted = json.load(open(args.items_file))
        pool = {it["item_id"]: it for it in load_items(10 ** 6)}
        items = [pool[i] for i in wanted if i in pool]
    global NO_VERIFY
    NO_VERIFY = args.no_verify
    tasks = [(m, a, it) for m in models for a in arms for it in items]
    random.Random(1).shuffle(tasks)
    if args.order == "item":
        pos = {it["item_id"]: i for i, it in enumerate(items)}
        tasks.sort(key=lambda t: pos[t[2]["item_id"]])
    est = len(tasks) * 6 + sum(1 for t in tasks if t[1] == "G") + \
        sum(1 for t in tasks if t[1] == "S") * 2
    print(f"{len(items)} items x {len(models)} models x arms {arms} = {len(tasks)} tasks, "
          f"~{est} calls (incl. verifier, planner, ~est. repairs)", flush=True)
    if args.dry_run:
        return

    os.makedirs("results", exist_ok=True)
    print(f"cache: {llm.load_cache()} good entries", flush=True)
    if args.order == "cached":
        # anytime mode: items with the most calls already cached first, so complete
        # items accumulate as fast as possible when the run may be cut short
        short = {m: AR.SHORT[m] for m in models}
        cnt = {}
        for k in llm._cache:
            p = k.split("|")
            if len(p) >= 3:
                cnt[(p[0], p[2])] = cnt.get((p[0], p[2]), 0) + 1
        pos = {it["item_id"]: i for i, it in enumerate(items)}
        tasks.sort(key=lambda t: (-cnt.get((short[t[0]], t[2]["item_id"]), 0), pos[t[2]["item_id"]]))
    keys = llm.load_keys()
    t0, done, allrecs = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_task, m, a, it, keys) for m, a, it in tasks]
        for f in as_completed(futs):
            try:
                recs = f.result()
            except llm.NotCached:
                continue
            except Exception as e:                       # never lose the whole run
                print(f"  task crashed: {type(e).__name__}: {e}", flush=True)
                continue
            for r in recs:
                r["correct"] = GR.correct(r["answer"], r["gold"]) if r.get("answer") else False
            allrecs += recs
            done += 1
            if done % 20 == 0 or done == len(tasks):
                el = time.time() - t0
                errs = sum(1 for r in allrecs if r.get("error"))
                print(f"  {done}/{len(tasks)} tasks  {el/60:6.1f}min  "
                      f"eta {(len(tasks)-done)/max(done/el,1e-9)/60:6.1f}min  "
                      f"errors={errs} 429s={llm.STATS['r429']} "
                      f"degenerate={llm.STATS.get('degenerate', 0)}", flush=True)

    with open(args.out, "w") as f:
        for r in sorted(allrecs, key=lambda r: (r["model"], r["arm"], r["item_id"], r["slot"])):
            f.write(json.dumps(r) + "\n")
    errs = sum(1 for r in allrecs if r.get("error"))
    print(f"wrote {args.out}: {len(allrecs)} records | {errs} errors")
    print("RUN_COMPLETE" if errs == 0 else "RUN_HAS_ERRORS (rerun the same command to retry)")


if __name__ == "__main__":
    main()
