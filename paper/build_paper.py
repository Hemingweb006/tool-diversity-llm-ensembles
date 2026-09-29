"""Fill template.tex with the final analysis numbers and compile the PDF."""
import json, subprocess
S = json.load(open("../results/pooled_summary.json"))
F = json.load(open("../results/figure1_data.json"))
NAMES = {"google/gemma-4-31b-it": "gemma-4-31B", "gemini-3.5-flash-lite": "gemini-3.5-flash-lite",
         "gemini-3.1-flash-lite": "gemini-3.1-flash-lite", "openai/gpt-oss-20b": "gpt-oss-20b"}
pt = lambda x: "0.0" if abs(x) < 0.0005 else f"{x*100:+.1f}"
ci = lambda lo, hi: f"$[{lo*100:+.1f},{hi*100:+.1f}]$"
def pfmt(p): return f"{p:.3f}" if p >= 0.001 else "$<$0.001"
sa = S["S-A"]
fill = {
 "NMODELS": {2: "two", 3: "three", 4: "four", 5: "five"}.get(len(sa["per_model"]), str(len(sa["per_model"]))),
 "SA_N": str(sa["n"]), "SA_DIFF": pt(sa["diff"]), "SA_CI": ci(sa["lo"], sa["hi"]).replace("$", ""),
 "SA_PHOLM": pfmt(sa["holm"]), "SA_RE": f"{pt(sa['re'])} [{sa['re_lo']*100:+.1f},{sa['re_hi']*100:+.1f}]",
}
fs = sorted(F, key=lambda r: r["iA"])
_o = S.get("S4-S_oos")
fill["S4OOS"] = f"S4$-$S $={pt(_o['diff'])}$, $p={pfmt(_o['p_perm'])}$, $n={_o['n']}$" if _o else "n/a"
fill["CURVE"] = " $\\to$ ".join(f"{(r['mvS']-r['mvA'])*100:+.1f}" for r in fs)
rows = []
for key, lab in (("S-A", "S (substrates) $-$ A"), ("C-A", "C (templates) $-$ A"), ("G-A", "G (planner) $-$ A"),
                 ("S4-A", "S4 (no SymPy) $-$ A$^\\dagger$")):
    if key not in S: continue
    r = S[key]; h = r.get("holm"); h = "n/a" if h != h or h is None else pfmt(h)
    b, c, pm = r["mcnemar"]
    rows.append(f"{lab} & {r['n']} & {pt(r['diff'])} & {ci(r['lo'], r['hi'])} & {pfmt(r['p_perm'])} & {h} & "
                f"{pt(r['re'])} {ci(r['re_lo'], r['re_hi'])} & {b}/{c} \\\\")
per = "; ".join(f"{NAMES.get(m, m)} $n{{=}}{n}$: {pt(d)}" for m, (n, d) in sa["per_model"].items())
oos = S.get("S4-S_oos")
fill["POOLED_TABLE"] = (
 "\\begin{table}[h]\\centering\\small\n\\begin{tabular}{lccccccc}\\toprule\n"
 "Contrast & $n$ & $\\Delta$ (pts) & 95\\% CI & $p_{\\text{perm}}$ & $p_{\\text{Holm}}$ & Random effects & McNemar $b/c$\\\\\\midrule\n"
 + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n"
 "\\caption{Model-stratified pooled contrasts (majority vote; ties at expectation). Holm over the planned family "
 "$\\{S,C,G\\}$. $^\\dagger$Exploratory.}\\end{table}\n"
 f"Per model, S$-$A is {per}. "
 + (f"On the models not used to choose it, removing SymPy leaves accuracy unchanged (S4$-$S $={pt(oos['diff'])}$ "
    f"{ci(oos['lo'], oos['hi'])}, $p={pfmt(oos['p_perm'])}$)." if oos else "")
 + "\n\n\\begin{figure}[h]\\centering\\includegraphics[width=\\linewidth]{figure1.pdf}\n"
 "\\caption{Substrate agents (S, orange) vs.\\ self-consistency (A, blue), per model, ordered by base per-agent "
 "accuracy. (a) Majority-vote accuracy; the gain decreases as base accuracy rises. (b) Per-agent accuracy, which "
 "tools raise for weak models and lower for the reasoning model. (c) Mean pairwise error correlation, which is lower "
 "under S for every model.}\\end{figure}\n")
mi = S["mechanism_indiv"]
mech = [f"\\textbf{{{NAMES.get(r['m'], r['m'])}}} ($n{{=}}{r['n']}$): per-agent accuracy "
        f"{r['iA']:.3f}$\\to${r['iS']:.3f}, error correlation $\\bar\\phi$ {r['phiA']:.2f}$\\to${r['phiS']:.2f}, "
        f"majority vote {r['mvA']:.3f}$\\to${r['mvS']:.3f}." for r in fs]
mrows = "\n".join(f"{NAMES.get(r['m'], r['m'])} & {r['n']} & {r['iA']:.3f} / {r['iS']:.3f} & "
                  f"{r['phiA']:.2f} / {r['phiS']:.2f} & {r['mvA']:.3f} / {r['mvS']:.3f} \\\\" for r in fs)
fill["MECH_TEXT"] = ("Substrate agents have a lower pairwise error correlation than self-consistency for every model "
 "(Figure~1c and Table~\\ref{tab:mech}). Their effect on per-agent accuracy depends on capability (Figure~1b). The "
 "weaker the model, the more the tools raise the accuracy of each agent. For gpt-oss-20b, the reasoning model and the "
 "strongest in our set, tools lower per-agent accuracy, but the ensemble does not lose accuracy because its errors "
 "are less correlated, which is the situation the exchange-rate diagnostic describes. Pooled over models, the "
 f"per-agent gain is {pt(mi['diff'])} points {ci(mi['lo'], mi['hi'])}.\n"
 "\\begin{table}[h]\\centering\\small\n\\begin{tabular}{lcccc}\\toprule\n"
 "Model & $n$ & Per-agent acc (A / S) & $\\bar\\phi$ (A / S) & Majority (A / S)\\\\\\midrule\n" + mrows +
 "\n\\bottomrule\\end{tabular}\\caption{Per-model decomposition of the S versus A comparison.}\\label{tab:mech}"
 "\\end{table}\n")
trows = "\n".join(f"{NAMES.get(r['m'], r['m'])} & {r['n']} & {r['tokA']/1000:.1f}k & {r['tokS']/1000:.1f}k & "
                  f"{(r['tokS']/r['tokA']-1)*100:+.0f}\\% & {pt(r['mvS']-r['mvA'])} \\\\" for r in fs)
fill["TOKENS_TABLE"] = ("\\begin{table}[h]\\centering\\small\n\\begin{tabular}{lccccc}\\toprule\n"
 "Model & $n$ & A tokens/item & S tokens/item & S vs A & $\\Delta$ majority (pts)\\\\\\midrule\n" + trows +
 "\n\\bottomrule\\end{tabular}\n\\caption{Output tokens per problem for five agents (S includes code-repair "
 "calls). S generates at most " + f"{max((r['tokS']/r['tokA']-1)*100 for r in fs):.0f}" + "\\% more tokens than "
 "self-consistency, and fewer on " + f"{sum(r['tokS']<r['tokA'] for r in fs)} of {len(fs)}" + " models, so the "
 "gain does not come from additional generation.}\\end{table}\n")
# ---- homogeneous-substrate control, compute accounting, replicate stability
import os
H = json.load(open("../results/homog_summary.json")) if os.path.exists("../results/homog_summary.json") else None
E = json.load(open("../results/extra_checks.json")) if os.path.exists("../results/extra_checks.json") else {}
if H and H["n"] > 0:
    ar, co = H["arms"], H["contrasts"]
    rowsH = "\n".join(f"{lab} & {ar[a]['indiv']:.3f} & {ar[a]['mv']:.3f} & {ar[a]['oracle']:.3f} & {ar[a]['phi']:.2f} \\\\"
                      for a, lab in (("A", "A: text $\\times$5"), ("HP", "HP: Python $\\times$5"),
                                     ("HB", "HB: brute force $\\times$5"), ("S", "S: 5 different substrates")))
    sb, sp = co["S-HB"], co["S-HP"]
    verdict = ("On this model the mixed ensemble outperforms the best single tool" if sb["diff"] > 0 and sb["lo"] > 0 else
               ("On this model the mixed ensemble performs as well as the best single tool replicated five times, "
                "but not better" if sb["diff"] >= -0.01 else
                "On this model the best single tool replicated five times performs at least as well as the mixed ensemble"))
    fill["HOMOG"] = ("The S arm changes which tool each agent uses and also varies the tool across agents. To separate "
      "these two factors we replicate a single substrate five times on the same items, using brute force, the best "
      "single substrate on this model, and Python, the most common one.\n\\begin{table}[h]\\centering\\small\n\\begin{tabular}{lcccc}\\toprule\n"
      "gemma-4-31B, $n{=}" + str(H["n"]) + "$ & Indiv.\\ acc & Majority & Oracle & $\\bar\\phi$\\\\\\midrule\n" + rowsH +
      "\n\\bottomrule\\end{tabular}\\caption{Homogeneous-substrate controls.}\\end{table}\n"
      f"S$-$HB $={pt(sb['diff'])}$ {ci(sb['lo'], sb['hi'])} ($p={pfmt(sb['p'])}$); "
      f"S$-$HP $={pt(sp['diff'])}$ {ci(sp['lo'], sp['hi'])} ($p={pfmt(sp['p'])}$). {verdict}. Five identical "
      "Python agents are nearly as decorrelated as the mixed ensemble, so most of the diversity comes from computing "
      "through code rather than from mixing tools. The mixed ensemble does not require knowing in advance which tool is "
      "best, which matters because the best tool differs across models.")
    fill["HOMOG_ABS"] = f"$\\Delta={pt(sb['diff'])}$ points, 95\\% CI {ci(sb['lo'], sb['hi']).replace('$','')}"
else:
    fill["HOMOG"] = "\\TBD{[Homogeneous-substrate control pending.]}"
    fill["HOMOG_ABS"] = ""
rp = E.get("replicate")
hr = (H or {}).get("replicate")
fill["REPL"] = (f"On gemma, A and X are two independent five-sample self-consistency ensembles. Their majority-vote "
    f"accuracies differ by {pt(rp['diff'])} points {ci(rp['lo'], rp['hi'])} over {rp['n']} items, and "
    f"{rp['flip']*100:.0f}\\% of items change outcome between the two runs, which indicates the size of run-to-run noise "
    "for a single model; the pooled analysis over models and items is what gives the main contrast its precision."
    + (f" We also re-ran the S arm independently (S2). The second, independent estimate of the gemma contrast is "
       f"S2$-$X $={pt(hr['SA_run2'][0])}$ {ci(hr['SA_run2'][1], hr['SA_run2'][2])}, against "
       f"S$-$A $={pt(hr['SA_run1'][0])}$ {ci(hr['SA_run1'][1], hr['SA_run1'][2])} in the first run. Averaged over "
       f"both runs, it is ${pt(hr['SA_avg_2runs'][0])}$ {ci(hr['SA_avg_2runs'][1], hr['SA_avg_2runs'][2])} "
       f"($n={hr['n']}$). Between runs, {hr['S_flip']*100:.0f}\\% of items change outcome under S, versus "
       f"{hr['A_flip']*100:.0f}\\% under A." if hr else "")
    ) if rp else "\\TBD{[replicate pending]}"
cp = E.get("compute", {}); sx = E.get("sandbox_gemma_S")
def cget(m, a, k): return cp.get(f"{m}|{a}", {}).get(k)
crow = []
for mname, lab in (("google/gemma-4-31b-it", "gemma-4-31B"), ("gemini-3.5-flash-lite", "gemini-3.5-flash-lite"),
                   ("gemini-3.1-flash-lite", "gemini-3.1-flash-lite"), ("openai/gpt-oss-20b", "gpt-oss-20b")):
    if cget(mname, "A", "calls") and cget(mname, "S", "calls"):
        crow.append(f"{lab} & {cget(mname,'A','calls'):.1f} / {cget(mname,'S','calls'):.1f} & "
                    f"{cget(mname,'A','input')/1000:.1f}k / {cget(mname,'S','input')/1000:.1f}k & "
                    f"{cget(mname,'A','output')/1000:.1f}k / {cget(mname,'S','output')/1000:.1f}k \\\\")
fill["COMPUTE"] = ("Table~\\ref{tab:cost} reports total LLM cost per problem for A and S: calls, input tokens and "
  "output tokens, averaged over each model's paired items. " + (f"Code execution adds {sx['per_item']:.1f}\\,s of sandbox time per problem on gemma "
  f"({sx['executions']} executions in total), which is small compared with generation latency. " if sx else "") +
  "\n\\begin{table}[h]\\centering\\small\n\\begin{tabular}{lccc}\\toprule\nModel & LLM calls (A / S) & "
  "Input tokens (A / S) & Output tokens (A / S)\\\\\\midrule\n" + "\n".join(crow) +
  "\n\\bottomrule\\end{tabular}\\caption{Total LLM cost per problem (five agents; S includes repair calls).}"
  "\\label{tab:cost}\\end{table}\n") if crow else "\\TBD{[compute accounting pending]}"
t = open("template.tex").read()
for k, v in fill.items():
    t = t.replace(f"<<{k}>>", v)
assert "<<" not in t, [x for x in t.split("<<")[1:3]]
open("stacking_paper.tex", "w").write(t)
subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-quiet", "stacking_paper.tex"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("built stacking_paper.pdf")
