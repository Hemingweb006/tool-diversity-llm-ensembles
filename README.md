# Computing Through Tools Decorrelates Same-Model LLM Ensembles

Code, data and cached model outputs for the paper by **Reda Louriki** ([`paper/stacking_paper.pdf`](paper/stacking_paper.pdf)).

Agents built on the same LLM tend to make the same mistakes, which limits what majority voting can gain. We compare several ways of making five agents of one model disagree usefully on recent competition math problems (MathArena, 2025 and 2026). Letting each agent compute through a different tool (plain text, executed Python, SymPy, brute-force search, solve-then-check code) improves majority-vote accuracy over self-consistency by **+8.3 points** (95% CI [3.7, 12.8], Holm-adjusted p = 0.002, 288 paired items, four models). A homogeneous control shows that the gain comes mainly from computing through tools, which lowers error correlation, rather than from mixing different tools.

| Contrast (majority vote, pooled over 4 models) | Δ accuracy (points) | 95% CI |
|---|---|---|
| S substrates − A self-consistency | +8.3 | [+3.7, +12.8] |
| C fixed method templates − A | +1.5 | n.s. |
| G planner-generated plans − A | −3.6 | n.s. |

## Reproduce the paper without any API key

Every LLM call made for the paper is stored in `results/cache*.jsonl`. The pipeline reads them in `--cache-only` mode, so it makes no network call and needs no key.

```bash
pip install -r requirements.txt
./reproduce.sh
```

This reruns every statistical test, the figure and the MathArena comparison on the per-agent records used in the paper (`results/records_*.jsonl`), and writes the reports to `results/*.txt`. If `latexmk` is installed it also rebuilds `paper/stacking_paper.pdf` from `paper/template.tex`. It takes a few minutes. We checked that it reproduces every number in the paper exactly, except the sandbox time per problem, which is measured on your machine.

`./reproduce.sh --rebuild` first regenerates those records from the cached LLM outputs: it re-extracts and re-grades every answer and re-executes the agents' code (about 30 minutes). Code runs under a time limit, so on another machine a few runs can finish or time out differently and some secondary numbers can move slightly; in our test, the pooled S − A result was identical.

Requirements: Linux and Python 3.10 or newer. Agent code runs in a network-isolated sandbox (`unshare -n`), which needs user namespaces; on macOS or Windows, use Docker or WSL.

| Paper section | Script | Output |
|---|---|---|
| Main comparison, pooled result, per-model effects, ablation, exchange-rate diagnostic, compute-matched verifiers | `pooled_analysis.py` | `results/pooled_report.txt`, `results/pooled_summary.json` |
| Homogeneous controls (5× brute force, 5× Python) and independent replicate | `homog_analysis.py` | `results/homog_report.txt` |
| Token and call accounting, sandbox time, replicate stability | `extra_checks.py` | `results/extra_checks.txt` |
| Figure 1 | `make_figure.py` | `paper/figure1.pdf` |
| Comparison with the MathArena leaderboard | `leaderboard_compare.py` | `results/leaderboard_comparison.txt` |

The GSM8K pilot described in Section 5.1 of the paper is not part of this repository.

## Repository layout

```
run_v2.py            runs the agents (arms A, C, G, S, plus controls HB, HP, S2, X) and the verifier
arms_v2.py           prompts and substrate definitions for every arm
llm.py               minimal client for OpenAI-compatible endpoints, with a disk cache
sandbox.py           network-isolated execution of agent code
grading.py           answer extraction and equivalence (math-verify, SymPy)
aggregate.py         majority vote, weighted votes, diversity measures, bootstrap
verify10.py          10-candidate verifier for the compute-matched baseline (A×10)
verify_s4.py         verifier for the S4 ablation (S without SymPy)
pooled_analysis.py   pooled statistics (paired bootstrap, Holm, sign-flip permutation, random effects)
data/items90.json    the fixed set of 90 problems used in every analysis
data/matharena_2025_2026.json      problems and gold answers
data/matharena_leaderboard/*.csv   MathArena per-problem results, used for the leaderboard comparison
results/cache*.jsonl               every model response used in the paper
results/records_*.jsonl            per-agent records that the pipeline does not rebuild (gemma runs)
results/clusters_cache.json        cached answer-equivalence clusters
paper/template.tex, build_paper.py fill the paper from the analysis outputs
```

## Run new experiments

You need your own keys. Copy `.env.example` to `.env` and fill in the providers you use. `.env` is in `.gitignore`: never commit it.

| Model in the paper | Model id | Provider | Key |
|---|---|---|---|
| gemma-4-31B | `google/gemma-4-31b-it` | NVIDIA NIM | `NVIDIA_API_KEY` |
| gpt-oss-20b | `openai/gpt-oss-20b` (or `groq:openai/gpt-oss-20b`) | NVIDIA NIM (or Groq) | `NVIDIA_API_KEY` (or `GROQ_API_KEY`) |
| Gemini 3.5 Flash-Lite | `gemini-3.5-flash-lite` | Google AI Studio | `GOOGLE_API_KEY` |
| Gemini 3.1 Flash-Lite | `gemini-3.1-flash-lite` | Google AI Studio | `GOOGLE_API_KEY` |

Example: arms A and S on the 90 problems with a new cache file, so you do not mix your outputs with ours.

```bash
export STACK_CACHE=results/cache_mine.jsonl
python3 run_v2.py --items-file data/items90.json --models google/gemma-4-31b-it \
    --arms A,S --no-verify --workers 32 --order item --out results/records_mine.jsonl
```

Runs can be resumed: completed calls are read back from the cache, and only missing or failed calls are sent again. Free tiers are slow (Google allows 15 requests per minute and 500 per day per model; Groq limits tokens per minute), and `llm.py` paces requests accordingly. Some model ids above may have changed at the providers since the study.

## Licenses and data

- Code: MIT license (see `LICENSE`).
- The competition problems, answers and leaderboard results come from [MathArena](https://matharena.ai) ([Balunović et al., 2025](https://arxiv.org/abs/2505.23281)), released under CC BY-NC-SA 4.0. `data/` and the cached model outputs in `results/` are shared under the same license, for non-commercial research use with attribution.

## Citation

```bibtex
@misc{louriki2026tools,
  title  = {Computing Through Tools Decorrelates Same-Model {LLM} Ensembles},
  author = {Louriki, Reda},
  year   = {2026},
  note   = {arXiv preprint (identifier to be added)}
}
```

Contact: reda.louriki-etu@etu.univh2c.ma
