#!/bin/bash
# Rebuild every number, table and figure of the paper from the cached model outputs.
# No API key and no network needed: every LLM call is read from results/cache*.jsonl.
set -e
cd "$(dirname "$0")"
export STACK_CACHE=results/cache_readonly.jsonl   # nothing is appended in --cache-only mode
I="--items-file data/items90.json --cache-only --workers 16 --order item"

# By default the analyses run on results/records_*.jsonl, the exact records used in the paper.
# With --rebuild, the records are first rebuilt from the cached LLM outputs. This re-executes the
# agents' code under a time limit, so on a different machine a few runs can finish or time out
# differently, and a handful of numbers may move slightly. The headline result does not change.
if [ "$1" = "--rebuild" ]; then
echo "[1/6] rebuilding per-model records from the cache"
python3 run_v2.py $I --models gemini-3.5-flash-lite --arms A,C,S --no-verify --out results/records_final_gemflashlite.jsonl | tail -1
python3 run_v2.py $I --models gemini-3.1-flash-lite --arms A,S --no-verify --out results/records_final_gem31flashlite.jsonl | tail -1
python3 run_v2.py $I --models openai/gpt-oss-20b --arms A,C,G,S --out results/records_final_gptoss20b.jsonl | tail -1
python3 run_v2.py $I --models groq:openai/gpt-oss-20b --arms A,S --no-verify --out results/records_final_gptoss20bgroq.jsonl | tail -1
python3 run_v2.py $I --models google/gemma-4-31b-it --arms HB,HP --no-verify --out results/records_final_homog.jsonl | tail -1
python3 run_v2.py $I --models google/gemma-4-31b-it --arms S2 --no-verify --out results/records_final_s2.jsonl | tail -1
else
echo "[1/6] using the records shipped in results/ (run with --rebuild to regenerate them)"
fi

echo "[2/6] pooled analysis (main result, per-model effects, ablations, compute-matched)"
python3 pooled_analysis.py 2>/dev/null > results/pooled_report.txt
echo "[3/6] homogeneous controls and replicate"
python3 homog_analysis.py 2>/dev/null > results/homog_report.txt
echo "[4/6] compute accounting and replicate stability"
python3 extra_checks.py 2>/dev/null > results/extra_checks.txt
echo "[5/6] figure 1 and MathArena comparison"
python3 make_figure.py 2>/dev/null > results/figure_log.txt
python3 leaderboard_compare.py > results/leaderboard_comparison.txt
echo "[6/6] paper"
if command -v latexmk >/dev/null; then (cd paper && python3 build_paper.py); else
  echo "latexmk not found: skipping the PDF build (paper/stacking_paper.pdf is the published version)"; fi
echo "done: reports are in results/*.txt"
