"""v2 arms: every arm = 5 agents per item on the SAME base model, T=0.7.

  A  self-consistency       : identical neutral prompt x5          (floor)
  C  fixed method templates : 5 hand-written procedural duals      (pilot's idea)
  G  generated duals        : a planner writes 5 problem-specific plans, 5 solvers
  S  substrate duals        : 5 different computational substrates (text / executed
                              Python / SymPy / brute force / solve-then-code-check)

Then an aggregation stage per arm (verifier) and post-hoc aggregators.
"""

# Ladder chosen after probing 2025-26 competition problems: gemma (non-reasoning,
# ~67%) and gpt-oss (reasoning, ~73%). nemotron-120b was dropped: 5/12 of its
# answers truncated at 16k tokens, i.e. its errors measured budget, not reasoning.
MODELS = [
    "google/gemma-4-31b-it",
    "openai/gpt-oss-20b",
]
SHORT = {"google/gemma-4-31b-it": "gemma31b",
         "nvidia/nemotron-3.5-lightning-30b-a3b": "nemolight30b",
         "gemini-3.5-flash-lite": "gemflashlite",
         "gemini-3.1-flash-lite": "gem31flashlite",
         "groq:qwen/qwen3.8-27b": "qwen38groq",
         "groq:openai/gpt-oss-20b": "gptoss20bgroq",
         "meta/muse-glimmer-30b": "museglimmer30b",
         "nvidia/nemotron-3-super-120b-a12b": "nemo120b",
         "openai/gpt-oss-20b": "gptoss20b"}
T = 0.7
MAX_TOKENS = 16384

BOXED = "Put your final answer in \\boxed{} at the end, in simplest exact form."
CODE_OUT = ("The program must print exactly one final line of the form\n"
            "ANSWER: <answer>\n"
            "where <answer> is the exact final answer written in LaTeX (use sympy.latex() "
            "for SymPy objects). Put the whole program in ONE ```python code block. "
            "The program runs with Python 3 and SymPy available, no internet, 30 s limit.")

# ------------------------------------------------------------------ A
NEUTRAL = "Solve the following problem. Reason step by step. " + BOXED
ARM_A = [dict(tag="neutral", kind="text", prompt=NEUTRAL) for _ in range(5)]

# ------------------------------------------------------------------ C
_C = [
    ("direct-analytic",
     "Solve by a DIRECT ANALYTIC DERIVATION: set up the key relation and push it through "
     "algebraic manipulation to the answer, with no casework and no guessing."),
    ("casework",
     "Solve by EXHAUSTIVE CASEWORK: split the problem into mutually exclusive, "
     "collectively exhaustive cases, solve each case separately, then combine."),
    ("backward-verify",
     "Solve BACKWARDS: start from what is asked and work back to the givens. Then "
     "verify your candidate by substituting it into every condition of the problem."),
    ("small-cases-pattern",
     "Solve via SMALL CASES: first work out small or special instances by hand, identify "
     "the pattern or invariant, then generalise it and justify the general step."),
    ("bounds-sanity",
     "Solve via BOUNDS THEN EXACT: first estimate the magnitude or bounds of the answer, "
     "then compute it exactly, then check the exact value against the bounds and a "
     "special case."),
]
ARM_C = [dict(tag=n, kind="text", prompt=f"Solve the following problem.\n\nMETHOD ({n}): {i}\n\n{BOXED}")
         for n, i in _C]

# ------------------------------------------------------------------ G
PLANNER = """You will NOT solve the problem below. Your job is to design 5 genuinely
different ways to solve it, for 5 independent solvers.

The 5 approaches must differ in their CORE IDEA or FORMALISATION, not in wording. For
example: different choice of unknowns, a different theorem or identity, algebraic vs
geometric vs combinatorial view, direct vs complementary counting, coordinates vs
synthetic, generating functions vs recursion, a symmetry/invariant argument, bounding.
Each approach must be a complete, viable route to the exact answer on its own.

Do NOT compute or state the final answer, or any number that reveals it.

Reply with JSON only:
{"plans": [{"name": "<short name>", "plan": "<3-6 sentences describing the route>"}, ...5 items]}

PROBLEM:
"""


def g_solver_prompt(name, plan):
    return (f"Solve the following problem using EXACTLY this approach.\n\n"
            f"APPROACH ({name}): {plan}\n\nFollow the approach faithfully; if it hits a "
            f"dead end, repair it within the same approach rather than switching. {BOXED}")


# ------------------------------------------------------------------ S
ARM_S = [
    dict(tag="text-cot", kind="text", prompt=NEUTRAL),
    dict(tag="python", kind="code",
         prompt=("Solve the following problem by writing a Python program that COMPUTES the "
                 "answer exactly (use integers, fractions.Fraction, or SymPy for exactness). "
                 "Keep prose to a minimum; the program does the work. " + CODE_OUT)),
    dict(tag="sympy-symbolic", kind="code",
         prompt=("Solve the following problem by FORMALISING it in SymPy: declare symbols, "
                 "write every condition as an equation/expression, and let SymPy solve or "
                 "simplify exactly. Do not solve it by hand first. " + CODE_OUT)),
    dict(tag="brute-force", kind="code",
         prompt=("Solve the following problem by BRUTE FORCE: write a Python program that "
                 "exhaustively enumerates or directly searches/simulates the concrete objects "
                 "involved. If the problem is continuous, compute numerically at high precision "
                 "(mpmath/SymPy) and recover the exact form (e.g. sympy.nsimplify). Avoid "
                 "clever derivations. " + CODE_OUT)),
    dict(tag="solve-then-check", kind="check",
         prompt=("Solve the following problem analytically, reasoning step by step, and state "
                 "a candidate answer in \\boxed{}. THEN write a Python program that "
                 "independently CHECKS the candidate (substitute it back, recompute it a "
                 "different way, or test the defining conditions) and prints the final answer, "
                 "corrected if the check fails. " + CODE_OUT)),
]

REPAIR = ("Your program failed.\n\n{err}\n\nFix it and reply with the complete corrected "
          "program in ONE ```python block. It must print the final line 'ANSWER: <answer>'.")

# ------------------------------------------------------------------ verifier
VERIFIER = """Below is a math problem and {k} candidate solutions written by independent
solvers. Some may be wrong. Check each candidate's reasoning and final answer
critically: look for invalid steps, arithmetic slips, misread conditions, and code that
computes the wrong thing.

Then decide which candidate final answer is correct. Your final answer MUST be one of the
candidates' final answers (choose the best-supported one). End with the chosen answer
in \\boxed{{}}.

PROBLEM:
{problem}

{candidates}"""

# X: 5 more neutral samples. A+X = A10, the compute-matched self-consistency baseline
# (with a verifier over all 10 candidates, see verify10.py).
ARM_X = [dict(tag="neutral-extra", kind="text", prompt=NEUTRAL) for _ in range(5)]

# Homogeneous-substrate controls (reviewer #1): the SAME substrate replicated 5x.
# Slot 0 reuses the corresponding S-arm sample (an i.i.d. draw of the same prompt).
ARM_HB = [dict(ARM_S[3], tag="brute-force") for _ in range(5)]    # best single substrate on gemma
ARM_HP = [dict(ARM_S[1], tag="python") for _ in range(5)]

ARM_S2 = [dict(c) for c in ARM_S]          # independent replicate of S (run-to-run stability, reviewer #6)

ARMS = {"A": ARM_A, "C": ARM_C, "S": ARM_S, "X": ARM_X, "HB": ARM_HB, "HP": ARM_HP, "S2": ARM_S2}   # G is built per item
ARM_ORDER = ["A", "X", "C", "G", "S", "S2", "HB", "HP"]
LABELS = {"S2": "S2 substrates (replicate)", "HB": "HB brute-force x5", "HP": "HP python x5", "A": "A self-consistency", "X": "X +5 neutral (A10)", "C": "C fixed templates",
          "G": "G generated duals", "S": "S substrate duals"}
