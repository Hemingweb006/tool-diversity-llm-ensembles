"""Answer extraction, equivalence and clustering for MATH.

Voting on raw strings would split "\\frac12", "1/2" and "0.5" into three votes and
quietly punish exactly the substrate-diverse agents (code prints 1/2, text writes
\\frac{1}{2}). So every analysis votes over *equivalence clusters* built with
math-verify, never over strings.

math-verify uses signal.alarm for timeouts, so call these from the MAIN thread only.
"""
import re
from functools import lru_cache

from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

_CFG = [LatexExtractionConfig(), ExprExtractionConfig()]


def last_boxed(text):
    """Content of the last \\boxed{...} (brace-balanced), or None."""
    if not text:
        return None
    i = text.rfind("\\boxed")
    if i < 0:
        i = text.rfind("\\fbox")
        if i < 0:
            return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k].strip()
    return None


def answer_line(stdout):
    """Last 'ANSWER: ...' line printed by an executed program."""
    if not stdout:
        return None
    m = re.findall(r"^\s*ANSWER:\s*(.+?)\s*$", stdout, re.M)
    return m[-1].strip() if m else None


def _sympy_latex(s):
    """Plain program output ('pi', 'sqrt(12)', '6-5*I', '(1, -2)') -> LaTeX.
    math-verify's plain-expression parser misses these; code agents print them.
    (sympify evaluates the string: acceptable here because this runs only in the
    disposable cloud container, on math answers, under a timeout.)"""
    if "\\" in s or "{" in s or "=" in s or len(s) > 200:
        return None
    try:
        import sympy
        e = sympy.sympify(s.replace("^", "**"), evaluate=False)
        if isinstance(e, (tuple, list, sympy.Tuple)):
            return ", ".join(sympy.latex(x) for x in e)
        return sympy.latex(e)
    except Exception:
        return None


@lru_cache(maxsize=200000)
def _parsed(ans):
    s = ans.strip()
    if not s:
        return None

    def go():
        cands = [s]
        lx = _sympy_latex(s)
        if lx:
            cands.insert(0, lx)
        for c in cands:
            wrapped = c if ("$" in c or "\\boxed" in c) else f"${c}$"
            p = parse(wrapped, extraction_config=_CFG) or parse(c, extraction_config=_CFG)
            if p:
                return p
        return None
    try:
        return go()
    except Exception:
        return None


def _norm_str(s):
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\^\{?\\circ\}?", "", s)
    s = s.replace("$", "").replace(" ", "").rstrip(".")
    s = s.replace("*I", "i").replace("*", "")        # python-style complex / products
    return s


@lru_cache(maxsize=400000)
def equivalent(a, b):
    """Symmetric equivalence of two answer strings."""
    if a is None or b is None:
        return False
    if _norm_str(a) == _norm_str(b):
        return True
    pa, pb = _parsed(a), _parsed(b)
    if pa is None or pb is None:
        return False

    try:
        return bool(verify(pa, pb) or verify(pb, pa))
    except Exception:
        return False


def correct(pred, gold):
    return pred is not None and equivalent(gold, pred)


def cluster(answers, gold=None):
    """answers: list[str|None] -> list of cluster ids (ints). None -> its own
    'no answer' cluster -1. If gold is given, the gold cluster gets id 0 when
    present, so callers can test correctness as id == 0."""
    reps, ids = [], []
    if gold is not None:
        reps.append(gold)
    for a in answers:
        if a is None:
            ids.append(-1)
            continue
        for k, r in enumerate(reps):
            if equivalent(a, r):
                ids.append(k)
                break
        else:
            reps.append(a)
            ids.append(len(reps) - 1)
    return ids


if __name__ == "__main__":
    tests = [("\\frac{3}{56}", "3/56", True), ("\\frac{1}{2}", "0.5", True),
             ("90^\\circ", "90", True), ("6 - 5i", "6-5i", True), ("\\pi", "pi", True),
             ("1,-2", "-2, 1", True), ("x=5", "5", True), ("2\\sqrt{3}", "sqrt(12)", True),
             ("p - q", "-q + p", True), ("144", "145", False), ("\\frac{3}{56}", "3/55", False)]
    ok = 0
    for g, p, want in tests:
        got = correct(p, g)
        ok += got == want
        print(f"  {'OK ' if got == want else 'BAD'} gold={g!r:<16} pred={p!r:<12} -> {got}")
    print(f"{ok}/{len(tests)} grading tests pass")
    print("boxed:", last_boxed("so \\boxed{\\frac{1}{\\sqrt{2}}} done"))
    print("cluster:", cluster(["1/2", "\\frac{1}{2}", "0.5", "3", None, "3.0"], gold="\\frac12"))
