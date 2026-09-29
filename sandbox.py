"""Execute model-written Python in an isolated subprocess.

Runs inside this disposable cloud container (never on the user's machine), with a
CPU/memory cap, a wall-clock timeout, no inherited proxy/credential environment,
and a throwaway working directory.
"""
import os, re, resource, subprocess, sys, tempfile

TIMEOUT_S = 30
MEM_BYTES = 2 * 1024 ** 3


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S, TIMEOUT_S + 2))
    resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    os.setsid()


def extract_code(text):
    """Last ```python fenced block (or last ``` block)."""
    if not text:
        return None
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else None


def run(code):
    """-> dict(ok, stdout, stderr, timeout)"""
    if not code:
        return dict(ok=False, stdout="", stderr="no code block found", timeout=False)
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "PYTHONHASHSEED": "0",
           "HOME": "/tmp", "LANG": "C.UTF-8"}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(code)
        try:
            # unshare -n: no network, so agents compute answers rather than look them up
            p = subprocess.run(["unshare", "-n", sys.executable, path], cwd=d, env=env,
                               capture_output=True,
                               text=True, timeout=TIMEOUT_S + 5, preexec_fn=_limits)
            return dict(ok=p.returncode == 0, stdout=p.stdout[-4000:],
                        stderr=p.stderr[-2000:], timeout=False)
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode()[-4000:] if isinstance(e.stdout, bytes) else (e.stdout or "")
            return dict(ok=False, stdout=out, stderr="TIMEOUT", timeout=True)


if __name__ == "__main__":
    print(run("import sympy as sp\nx=sp.symbols('x')\nr=sp.solve(x**2-2,x)\nprint('ANSWER:', sp.latex(max(r)))"))
    print(run("while True: pass")["stderr"])
    print(run("import urllib.request\nurllib.request.urlopen('https://example.com',timeout=5)")["ok"])
