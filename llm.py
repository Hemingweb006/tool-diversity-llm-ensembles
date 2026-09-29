"""Dependency-free multi-provider chat client (OpenAI-compatible endpoints):
NVIDIA NIM, Google AI Studio, Groq (model prefix "groq:") and DeepSeek.

API keys are read from environment variables or from a local .env file (never committed):
NVIDIA_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, DEEPSEEK_API_KEY.

- disk cache keyed by an explicit experiment key (resumable; errors are NOT cached,
  so a rerun retries only what failed)
- per-model concurrency caps (NIM rate-limits per account/model)
- 429/5xx backoff honouring Retry-After
- reasoning models: falls back to reasoning_content when content is empty
"""
import json, os, random, re, threading, time, urllib.error, urllib.request

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
CACHE_PATH = os.environ.get("STACK_CACHE", "results/cache.jsonl")   # this process appends here
RATE_PER_MIN = {"gemini-3.5-flash-lite": 14, "gemini-3.1-flash-lite": 14,
                "groq:qwen/qwen3.8-27b": 1.2, "groq:openai/gpt-oss-20b": 1.8}        # Groq free: 8k tokens/min, ~6k tokens/call      # Google free tier: 15 req/min per model
_rate_lock, _rate_next = threading.Lock(), {}

MODEL_CONCURRENCY = {
    "google/gemma-4-31b-it": 60,
    "nvidia/nemotron-3-super-120b-a12b": 30,
    "openai/gpt-oss-20b": 45,
    "nvidia/nemotron-3.5-lightning-30b-a3b": 30,
    "nvidia/nemotron-3-ultra-550b-a55b": 30,
    "meta/llama-3.2-11b-vision-instruct": 30,
    "z-ai/glm-5.3": 20,
    "gemini-3.5-flash-lite": 6,
    "gemini-3.1-flash-lite": 6,
    "groq:qwen/qwen3.8-27b": 3,
    "groq:openai/gpt-oss-20b": 3,
    "meta/muse-glimmer-30b": 15,
}
_sems, _sem_lock = {}, threading.Lock()
CACHE_ONLY = False


class NotCached(Exception):
    """Raised in cache-only mode for a call that was never made."""
_lock = threading.Lock()
_cache = {}
STATS = {"calls": 0, "retries": 0, "r429": 0, "errors": 0}


def _sem(model):
    with _sem_lock:
        if model not in _sems:
            _sems[model] = threading.Semaphore(MODEL_CONCURRENCY.get(model, 4))
        return _sems[model]


def load_keys():
    keys = {}
    if os.path.exists(".env"):
        for line in open(".env"):
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                keys[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("NVIDIA_API_KEY", "DEEPSEEK_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"):
        if os.environ.get(k):
            keys[k] = os.environ[k]
    return keys


def _route(model, keys):
    if model.startswith("groq:"):
        return GROQ_URL, keys.get("GROQ_API_KEY")
    if model.startswith("gemini") or model.startswith("gemma-"):
        return GOOGLE_URL, keys.get("GOOGLE_API_KEY")
    if "/" in model:
        return NVIDIA_URL, keys.get("NVIDIA_API_KEY")
    return DEEPSEEK_URL, keys.get("DEEPSEEK_API_KEY")


def _pace(model):
    """Space request starts so a per-model requests/min quota is never exceeded."""
    rpm = RATE_PER_MIN.get(model)
    if not rpm:
        return
    with _rate_lock:
        now = time.time()
        start = max(now, _rate_next.get(model, 0.0))
        _rate_next[model] = start + 60.0 / rpm
    if start > now:
        time.sleep(start - now)


def load_cache():
    """Load every results/cache*.jsonl (several processes each append to their own
    file, so no two processes ever write the same file)."""
    import glob
    _cache.clear()
    paths = sorted(set(glob.glob("results/cache*.jsonl")) | ({CACHE_PATH} if os.path.exists(CACHE_PATH) else set()))
    for path in paths:
        for line in open(path):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error") or (not rec.get("mock") and _degenerate(rec.get("text", ""), rec.get("content_empty", False))):
                continue                       # never trust a cached failure
            _cache[rec["key"]] = rec
    return len(_cache)


def _degenerate(text, content_empty=False):
    """Server-side garbage (e.g. kimi-k3 returning 'The!!!!!!...' with empty content,
    ~33 tokens, finish=stop). An infrastructure fault, not model reasoning -> retry,
    never score it."""
    text = text or ""
    if not re.search(r"[A-Za-z0-9]", text):
        return True
    return content_empty and bool(re.search(r"([^\w\s])\1{15,}", text))


def _append(rec):
    with _lock:
        _cache[rec["key"]] = rec
        with open(CACHE_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")


class Mock:
    """Synthetic LLM with planted error structure, for validating the pipeline
    without spending credits. Lower SHARED fraction => less common-mode failure =>
    analysis MUST show lower Q/phi and higher ambiguity for that arm."""
    SHARED = {"A": 0.95, "B": 0.88, "C": 0.45, "D": 0.65, "E": 0.35, "F": 0.95}
    SKILL = {"A": 0.78, "B": 0.78, "C": 0.72, "D": 0.78, "E": 0.72, "F": 0.78}

    def __init__(self, seed=0):
        self.seed = seed

    def __call__(self, key, model, messages, temperature, max_tokens):
        arm, item = key.split("|")[0], key.split("|")[1]
        shared, skill = self.SHARED.get(arm, 0.8), self.SKILL.get(arm, 0.75)
        rnd = random.Random(abs(hash((self.seed, key))) % 10 ** 8)
        common = random.Random(abs(hash((self.seed, arm, item))) % 10 ** 8).random()
        p_fail = 1 - skill
        fails = (common < p_fail * shared) or (rnd.random() < p_fail * (1 - shared))
        truth = int(item.split("#")[-1])
        ans = truth if not fails else truth + rnd.choice([1, 2, 10, -3, 7])
        return dict(text=f"reasoning...\n#### {ans}", usage={"total_tokens": 100},
                    model=model, mock=True)


def call(key, model, messages, temperature=0.7, max_tokens=2048,
         keys=None, mock=None, max_retries=10):
    if key in _cache:
        return _cache[key]
    if CACHE_ONLY:
        raise NotCached(key)
    if mock is not None:
        rec = mock(key, model, messages, temperature, max_tokens)
        rec["key"] = key
        _append(rec)
        return rec

    url, api_key = _route(model, keys or {})
    if not api_key:
        raise SystemExit(f"No API key for {model}")
    body = dict(model=model.split(":", 1)[1] if model.startswith("groq:") else model,
                messages=messages, max_tokens=max_tokens,
                temperature=temperature, stream=False)
    data = json.dumps(body).encode()
    hdrs = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 stacking-bench"}

    last = None
    with _sem(model):
        for attempt in range(max_retries):
            STATS["calls"] += 1
            _pace(model)
            try:
                req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
                with urllib.request.urlopen(req, timeout=1200) as r:
                    payload = json.loads(r.read().decode())
                ch = payload["choices"][0]
                msg = ch["message"]
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                rec = dict(key=key, model=model, temperature=temperature,
                           text=content if content.strip() else reasoning,
                           content_empty=not content.strip(),
                           usage=payload.get("usage", {}),
                           finish=ch.get("finish_reason"))
                if _degenerate(rec["text"], rec["content_empty"]):
                    last = "degenerate output"
                    STATS["degenerate"] = STATS.get("degenerate", 0) + 1
                    STATS["retries"] += 1
                    time.sleep(min(60, 2 ** attempt) + random.random())
                    continue
                _append(rec)
                return rec
            except urllib.error.HTTPError as e:
                code = e.code
                last = f"HTTP {code}: {e.read().decode()[:160]}"
                if code in (401, 403):
                    raise SystemExit(f"Auth failure for {model} -> {last}")
                if code == 404:
                    break
                if code == 429:
                    STATS["r429"] += 1
                    ra = e.headers.get("Retry-After") if e.headers else None
                    wait = float(ra) if ra and ra.replace(".", "").isdigit() else None
                    STATS["retries"] += 1
                    time.sleep(wait or min(90, 3 * (2 ** attempt)) + random.random() * 2)
                    continue
                if code not in (500, 502, 503, 504):
                    break
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
            STATS["retries"] += 1
            time.sleep(min(90, 2 ** attempt) + random.random())
    STATS["errors"] += 1
    return dict(key=key, model=model, text="", error=last or "unknown", usage={})


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract(text, strict=False):
    """GSM8K answer extraction: prefer the '#### x' line, else last number.
    strict=True returns None unless an explicit '####' answer line exists."""
    if not text:
        return None
    m = re.findall(r"####\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    cand = m[-1] if m else None
    if cand is None and strict:
        return None
    if cand is None:
        nums = _NUM.findall(text.replace("$", ""))
        cand = nums[-1] if nums else None
    if cand is None:
        return None
    cand = cand.replace(",", "").rstrip(".")
    try:
        v = float(cand)
    except ValueError:
        return None
    return int(v) if abs(v - round(v)) < 1e-9 else round(v, 4)
