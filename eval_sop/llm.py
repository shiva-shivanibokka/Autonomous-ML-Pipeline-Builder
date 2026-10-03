"""
A cached chat model for the benchmark, for any OpenAI-compatible endpoint.

Stands in for core.providers.get_llm / get_codegen_llm. It talks to an
OpenAI-compatible endpoint given by base_url + model: a local Ollama server
(default http://localhost:11434/v1, no key) or a free-tier API such as Groq
(https://api.groq.com/openai/v1). A key, if needed, is passed in as a value;
it is never placed in os.environ, so generated code cannot see it.

Every response is appended to a JSONL cache keyed on
(model, temperature, max_tokens, messages); a later call with an identical
prompt is answered from the cache.

Why a cache: the system's LLM steps (plan, preprocessing advice, feature
engineering code) see the whole CSV and never the split seed, so their prompts
are identical across seeds. Caching makes seeds 1..k reuse the seed-0 LLM
outputs, which (a) keeps the run inside a local-model budget and (b) makes the
LLM arm exactly re-runnable from the committed cache. The cost: the reported
seed variance is split/model-RNG variance only, not LLM sampling variance.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from openai import OpenAI, RateLimitError

DEFAULT_BASE_URL = "http://localhost:11434/v1"
_TMP_RE = re.compile(r"amlpb_exec_\w+")


class _Resp:
    def __init__(self, content: str):
        self.content = content


class CachedChat:
    """Minimal .invoke(messages) -> obj.content, like a LangChain chat model."""

    _lock = threading.Lock()

    def __init__(self, model: str, temperature: float, max_tokens: int, cache_path: Path,
                 role: str, stats: dict[str, Any], base_url: str = DEFAULT_BASE_URL,
                 api_key: str = "", num_ctx: int | None = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_path = Path(cache_path)
        self.role = role
        self.stats = stats
        self.base_url = base_url
        # num_ctx: talk to Ollama's native /api/chat so the context window can be
        # capped (its OpenAI-compatible endpoint has no context option).
        self.num_ctx = num_ctx
        self._client = OpenAI(api_key=api_key or "not-needed", base_url=base_url,
                              timeout=1800, max_retries=0)
        self._cache = self._load()

    def _load(self) -> dict[str, str]:
        cache: dict[str, str] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    cache[rec["key"]] = rec["response"]
        return cache

    def _key(self, msgs: list[dict[str, str]]) -> str:
        # Self-correction prompts quote tracebacks that contain the executor's
        # random throwaway directory (amlpb_exec_XXXX); mask it so an otherwise
        # identical prompt on another seed hits the cache.
        norm = [{**m, "content": _TMP_RE.sub("amlpb_exec_X", m["content"])} for m in msgs]
        blob = json.dumps(
            {"model": self.model, "t": self.temperature, "max": self.max_tokens, "m": norm},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def invoke(self, messages):
        msgs = []
        for m in messages:
            role = {"system": "system", "human": "user", "ai": "assistant"}.get(
                getattr(m, "type", "human"), "user"
            )
            msgs.append({"role": role, "content": m.content})
        key = self._key(msgs)
        with self._lock:
            self._cache.update(self._load())  # pick up entries from other processes
            if key in self._cache:
                self.stats["cache_hits"] = self.stats.get("cache_hits", 0) + 1
                return _Resp(self._cache[key])
        t0 = time.time()
        r = None
        if self.num_ctx:
            text, usage = self._ollama_native(msgs)
            return self._record(key, msgs, text, usage, t0)
        for attempt in range(8):  # free tiers rate-limit (HTTP 429): back off and retry
            try:
                r = self._client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=0,
                )
                break
            except RateLimitError:
                self.stats["rate_limited"] = self.stats.get("rate_limited", 0) + 1
                time.sleep(min(120, 10 * 2 ** attempt))
        if r is None:
            raise RuntimeError("LLM endpoint kept rate-limiting; giving up on this call")
        text = r.choices[0].message.content or ""
        return self._record(key, msgs, text, r.usage.model_dump() if r.usage else None, t0)

    def _ollama_native(self, msgs):
        import urllib.request

        root = self.base_url.rstrip("/")
        root = root[: -len("/v1")] if root.endswith("/v1") else root
        # Generation is capped at half the context so prompt + output fit in num_ctx.
        body = {"model": self.model, "messages": msgs, "stream": False,
                "options": {"temperature": self.temperature, "seed": 0, "num_ctx": self.num_ctx,
                            "num_predict": min(self.max_tokens, self.num_ctx // 2)}}
        req = urllib.request.Request(root + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        usage = {"prompt_tokens": out.get("prompt_eval_count"), "completion_tokens": out.get("eval_count"),
                 "num_ctx": self.num_ctx, "num_predict": body["options"]["num_predict"],
                 "done_reason": out.get("done_reason")}
        return out["message"]["content"] or "", usage

    def _record(self, key, msgs, text, usage, t0):
        dt = time.time() - t0
        self.stats["llm_calls"] = self.stats.get("llm_calls", 0) + 1
        self.stats["llm_seconds"] = self.stats.get("llm_seconds", 0.0) + dt
        rec = {
            "key": key,
            "role": self.role,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": msgs,
            "response": text,
            "latency_s": round(dt, 2),
            "usage": usage,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with self._lock:
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            self._cache[key] = text
        return _Resp(text)
