"""LLM provider API reader (llama-3.1-8b-instruct, OpenAI-compatible).

The end-to-end reader uses Llama-3.1-8B-Instruct with a 4096-token context
and greedy decoding (temperature=0).

Endpoint: ${LLM_BASE_URL}/chat/completions
Auth:     Bearer <API_KEY>
Note:     Cloudflare blocks default urllib UA; pass a browser UA header.

Cached responses are keyed by sha1(prompt) for deterministic reruns.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = os.environ.get("LLM_BASE_URL", "https://api.apertis.ai/v1").rstrip("/") + "/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instruct"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/127.0.0.0 Safari/537.36"
)


def _load_key() -> str:
    key = os.environ.get("LLM_API_KEY")
    if key:
        return key.strip()
    raise RuntimeError("LLM_API_KEY not set")


class LLMReader:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: Path = Path(".cache/llm_response_cache"),
        max_tokens: int = 256,
        temperature: float = 0.0,
        max_retries: int = 5,
    ) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self._key = _load_key()

    def _cache_path(self, prompt: str) -> Path:
        h = hashlib.sha1(
            f"{self.model}|{self.temperature}|{self.max_tokens}|{prompt}".encode()
        ).hexdigest()
        return self.cache_dir / f"{h}.json"

    def chat(self, system: str, user: str) -> dict:
        prompt = f"SYS::{system}\nUSER::{user}"
        cp = self._cache_path(prompt)
        if cp.exists():
            return json.loads(cp.read_text())
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        data = json.dumps(body).encode()
        last_err: str | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                API_URL,
                data=data,
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    payload = json.loads(r.read().decode())
                content = payload["choices"][0]["message"]["content"]
                out = {
                    "content": content,
                    "usage": payload.get("usage", {}),
                    "model": payload.get("model", self.model),
                }
                cp.write_text(json.dumps(out))
                return out
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}: {e.read().decode()[:300]}"
                if e.code in (401, 402, 403):
                    raise RuntimeError(f"LLM provider auth/credit failure: {last_err}")
                time.sleep(2 ** attempt)
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM provider call failed after {self.max_retries} retries: {last_err}")


SYSTEM_QA = (
    "You are a careful question-answering assistant. Answer the user's question "
    "using ONLY the provided context paragraphs. Give the shortest possible "
    "answer (a name, number, date, entity, or short phrase). If the context "
    "does not contain the answer, reply 'insufficient evidence'."
)


def build_user_prompt(question: str, paragraphs: list[str], char_budget: int = 12000) -> str:
    """Assemble retrieval-augmented prompt under a char budget (~3000-4000 tokens)."""
    ctx_lines: list[str] = []
    used = 0
    for i, p in enumerate(paragraphs, 1):
        line = f"[{i}] {p.strip()}\n"
        if used + len(line) > char_budget:
            break
        ctx_lines.append(line)
        used += len(line)
    return f"Context:\n{''.join(ctx_lines)}\nQuestion: {question}\nAnswer:"
