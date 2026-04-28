"""Cached LLM client. SQLite-backed prompt-hash cache so re-runs are free."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import time
from contextlib import closing
from typing import Any, Iterable

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import EMBED_MODEL, PATHS


_CACHE_PATH = PATHS.cache / "llm.sqlite"


def _conn() -> sqlite3.Connection:
    PATHS.ensure()
    c = sqlite3.connect(_CACHE_PATH, timeout=30, isolation_level=None)
    c.execute(
        "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, model TEXT, response TEXT, ts INTEGER)"
    )
    c.execute(
        "CREATE TABLE IF NOT EXISTS embed_cache (key TEXT PRIMARY KEY, model TEXT, vector BLOB, ts INTEGER)"
    )
    return c


def _cache_key(model: str, payload: Any) -> str:
    blob = json.dumps({"model": model, "payload": payload}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> str | None:
    with closing(_conn()) as c:
        row = c.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
        return row[0] if row else None


def _cache_put(key: str, model: str, response: str) -> None:
    with closing(_conn()) as c:
        c.execute(
            "INSERT OR REPLACE INTO cache (key, model, response, ts) VALUES (?,?,?,?)",
            (key, model, response, int(time.time())),
        )


# ---- chat completion ---------------------------------------------------------

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=4, min=4, max=120))
def _call_anthropic(model: str, system: str, user: str, max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = []
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=4, min=4, max=120))
def _call_openai(model: str, system: str, user: str, max_tokens: int, reasoning_effort: str | None) -> str:
    from openai import OpenAI

    client = OpenAI()
    kwargs: dict = dict(
        model=model,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    # GPT-5 family burns reasoning tokens before output. For pattern/extraction
    # tasks, 'low' or 'minimal' keeps the budget available for the answer.
    if reasoning_effort and model.startswith("gpt-5"):
        kwargs["reasoning_effort"] = reasoning_effort
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def _is_anthropic(model: str) -> bool:
    return model.startswith("claude")


def chat(
    model: str,
    user: str,
    system: str = "You are a careful, terse assistant.",
    max_tokens: int = 4096,
    reasoning_effort: str | None = "low",
    use_cache: bool = True,
) -> str:
    """Single-turn chat with prompt-hash caching.

    `reasoning_effort` only affects OpenAI gpt-5* models. Default 'low' is
    appropriate for most extraction/labeling tasks. Pass 'medium' or 'high'
    for the actual review and meta-review where reasoning helps.
    """
    payload = {"system": system, "user": user, "max_tokens": max_tokens, "re": reasoning_effort}
    key = _cache_key(model, payload)
    if use_cache:
        hit = _cache_get(key)
        if hit is not None:
            return hit
    if _is_anthropic(model):
        out = _call_anthropic(model, system, user, max_tokens)
    else:
        out = _call_openai(model, system, user, max_tokens, reasoning_effort)
    _cache_put(key, model, out)
    return out


async def chat_async(model: str, user: str, **kwargs: Any) -> str:
    """asyncio-friendly wrapper. Real async would use the SDKs' async clients;
    a thread executor is fine here because we're rate-limited by the API anyway."""
    return await asyncio.to_thread(chat, model, user, **kwargs)


def chat_json(model: str, user: str, **kwargs: Any) -> dict:
    """chat() that expects a JSON object back. Strips code fences if present."""
    raw = chat(model, user, **kwargs)
    return _parse_json(raw)


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if not text:
        # Common GPT-5 failure mode: reasoning ate the entire budget.
        # Don't lose the rest of the panel over one empty response.
        return {"_error": "empty_response"}
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: -3]
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"_error": "parse_error", "_raw": raw[:500], "_msg": str(e)}


# ---- embeddings --------------------------------------------------------------

def _embed_cache_get(keys: list[str]) -> dict[str, list[float]]:
    if not keys:
        return {}
    # SQLite caps the parameter count per statement (default 999, modern builds 32766).
    # Chunk to stay safely under any limit.
    out: dict[str, list[float]] = {}
    chunk_size = 500
    with closing(_conn()) as c:
        for s in range(0, len(keys), chunk_size):
            chunk = keys[s : s + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            rows = c.execute(
                f"SELECT key, vector FROM embed_cache WHERE key IN ({placeholders})", chunk
            ).fetchall()
            for k, v in rows:
                out[k] = json.loads(v)
    return out


def _embed_cache_put(items: dict[str, list[float]], model: str) -> None:
    if not items:
        return
    now = int(time.time())
    with closing(_conn()) as c:
        c.executemany(
            "INSERT OR REPLACE INTO embed_cache (key, model, vector, ts) VALUES (?,?,?,?)",
            [(k, model, json.dumps(v), now) for k, v in items.items()],
        )


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60))
def _embed_openai(model: str, batch: list[str]) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI()
    resp = client.embeddings.create(model=model, input=batch)
    return [d.embedding for d in resp.data]


def embed(texts: Iterable[str], model: str | None = None, batch: int = 100) -> list[list[float]]:
    """Embed texts with caching. Defaults to OpenAI text-embedding-3-large.

    Pass `model='local:<sentence-transformers-name>'` to use a local model
    (requires the `local-embed` extra).
    """
    model = model or EMBED_MODEL
    texts = list(texts)
    keys = [_cache_key(model, t) for t in texts]
    cached = _embed_cache_get(keys)

    out: list[list[float] | None] = [cached.get(k) for k in keys]
    missing_idx = [i for i, v in enumerate(out) if v is None]
    if missing_idx:
        if model.startswith("local:"):
            from sentence_transformers import SentenceTransformer  # type: ignore

            st = SentenceTransformer(model.removeprefix("local:"))
            vecs = st.encode([texts[i] for i in missing_idx], show_progress_bar=False).tolist()
        else:
            vecs = []
            for s in range(0, len(missing_idx), batch):
                chunk = [texts[i] for i in missing_idx[s : s + batch]]
                vecs.extend(_embed_openai(model, chunk))
        new_items = {keys[i]: v for i, v in zip(missing_idx, vecs)}
        _embed_cache_put(new_items, model)
        for i, v in zip(missing_idx, vecs):
            out[i] = v
    return out  # type: ignore[return-value]


def have_anthropic() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def have_openai() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))
