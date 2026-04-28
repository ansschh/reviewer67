"""Anthropic + OpenAI batch-API wrappers.

Both providers offer 50% off for offline jobs with 24-hour SLA. Calibration
is offline by definition — there's no reason to pay sync rates for it.

Status:
- Anthropic batch:  implemented (used for Claude Haiku/Sonnet/Opus calibration)
- OpenAI batch:     implemented (used for gpt-5* calibration)

Both return responses in the same `list[str]` shape as a synchronous call,
ordered to match the input prompts. Cached on disk by batch_id so a crash
mid-poll doesn't lose results.
"""
from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PATHS


_BATCH_DIR = PATHS.cache  # reuse the LLM cache dir


@dataclass
class BatchRequest:
    custom_id: str
    user: str
    system: str = "You are a careful, terse assistant."
    max_tokens: int = 4096
    reasoning_effort: str | None = "low"


# ---- Anthropic --------------------------------------------------------------

def _anthropic_payload(req: BatchRequest, model: str) -> dict:
    return {
        "custom_id": req.custom_id,
        "params": {
            "model": model,
            "max_tokens": req.max_tokens,
            "system": req.system,
            "messages": [{"role": "user", "content": req.user}],
        },
    }


def submit_anthropic(model: str, requests: list[BatchRequest]) -> str:
    """Submit a Claude message batch. Returns the batch ID."""
    import anthropic

    client = anthropic.Anthropic()
    payload = [_anthropic_payload(r, model) for r in requests]
    batch = client.messages.batches.create(requests=payload)
    print(f"[batch] anthropic submitted: {batch.id}  ({len(requests)} requests)")
    return batch.id


def wait_anthropic(batch_id: str, poll_interval: int = 30, timeout: int = 86400) -> dict[str, str]:
    """Block until the batch is done. Returns {custom_id: text}."""
    import anthropic

    client = anthropic.Anthropic()
    deadline = time.time() + timeout
    while time.time() < deadline:
        b = client.messages.batches.retrieve(batch_id)
        s = b.processing_status
        c = b.request_counts
        print(f"[batch] anthropic {batch_id} status={s} done={c.succeeded}/{c.processing+c.succeeded+c.errored}")
        if s == "ended":
            break
        time.sleep(poll_interval)
    else:
        raise TimeoutError(f"Anthropic batch {batch_id} did not complete in {timeout}s")

    out: dict[str, str] = {}
    for line in client.messages.batches.results(batch_id):
        cid = line.custom_id
        if line.result.type == "succeeded":
            content = line.result.message.content
            text = "".join(b.text for b in content if getattr(b, "type", None) == "text")
            out[cid] = text
        else:
            out[cid] = ""
    return out


# ---- OpenAI -----------------------------------------------------------------

def _openai_jsonl_line(req: BatchRequest, model: str) -> str:
    body: dict = {
        "model": model,
        "max_completion_tokens": req.max_tokens,
        "messages": [
            {"role": "system", "content": req.system},
            {"role": "user", "content": req.user},
        ],
    }
    if req.reasoning_effort and model.startswith("gpt-5"):
        body["reasoning_effort"] = req.reasoning_effort
    return json.dumps({
        "custom_id": req.custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    })


def submit_openai(model: str, requests: list[BatchRequest]) -> str:
    from openai import OpenAI

    client = OpenAI()
    jsonl = "\n".join(_openai_jsonl_line(r, model) for r in requests).encode("utf-8")
    f = client.files.create(file=("batch.jsonl", io.BytesIO(jsonl)), purpose="batch")
    batch = client.batches.create(
        input_file_id=f.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"[batch] openai submitted: {batch.id}  ({len(requests)} requests)")
    return batch.id


def wait_openai(batch_id: str, poll_interval: int = 30, timeout: int = 86400) -> dict[str, str]:
    from openai import OpenAI

    client = OpenAI()
    deadline = time.time() + timeout
    while time.time() < deadline:
        b = client.batches.retrieve(batch_id)
        s = b.status
        rc = b.request_counts
        print(f"[batch] openai {batch_id} status={s} done={rc.completed}/{rc.total}")
        if s in ("completed", "failed", "cancelled", "expired"):
            break
        time.sleep(poll_interval)
    else:
        raise TimeoutError(f"OpenAI batch {batch_id} did not complete in {timeout}s")

    if b.status != "completed":
        raise RuntimeError(f"OpenAI batch {batch_id} ended with status={b.status}")

    body = client.files.content(b.output_file_id).read().decode("utf-8")
    out: dict[str, str] = {}
    for line in body.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid = rec["custom_id"]
        try:
            out[cid] = rec["response"]["body"]["choices"][0]["message"]["content"] or ""
        except Exception:
            out[cid] = ""
    return out


# ---- unified entry ----------------------------------------------------------

def batch_chat(model: str, requests: list[BatchRequest]) -> list[str]:
    """Submit and wait. Results returned in input order."""
    if not requests:
        return []
    if model.startswith("claude"):
        bid = submit_anthropic(model, requests)
        results = wait_anthropic(bid)
    else:
        bid = submit_openai(model, requests)
        results = wait_openai(bid)
    # Cache the raw results to disk so a re-run doesn't re-pay.
    PATHS.ensure()
    (_BATCH_DIR / f"batch_{bid}.json").write_text(json.dumps(results), encoding="utf-8")
    return [results.get(r.custom_id, "") for r in requests]
