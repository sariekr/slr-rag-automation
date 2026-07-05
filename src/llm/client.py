"""
OpenRouter chat-completions client.

Token usage and latency are returned with every call so callers can track cost
and timing.

Uses `requests` (declared in requirements.txt) against the OpenRouter
chat-completions endpoint. The call pattern (Bearer auth, 429/5xx retry with
backoff) is a standard resilient HTTP client. `requests` is imported lazily so
the module loads before `pip install`, matching the rest of the package.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL,
    LLM_MAX_TOKENS, LLM_TEMPERATURE,
)

MAX_RETRIES = 4
RETRY_DELAY = 2.0          # base seconds; scaled per attempt
REQUEST_TIMEOUT = 120      # seconds per request


@dataclass
class LLMResponse:
    """One completion plus telemetry for cost/time metrics."""
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    finish_reason: str = ""    # "stop" | "length" (truncated) | ...: lets callers detect cut-off output


def call_llm(prompt: str, system: str = "", model: str = LLM_MODEL,
             max_tokens: int = LLM_MAX_TOKENS) -> LLMResponse:
    """Single chat completion via OpenRouter.

    Returns the raw completion text plus token usage and wall-clock latency.
    Parsing the text (e.g. JSON for extraction) is the task layer's job.
    max_tokens defaults to config.LLM_MAX_TOKENS; raise it for batched outputs
    (e.g. extracting many papers in one long-context call).

    Retries 429 and 5xx with backoff; a 4xx (bad request / auth) raises at once.
    Raises RuntimeError on a missing key or after exhausting retries; the task
    layer should catch this so one failed paper does not abort a whole run.
    """
    import requests

    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set; export it before running the LLM arm."
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    }

    last_err = ""
    for attempt in range(MAX_RETRIES):
        t0 = time.time()
        try:
            resp = requests.post(
                OPENROUTER_BASE_URL, json=payload, headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as e:        # network / timeout
            last_err = f"request error: {e}"
            time.sleep(RETRY_DELAY * (attempt + 1))
            continue

        if resp.status_code == 429:                    # rate limited -> longer backoff
            last_err = "429 rate limited"
            time.sleep(RETRY_DELAY * (attempt + 1) * 2)
            continue
        if resp.status_code >= 500:                    # transient server error
            last_err = f"{resp.status_code} server error"
            time.sleep(RETRY_DELAY * (attempt + 1))
            continue
        if resp.status_code != 200:                    # 4xx -> not retryable
            raise RuntimeError(f"OpenRouter {resp.status_code}: {resp.text[:300]}")

        latency = time.time() - t0
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=latency,
            finish_reason=str(choice.get("finish_reason") or ""),
        )

    raise RuntimeError(f"OpenRouter failed after {MAX_RETRIES} attempts ({last_err}).")
