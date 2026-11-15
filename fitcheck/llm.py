"""Helpers for the optional Claude calls: a startup key check and readable error logs.

Every Claude call in fitcheck uses structured outputs (output_config.format), which the
anthropic package supports from 0.77.0. An older package raises a TypeError on every
call, so the check below looks at the installed version first.

Errors are logged to stderr as one line: where, exception type, HTTP status, message.
The API key is never printed; anything that looks like a key is redacted.
"""
from __future__ import annotations

import json
import os
import re
import sys

from .config import MODEL

MIN_SDK = (0, 77, 0)
MIN_SDK_TEXT = ".".join(map(str, MIN_SDK))
INSTALL = f"python3 -m pip install -U 'anthropic>={MIN_SDK_TEXT}'"
_KEYISH = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")


def _redact(text: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key and len(key) > 8:
        text = text.replace(key, "[redacted]")
    return _KEYISH.sub("[redacted]", text)


def describe(e: BaseException) -> str:
    """One line: exception type, HTTP status if any, API error type if any, message."""
    parts = [type(e).__name__]
    status = getattr(e, "status_code", None)
    if status:
        parts.append(str(status))
    etype = getattr(e, "type", None)
    if isinstance(etype, str) and etype:
        parts.append(etype)
    msg = getattr(e, "message", None) or str(e)
    msg = " ".join(_redact(str(msg)).split())
    if len(msg) > 300:
        msg = msg[:297] + "..."
    return " ".join(parts) + (f": {msg}" if msg else "")


def log(where: str, what: str) -> None:
    print(f"[fitcheck] {where}: {what}", file=sys.stderr, flush=True)


def log_error(where: str, e: BaseException) -> None:
    log(where, f"Claude call failed: {describe(e)}")


def sdk_version() -> tuple[str | None, tuple[int, ...]]:
    try:
        import anthropic
    except ImportError:
        return None, ()
    v = getattr(anthropic, "__version__", "0")
    nums = tuple(int(n) for n in re.findall(r"\d+", v)[:3])
    return v, nums


def sdk_problem() -> str | None:
    """Why live calls can't work with the installed package, or None if it's fine."""
    v, nums = sdk_version()
    if v is None:
        return "the anthropic package isn't installed. Run: python3 -m pip install -r requirements.txt"
    if nums < MIN_SDK:
        return f"anthropic {v} is too old for structured outputs (needs {MIN_SDK_TEXT}+). Run: {INSTALL}"
    return None


CHECK_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def key_check(client=None, model: str = MODEL) -> tuple[bool, str]:
    """A tiny structured-output call, the same shape the demo uses. Returns (ok, one line)."""
    if client is None:
        problem = sdk_problem()
        if problem:
            return False, f"Key check FAILED: {problem}"
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False, "Key check skipped: ANTHROPIC_API_KEY isn't set, so the demo runs in no-key mode."
        import anthropic

        client = anthropic.Anthropic()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1000,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": CHECK_SCHEMA}},
            messages=[{"role": "user", "content": 'Reply with {"ok": true}.'}],
        )
    except Exception as e:
        return False, "Key check FAILED: " + _explain(e, model)
    text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
    try:
        json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False, f"Key check FAILED: {model} answered but not with JSON (stop_reason {resp.stop_reason})."
    v, _ = sdk_version()
    return True, f"Key check OK: {model} answered" + (f" (anthropic {v})." if v else ".")


def _explain(e: BaseException, model: str) -> str:
    name = type(e).__name__
    status = getattr(e, "status_code", None)
    detail = describe(e)
    if name == "AuthenticationError" or status == 401:
        return f"the API key was rejected (401). Check ANTHROPIC_API_KEY. [{detail}]"
    if name == "PermissionDeniedError" or status == 403:
        return f"this key isn't allowed to use {model} (403). [{detail}]"
    if name == "NotFoundError" or status == 404:
        return f"model {model} isn't available to this key (404). Set FITCHECK_MODEL to another model. [{detail}]"
    if name == "RateLimitError" or status in (402, 429):
        return f"rate limit or credit balance (status {status}). [{detail}]"
    if name in ("APIConnectionError", "APITimeoutError"):
        return f"couldn't reach the Claude API. [{detail}]"
    if name == "TypeError":
        return f"the installed anthropic package doesn't accept this call; {sdk_problem() or 'upgrade it: ' + INSTALL} [{detail}]"
    return detail
