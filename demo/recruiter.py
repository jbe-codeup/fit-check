"""The recruiter agent: reads a job description, interviews the candidate agent over A2A, and gives a verdict.

It is a separate agent from the candidate agent and only talks to it over HTTP with A2A
JSON-RPC (`message/send`), the same way a third-party recruiting agent would.

Two modes:
  - no key (default): questions come from the job's script in demo/jobs.json and
    the verdict is a plain rule: Fit only if every must-have got a cited answer.
  - ANTHROPIC_API_KEY set: Claude (FITCHECK_MODEL, Haiku by default) writes the
    questions from the job description and writes the verdict from the transcript.
    If a call fails it falls back to the script / rule and says so.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Iterator

HERE = Path(__file__).resolve().parent
JOBS = json.loads((HERE / "jobs.json").read_text(encoding="utf-8"))
MODEL = os.environ.get("FITCHECK_MODEL", "claude-haiku-5-5")


# ---------------------------------------------------------------- A2A client
class A2AClient:
    def __init__(self, base_url: str, token: str | None = None):
        self.base = base_url.rstrip("/")
        self.token = token or os.environ.get("FITCHECK_A2A_TOKEN")
        self.context_id = str(uuid.uuid4())

    def _req(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("content-type", "application/json")
        if self.token:
            req.add_header("authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())

    def card(self) -> dict:
        return self._req("GET", "/.well-known/agent-card.json")

    def build_request(self, text: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4())[:8],
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": str(uuid.uuid4()),
                    "contextId": self.context_id,
                    "parts": [{"kind": "text", "text": text}],
                }
            },
        }

    def send(self, request: dict) -> dict:
        return self._req("POST", "/", request)


def parse_reply(resp: dict) -> dict:
    if "error" in resp:
        return {"text": resp["error"].get("message", "error"), "status": "error", "citations": []}
    parts = resp["result"]["parts"]
    text = " ".join(p["text"] for p in parts if p.get("kind") == "text")
    data = next((p["data"] for p in parts if p.get("kind") == "data"), {})
    return {"text": text, "status": data.get("status", "answered"), "citations": data.get("citations", []),
            "disclosure": data.get("disclosure"), "mode": data.get("mode"), "model": data.get("model")}


# ---------------------------------------------------------------- Claude (optional)
def _claude(system: str, user: str, schema: dict) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=system,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": user}],
    )
    text = next(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return json.loads(text)


QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"q": {"type": "string"}, "requirement": {"type": "string"}, "must": {"type": "boolean"}},
                "required": ["q", "requirement", "must"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["Fit", "Not a fit"]},
        "summary": {"type": "string"},
        "matched": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "summary", "matched", "gaps"],
    "additionalProperties": False,
}

RECRUITER_SYSTEM = """You are a recruiter's screening agent. You are interviewing another agent, the candidate agent,
which represents a candidate and answers only from the candidate's own notes, with citations.
Be fair and evidence-based: count a requirement as met only if an answer with citations supports it.
"I don't know" means there is no evidence, not that the candidate lacks the skill; say "no evidence in their notes"."""


def plan_questions(job: dict, live: bool) -> tuple[list[dict], str]:
    if live:
        try:
            out = _claude(
                RECRUITER_SYSTEM,
                "Job description:\n" + job["description"] +
                "\n\nWrite exactly 5 short screening questions about the candidate (call them 'the candidate'), "
                "one per key requirement, most important first. Mark must-haves with must=true (at most 4).",
                QUESTION_SCHEMA,
            )
            qs = [q for q in out["questions"] if q.get("q")][:5]
            if qs:
                return qs, "claude"
        except Exception as e:  # fall back to the script, and say so
            return job["questions"], f"script (Claude unavailable: {type(e).__name__})"
    return job["questions"], "script"


def rule_verdict(turns: list[dict]) -> dict:
    musts = [t for t in turns if t["must"]]
    met = [t["requirement"] for t in turns if t["status"] == "answered"]
    gaps = [t["requirement"] for t in turns if t["status"] != "answered"]
    must_met = [t for t in musts if t["status"] == "answered"]
    fit = bool(musts) and len(must_met) == len(musts)
    if fit:
        extra = [t["requirement"] for t in turns if not t["must"] and t["status"] == "answered"]
        summary = (f"All {len(musts)} must-haves are backed by cited notes"
                   + (f", plus the nice-to-have: {', '.join(extra)}." if extra else "."))
    else:
        missing = [t["requirement"] for t in musts if t["status"] != "answered"]
        summary = (f"{len(missing)} of {len(musts)} must-haves have no evidence in the candidate's notes: {', '.join(missing)}."
                   + (f" The notes do cover {', '.join(met)}, but that isn't enough for this role." if met else ""))
    return {"verdict": "Fit" if fit else "Not a fit", "summary": summary, "matched": met, "gaps": gaps, "by": "rule"}


def judge(job: dict, turns: list[dict], live: bool) -> dict:
    if live:
        try:
            transcript = "\n\n".join(
                f"Q ({'must-have' if t['must'] else 'nice-to-have'}: {t['requirement']}): {t['q']}\n"
                f"A [{t['status']}, {len(t['citations'])} citations]: {t['answer']}"
                for t in turns
            )
            out = _claude(
                RECRUITER_SYSTEM,
                f"Job description:\n{job['description']}\n\nInterview transcript:\n{transcript}\n\n"
                "Give the verdict. It is Fit only if every must-have is supported by a cited answer. "
                "Summary: 1-2 sentences a hiring manager can read in five seconds.",
                VERDICT_SCHEMA,
            )
            return {**out, "by": "claude"}
        except Exception as e:
            return {**rule_verdict(turns), "by": f"rule (Claude unavailable: {type(e).__name__})"}
    return rule_verdict(turns)


# ---------------------------------------------------------------- the interview
def interview(job_id: str | dict, url: str, live: bool | None = None, pace: float = 1.0) -> Iterator[dict]:
    """Yield UI events for one interview. pace=0 for no pauses.

    job_id is a key of demo/jobs.json, or a job dict with the same fields
    (title, description, questions); the evals pass their own sample jobs this way."""
    if live is None:
        live = bool(os.environ.get("ANTHROPIC_API_KEY"))
    job = JOBS[job_id] if isinstance(job_id, str) else job_id
    if not isinstance(job_id, str):
        job_id = job.get("id", "custom")
    a2a = A2AClient(url)

    def wait(s):
        if pace:
            time.sleep(s * pace)

    card = a2a.card()
    yield {"type": "start", "job": {k: job.get(k, "") for k in ("label", "title", "company", "sample_note", "description")},
           "job_id": job_id, "recruiter_mode": f"Claude {MODEL}" if live else "scripted, no key"}
    yield {"type": "card", "card": card, "url": url}
    wait(0.8)

    yield {"type": "thinking", "who": "recruiter"}
    questions, qsource = plan_questions(job, live)
    yield {"type": "plan", "questions": questions, "source": qsource}
    wait(0.6)

    turns = []
    for i, item in enumerate(questions):
        req = a2a.build_request(item["q"])
        yield {"type": "question", "i": i, "text": item["q"], "requirement": item["requirement"],
               "must": item["must"], "rpc": req}
        wait(1.0)
        yield {"type": "thinking", "who": "candidate"}
        t0 = time.time()
        resp = a2a.send(req)
        ms = int((time.time() - t0) * 1000)
        reply = parse_reply(resp)
        wait(0.7)
        turns.append({**item, "answer": reply["text"], "status": reply["status"], "citations": reply["citations"]})
        yield {"type": "answer", "i": i, "text": reply["text"], "status": reply["status"],
               "citations": reply["citations"], "mode": reply.get("mode"), "model": reply.get("model"),
               "disclosure": reply.get("disclosure"), "ms": ms, "rpc": resp}
        wait(0.9)

    yield {"type": "thinking", "who": "recruiter"}
    verdict = judge(job, turns, live)
    wait(0.8)
    yield {"type": "verdict", **verdict}
    yield {"type": "done"}


# ---------------------------------------------------------------- terminal version
def run_cli(job_ids: list[str], url: str, fast: bool = False) -> int:
    for job_id in job_ids:
        for ev in interview(job_id, url, pace=0 if fast else 0.3):
            t = ev["type"]
            if t == "start":
                print(f"\n=== {ev['job']['label']}: {ev['job']['title']} ({ev['job']['sample_note']})")
                print(f"    recruiter: {ev['recruiter_mode']}")
            elif t == "card":
                print(f"    discovered A2A agent: {ev['card']['name']} at {ev['url']}")
            elif t == "question":
                print(f"\nRecruiter > {ev['text']}")
            elif t == "answer":
                print(f"Candidate > {ev['text']}")
                for c in ev["citations"]:
                    print(f"            source: {c['note']}")
            elif t == "verdict":
                print(f"\nVerdict: {ev['verdict']}\n  {ev['summary']}")
    return 0
