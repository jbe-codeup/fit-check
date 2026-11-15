"""The interview panel: reads a job description, interviews the candidate agent over A2A, and gives a verdict.

The panel is two agents: a Hiring Manager agent (ownership, cross-team work, motivation, team fit) and a
Senior Engineer agent (design trade-offs, failure modes, debugging under load, scale; it probes thin answers). They
share one scorecard and give one combined verdict. Each A2A request says which of them
is asking in a data part ({"from": ...}).

It is a separate agent from the candidate agent and only talks to it over HTTP with A2A
JSON-RPC (`message/send`), the same way a third-party recruiting agent would.

The interview:
  1. Scorecard: the job description becomes must-haves and nice-to-haves.
  2. One question per requirement. An item is met only by an answer with citations.
  3. Follow-ups: a cited answer to a technical must-have with no numbers is thin, so the Senior Engineer
     agent probes once for the measurable result (at most MAX_PROBES per interview).
  4. Two-way: the candidate agent gets to ask one question about the role, answered
     from the job's own facts.
  5. Verdict: Fit only if every must-have is met; it cites the scorecard item by item.

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
import re
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Iterator

from fitcheck import llm

HERE = Path(__file__).resolve().parent
JOBS = json.loads((HERE / "jobs.json").read_text(encoding="utf-8"))
MODEL = os.environ.get("FITCHECK_MODEL", "claude-haiku-5-5")
MAX_PROBES = int(os.environ.get("FITCHECK_MAX_PROBES", "1"))

# The interviewing side is a two-agent panel. Both question the candidate agent over A2A,
# share one scorecard and give one combined verdict.
PANEL = {
    "hm": {"name": "Hiring Manager Agent", "initials": "HM", "focus": "the role's owner: ownership, business impact, cross-team work, team fit"},
    "eng": {"name": "Senior Engineer Agent", "initials": "SE", "focus": "senior-level depth: design trade-offs, failure modes, debugging under load, scale; probes thin answers"},
}
TECH = re.compile(r"lambda|cold start|concurren|retr|idempot|monitor|observab|alert|scal|throughput|incident|outage|latency|architect|design|go\b|python|aws|kubernetes|ios|swift|profil|database|api", re.I)


def asker(item: dict) -> str:
    """Which panel agent asks this question: the script says, else a keyword guess."""
    by = item.get("by")
    return by if by in PANEL else ("eng" if TECH.search(item.get("q", "") + " " + item.get("requirement", "")) else "hm")


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

    def build_request(self, text: str, data: dict | None = None) -> dict:
        parts = [{"kind": "text", "text": text}] + ([{"kind": "data", "data": data}] if data else [])
        return {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4())[:8],
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": str(uuid.uuid4()),
                    "contextId": self.context_id,
                    "parts": parts,
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
            "nearest": data.get("nearest_citations", []),
            "disclosure": data.get("disclosure"), "mode": data.get("mode"), "model": data.get("model")}


def is_thin(text: str) -> bool:
    """A cited answer with no number in it says what, but not how big or how well."""
    return not re.search(r"\d", text)


def notes_of(citations: list[dict]) -> list[str]:
    out = []
    for c in citations:
        if c["note"] not in out:
            out.append(c["note"])
    return out


def role_answer(job: dict, question: str) -> str:
    """Answer the candidate's question from the job's own facts, never from outside knowledge."""
    from fitcheck.retrieval import expand, tokenize

    facts = job.get("role_facts") or []
    terms = set(expand(tokenize(question)))
    picked = [f for f in facts if terms & set(expand(tokenize(f)))] or facts
    return " ".join(picked) if picked else "The job description doesn't say, so I'll find out and follow up."


# ---------------------------------------------------------------- Claude (optional)
def _claude(system: str, user: str, schema: dict) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=system,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": user}],
    )
    text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"reply wasn't valid JSON (stop_reason {resp.stop_reason})") from None


QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"q": {"type": "string"}, "requirement": {"type": "string"}, "must": {"type": "boolean"},
                               "by": {"type": "string", "enum": ["hm", "eng"]}},
                "required": ["q", "requirement", "must", "by"],
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

RECRUITER_SYSTEM = """You are a two-agent screening panel: a Hiring Manager agent (ownership, cross-team work, motivation, team fit) and a
Senior Engineer agent (design trade-offs, failure modes, debugging under load, scale); each asks its own questions. You are interviewing another agent, the candidate agent,
which represents a candidate and answers only from the candidate's own notes, with citations.
Be fair and evidence-based: count a requirement as met only if an answer with citations supports it.
"I don't know" means there is no evidence, not that the candidate lacks the skill; say "no evidence in their notes"."""


def plan_questions(job: dict, live: bool) -> tuple[list[dict], str]:
    if live:
        try:
            out = _claude(
                RECRUITER_SYSTEM,
                "Job description:\n" + job["description"] +
                "\n\nWrite exactly 6 short screening questions about the candidate (call them 'the candidate'), "
                "at least as many from the Senior Engineer as from the Hiring Manager, one per key requirement, most important first. Mark must-haves with must=true (at most 5). "
                "The two askers have different roles and never overlap: by='eng' (Senior Engineer) asks at a senior engineer's level: "
                "design trade-offs, failure modes and retries, debugging under load, scale with numbers; by='hm' (Hiring Manager) for ownership, "
                "business impact, cross-team work, motivation and team fit.",
                QUESTION_SCHEMA,
            )
            qs = [q for q in out["questions"] if q.get("q")][:6]
            if qs:
                return qs, "claude"
        except Exception as e:  # fall back to the script, and say so
            llm.log_error("recruiter, planning questions", e)
            return job["questions"], f"script (Claude unavailable: {type(e).__name__})"
    return job["questions"], "script"


def scorecard(turns: list[dict]) -> list[dict]:
    return [{"requirement": t["requirement"], "must": t["must"], "status": "met" if t["status"] == "answered" else "gap",
             "evidence": notes_of(t["citations"]), "nearest": notes_of(t.get("nearest", [])), "probed": t.get("probed", False)}
            for t in turns]


def rule_verdict(turns: list[dict]) -> dict:
    card = scorecard(turns)
    musts = [c for c in card if c["must"]]
    must_met = [c for c in musts if c["status"] == "met"]
    met = [c["requirement"] for c in card if c["status"] == "met"]
    gaps = [c["requirement"] for c in card if c["status"] != "met"]
    fit = bool(musts) and len(must_met) == len(musts)
    if fit:
        extra = [c["requirement"] for c in card if not c["must"] and c["status"] == "met"]
        summary = (f"{len(must_met)} of {len(musts)} must-haves met, each backed by a cited note"
                   + (f", plus the nice-to-have: {', '.join(extra)}." if extra else "."))
    else:
        missing = [c["requirement"] for c in musts if c["status"] != "met"]
        summary = (f"{len(must_met)} of {len(musts)} must-haves met. No evidence for {', '.join(missing)}, "
                   "and the candidate's agent said so plainly instead of stretching.")
    return {"verdict": "Fit" if fit else "Not a fit", "summary": summary, "matched": met, "gaps": gaps,
            "scorecard": card, "by": "rule"}


def judge(job: dict, turns: list[dict], live: bool) -> dict:
    if live:
        try:
            transcript = "\n\n".join(
                f"Q ({'must-have' if t['must'] else 'nice-to-have'}: {t['requirement']}): {t['q']}\n"
                f"A [{t['status']}, {len(t['citations'])} citations]: {t['answer']}"
                + (f"\nFollow-up A: {t['followup']}" if t.get("followup") else "")
                for t in turns
            )
            out = _claude(
                RECRUITER_SYSTEM,
                f"Job description:\n{job['description']}\n\nInterview transcript:\n{transcript}\n\n"
                "Give the verdict. It is Fit only if every must-have is supported by a cited answer. "
                "Summary: 1-2 sentences a hiring manager can read in five seconds.",
                VERDICT_SCHEMA,
            )
            return {**out, "scorecard": scorecard(turns), "by": "claude"}
        except Exception as e:
            llm.log_error("recruiter, verdict", e)
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
           "job_id": job_id, "recruiter_mode": f"Claude {MODEL}" if live else "scripted, no key", "panel": PANEL}
    yield {"type": "card", "card": card, "url": url}
    wait(0.8)

    yield {"type": "thinking", "who": "recruiter"}
    questions, qsource = plan_questions(job, live)
    yield {"type": "plan", "questions": questions, "source": qsource}
    wait(0.6)

    turns, probes = [], 0
    for i, item in enumerate(questions):
        by = asker(item)
        req = a2a.build_request(item["q"], {"from": PANEL[by]["name"]})
        yield {"type": "question", "i": i, "by": by, "text": item["q"], "requirement": item["requirement"],
               "must": item["must"], "rpc": req}
        wait(1.0)
        yield {"type": "thinking", "who": "candidate"}
        t0 = time.time()
        resp = a2a.send(req)
        ms = int((time.time() - t0) * 1000)
        reply = parse_reply(resp)
        wait(0.7)
        turn = {**item, "answer": reply["text"], "status": reply["status"], "citations": reply["citations"],
                "nearest": reply["nearest"]}
        thin = (by == "eng" and reply["status"] == "answered" and item["must"] and probes < MAX_PROBES and is_thin(reply["text"]))
        yield {"type": "answer", "i": i, "text": reply["text"], "status": reply["status"],
               "citations": reply["citations"], "nearest": reply["nearest"], "thin": thin,
               "mode": reply.get("mode"), "model": reply.get("model"),
               "disclosure": reply.get("disclosure"), "ms": ms, "rpc": resp}
        wait(0.9)

        if thin:
            # The Senior Engineer agent probes once: same story, asking for the numbers.
            probes += 1
            fq = f"Follow-up on {item['requirement'][0].lower() + item['requirement'][1:]}: what was the measurable result?"
            req = a2a.build_request(fq, {"from": PANEL["eng"]["name"]})
            yield {"type": "question", "i": i, "by": "eng", "text": fq, "requirement": item["requirement"], "must": item["must"],
                   "followup": True, "rpc": req}
            wait(1.0)
            yield {"type": "thinking", "who": "candidate"}
            t0 = time.time()
            resp = a2a.send(req)
            ms = int((time.time() - t0) * 1000)
            f = parse_reply(resp)
            wait(0.7)
            turn["probed"] = True
            if f["status"] == "answered":
                turn["followup"] = f["text"]
                turn["citations"] = turn["citations"] + [c for c in f["citations"] if c not in turn["citations"]]
            yield {"type": "answer", "i": i, "text": f["text"], "status": f["status"], "citations": f["citations"],
                   "nearest": f["nearest"], "thin": False, "followup": True, "mode": f.get("mode"),
                   "model": f.get("model"), "disclosure": f.get("disclosure"), "ms": ms, "rpc": resp}
            wait(0.9)
        turns.append(turn)

    # Two-way: the candidate's agent asks one question about the role before the verdict.
    req = a2a.build_request("Before I wrap up: is there anything you'd like to ask about the role?",
                            {"intent": "ask_role", "from": PANEL["hm"]["name"]})
    yield {"type": "invite", "by": "hm", "text": "Before I wrap up: is there anything you'd like to ask about the role?", "rpc": req}
    wait(0.6)
    yield {"type": "thinking", "who": "candidate"}
    resp = a2a.send(req)
    cq = parse_reply(resp)
    wait(0.5)
    yield {"type": "role_question", "text": cq["text"], "rpc": resp}
    wait(0.8)
    yield {"type": "thinking", "who": "recruiter"}
    ra = role_answer(job, cq["text"])
    req = a2a.build_request(ra, {"intent": "role_answer", "from": PANEL["hm"]["name"]})
    resp = a2a.send(req)
    wait(0.6)
    yield {"type": "role_answer", "by": "hm", "text": ra, "rpc": req, "ack": parse_reply(resp)["text"]}
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
                print(f"    panel: {', '.join(p['name'] for p in PANEL.values())} ({ev['recruiter_mode']})")
            elif t == "card":
                print(f"    discovered A2A agent: {ev['card']['name']} at {ev['url']}")
            elif t == "plan":
                print("    scorecard: " + "; ".join(f"{q['requirement']} ({'must' if q['must'] else 'nice'})" for q in ev["questions"]))
            elif t in ("question", "invite"):
                who = PANEL[ev.get("by", "hm")]["name"].replace(" Agent", "")
                print(f"\n{who} > {'(follow-up) ' if ev.get('followup') else ''}{ev['text']}")
            elif t == "answer":
                print(f"Candidate > {ev['text']}")
                for n in notes_of(ev["citations"]):
                    print(f"            source: {n}")
                for n in notes_of(ev.get("nearest") or []):
                    print(f"            nearest experience: {n}")
            elif t == "role_question":
                print(f"Candidate > {ev['text']}")
            elif t == "role_answer":
                print(f"Hiring Manager > {ev['text']}")
            elif t == "verdict":
                print(f"\nPanel verdict: {ev['verdict']}\n  {ev['summary']}")
                for c in ev.get("scorecard", []):
                    mark = "met " if c["status"] == "met" else "gap "
                    src = ", ".join(c["evidence"]) or "no evidence in notes"
                    print(f"  [{mark}] {c['requirement']} ({'must' if c['must'] else 'nice'}): {src}")
    return 0
