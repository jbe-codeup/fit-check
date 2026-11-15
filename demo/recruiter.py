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
    scorecard from the job description, and the panel runs as a team: the two agents are
    separate Claude calls that read the shared transcript and, before each question, propose
    who asks what next, including follow-ups on each other's topics (see _team_loop).
    Each gives its own read, then the panel agrees a verdict or flags a split.
    If a call fails it falls back to the planned questions / rule and says so.
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

# The verdict follows from a per-must-have call: Claude names the must-haves without cited evidence,
# and the panel says Fit exactly when that list is empty. So a Not a fit always names its reason.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "unmet_must_haves": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "matched": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["unmet_must_haves", "summary", "matched", "gaps"],
    "additionalProperties": False,
}

RECRUITER_SYSTEM = """You are a two-agent screening panel: a Hiring Manager agent (ownership, cross-team work, motivation, team fit) and a
Senior Engineer agent (design trade-offs, failure modes, debugging under load, scale); each asks its own questions. You are interviewing another agent, the candidate agent,
which represents a candidate and answers only from the candidate's own notes, with citations.
Be fair and evidence-based: count a requirement as met only if an answer with citations supports it.
A cited answer that shows real hands-on experience with the requirement meets it; it does not need to match every word of the
job description or be perfect. Nice-to-haves never decide the verdict.
"I don't know" means there is no evidence, not that the candidate lacks the skill; say "no evidence in their notes"."""


def plan_questions(job: dict, live: bool) -> tuple[list[dict], str]:
    if live:
        try:
            out = _claude(
                RECRUITER_SYSTEM,
                "Job description:\n" + job["description"] +
                "\n\nWrite exactly 6 short screening questions about the candidate (call them 'the candidate'), "
                "at least as many from the Senior Engineer as from the Hiring Manager, one per key requirement, most important first. "
                "Each question covers one requirement that the job description states. Ask about what the candidate has done, never "
                "hypotheticals like 'how would you...', because the candidate's agent answers only from the candidate's own notes. "
                "Mark must-haves with must=true (at most 5): core requirements the description states as part of the job. "
                "Anything the description calls 'a plus', 'nice to have' or 'bonus' is must=false. "
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


def match_unmet(names: list[str], turns: list[dict]) -> list[str]:
    """Map Claude's unmet list onto the scorecard's must-haves. A name that is clearly a
    nice-to-have is dropped (nice-to-haves never decide the verdict); a name that matches
    nothing stays, so an unclear answer leans to Not a fit rather than a false Fit."""
    norm = lambda s: " ".join(re.findall(r"[a-z0-9]+", s.lower()))
    musts = {norm(t["requirement"]): t["requirement"] for t in turns if t["must"]}
    nices = {norm(t["requirement"]) for t in turns if not t["must"]}
    out = []
    for name in names:
        n = norm(name)
        if not n:
            continue
        hit = next((orig for k, orig in musts.items() if k == n or k in n or n in k), None)
        if hit:
            out.append(hit)
        elif not any(k == n or k in n or n in k for k in nices):
            out.append(name)
    return list(dict.fromkeys(out))


def judge(job: dict, turns: list[dict], live: bool) -> dict:
    if live:
        try:
            transcript = "\n\n".join(
                f"Q ({'must-have' if t['must'] else 'nice-to-have'}: {t['requirement']}): {t['q']}\n"
                f"A [{t['status']}, {len(t['citations'])} citations]: {t['answer']}"
                + (f"\nFollow-up A: {t['followup']}" if t.get("followup") else "")
                for t in turns
            )
            musts = [t["requirement"] for t in turns if t["must"]]
            out = _claude(
                RECRUITER_SYSTEM,
                f"Job description:\n{job['description']}\n\nInterview transcript:\n{transcript}\n\n"
                f"Must-haves: {json.dumps(musts)}\n\n"
                "unmet_must_haves: copy the exact name of each must-have above that no cited answer supports; leave it empty if every one is supported. "
                "A must-have is met when a cited answer shows real hands-on experience with it; it does not "
                "need to match every word of the job description or be perfect. Fit when every must-have is met this way; "
                "Not a fit when any must-have has no cited evidence. Nice-to-haves never decide the verdict. "
                "Summary: 1-2 sentences a hiring manager can read in five seconds; it calls the candidate a fit exactly when unmet_must_haves is empty.",
                VERDICT_SCHEMA,
            )
            unmet = match_unmet(out.pop("unmet_must_haves"), turns)
            return {**out, "verdict": "Not a fit" if unmet else "Fit", "unmet": unmet, "scorecard": scorecard(turns), "by": "claude"}
        except Exception as e:
            llm.log_error("recruiter, verdict", e)
            return {**rule_verdict(turns), "by": f"rule (Claude unavailable: {type(e).__name__})"}
    return rule_verdict(turns)



# ---------------------------------------------------------------- the panel as a team (live mode)
# With a key, the Hiring Manager and the Senior Engineer are two separate Claude calls, each
# with its own role and goals. Before every question both read the job, the shared scorecard
# and the transcript so far, and each proposes a move: ask a new topic, follow up on an
# answer (often one their teammate's question surfaced), or pass. A small floor rule picks
# whose move goes ahead. At the end each gives its own read, and the panel agrees a verdict
# or flags a split. Without a key the interview stays scripted, so tests and evals are stable.
MAX_TURNS = int(os.environ.get("FITCHECK_MAX_TURNS", "0"))  # 0 = scorecard rows + 3 follow-ups
MAX_FOLLOWUPS_PER_TOPIC = 1

TEAM_COMMON = """You are one of two agents on a screening panel that is interviewing another agent, the candidate agent.
The candidate agent represents a candidate and answers only from the candidate's own notes, with citations.
Call the person "the candidate". Your teammate sees the same job, scorecard and transcript. Work as a team:
don't repeat what has been answered, build on each other's questions, and leave topics in your teammate's area to them unless you have a sharp follow-up.
Ask about what the candidate has done, never hypotheticals like "how would you...", because the candidate agent answers only from the notes.
Be fair and evidence-based: a requirement is met when a cited answer shows real hands-on experience with it; it doesn't need to match
every word of the job description or be perfect. Nice-to-haves never decide the verdict. "I don't know" means no evidence in the notes, not a missing skill."""

ROLE_SYSTEM = {
    "hm": TEAM_COMMON + """

You are the Hiring Manager agent. You own the role. Your goals: what the candidate owned end to end, business impact,
how they work across teams, why they are looking, and team fit. You don't run technical deep dives; hand those to the Senior Engineer.
Follow up when an answer leaves the candidate's own part, the scope or the outcome for the business unclear.""",
    "eng": TEAM_COMMON + """

You are the Senior Engineer agent. Your goals: senior-level depth. Design trade-offs and why, failure modes and retries,
debugging under load, scale with real numbers. Drill into vague or number-free claims, including technical claims
that come up in answers to the Hiring Manager's questions. Ask one precise question at a time, the way a senior engineer would in a loop.""",
}

MOVE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["ask", "follow_up", "pass"]},
        "topic": {"type": "integer"},
        "question": {"type": "string"},
        "why": {"type": "string"},
        "urgency": {"type": "integer"},
    },
    "required": ["action", "topic", "question", "why", "urgency"],
    "additionalProperties": False,
}

READ_SCHEMA = {  # each agent's verdict follows from the must-haves it names as unmet
    "type": "object",
    "properties": {
        "unmet_must_haves": {"type": "array", "items": {"type": "string"}},
        "read": {"type": "string"},
    },
    "required": ["unmet_must_haves", "read"],
    "additionalProperties": False,
}


def _clip(text: str, n: int = 600) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _board(job: dict, rows: list[dict], log: list[dict]) -> str:
    lines = ["Job description:", job["description"], "", "Shared scorecard (topic number: requirement):"]
    for i, r in enumerate(rows):
        kind = "must-have" if r["must"] else "nice-to-have"
        if not r["asked"]:
            state = "not asked yet"
        else:
            state = ("met, cites " + ", ".join(notes_of(r["citations"]))) if r["status"] == "answered" else "no evidence so far"
            state += f"; follow-ups so far: {r['followups']}"
        lines.append(f"  {i}: {r['requirement']} ({kind}) - {state}. Suggested opener: {r['q']}")
    lines += ["", "Transcript so far:" if log else "Transcript so far: nothing yet, this is the first question."]
    for t in log:
        who = PANEL[t["by"]]["name"].replace(" Agent", "")
        lines.append(f"  [{who}, topic {t['i']}{', follow-up' if t['followup'] else ''}] Q: {t['q']}")
        lines.append(f"  [Candidate agent, {t['status']}{', cites ' + ', '.join(t['notes']) if t['notes'] else ''}] A: {_clip(t['a'])}")
    return "\n".join(lines)


def _move_prompt(by: str, board: str, left: int, unasked: list[int], must_cover: bool) -> str:
    mate = "Senior Engineer" if by == "hm" else "Hiring Manager"
    rule = (f"Only {left} question(s) left and topics {unasked} are not asked yet, so you must ask one of those (action 'ask')."
            if must_cover else
            f"{left} question(s) left in the interview. Topics not asked yet: {unasked or 'none'}.")
    return (board + "\n\n" + rule + "\n\nPropose your next move. action: 'ask' a topic not asked yet (prefer your own area), "
            f"'follow_up' to drill into an answer already given (often to one the {mate} asked, when it was vague, had no numbers, "
            f"or raised something in your area; at most {MAX_FOLLOWUPS_PER_TOPIC} follow-up per topic), or 'pass' if your teammate should go next or you have nothing worth asking. "
            "topic: the scorecard number. question: one sentence, addressed to the candidate agent. "
            "why: one short line a viewer reads, saying why you're asking this now (e.g. 'Drilling into the retry claim from the HM's question'). "
            "urgency: 1 to 5, how much this question matters right now.")


def propose_moves(job: dict, rows: list[dict], log: list[dict], left: int, must_cover: bool) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    board = _board(job, rows, log)
    unasked = [i for i, r in enumerate(rows) if not r["asked"]]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {by: pool.submit(_claude, ROLE_SYSTEM[by], _move_prompt(by, board, left, unasked, must_cover), MOVE_SCHEMA)
                for by in PANEL}
        return {by: f.result() for by, f in futs.items()}


def pick_move(moves: dict, rows: list[dict], asked: dict, must_cover: bool) -> dict | None:
    """The floor rule: the most urgent valid move goes first; on a tie a follow-up beats a new
    topic, then whoever has asked less. Near the cap, only unasked topics, so every row is covered."""
    cands = []
    for by, m in moves.items():
        action, i, q = m.get("action"), m.get("topic"), (m.get("question") or "").strip()
        if action not in ("ask", "follow_up") or not q or not isinstance(i, int) or not 0 <= i < len(rows):
            continue
        r = rows[i]
        if action == "ask" and r["asked"]:
            action = "follow_up"
        if action == "follow_up" and (not r["asked"] or r["followups"] >= MAX_FOLLOWUPS_PER_TOPIC):
            continue
        if must_cover and action != "ask":
            continue
        try:
            urgency = max(1, min(5, int(m.get("urgency") or 3)))
        except (TypeError, ValueError):
            urgency = 3
        cands.append({"by": by, "action": action, "topic": i, "question": q, "why": (m.get("why") or "").strip(), "urgency": urgency})
    if not cands:
        return None
    cands.sort(key=lambda c: (-c["urgency"], c["action"] != "follow_up", asked[c["by"]], c["by"] != "eng"))
    return cands[0]


def panel_reads(job: dict, turns: list[dict]) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    transcript = "\n\n".join(
        f"Q ({'must-have' if t['must'] else 'nice-to-have'}: {t['requirement']}): {t['q']}\n"
        f"A [{t['status']}, {len(t['citations'])} citations]: {_clip(t['answer'], 900)}"
        + (f"\nFollow-up A: {_clip(t['followup'], 600)}" if t.get("followup") else "")
        for t in turns
    )
    ask = (f"Job description:\n{job['description']}\n\nInterview transcript:\n{transcript}\n\n"
           f"Must-haves: {json.dumps([t['requirement'] for t in turns if t['must']])}\n\n"
           "Give your own read from your role. unmet_must_haves: the exact name of each must-have above that no cited answer supports; "
           "empty if every one is supported (you think it's a Fit). "
           "read: 1-2 sentences in the first person, as you'd say it to your teammate, naming the strongest evidence and your main concern; "
           "it calls the candidate a fit exactly when unmet_must_haves is empty.")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {by: pool.submit(_claude, ROLE_SYSTEM[by], ask, READ_SCHEMA) for by in PANEL}
        out = {}
        for by, f in futs.items():
            r = f.result()
            unmet = match_unmet(r.get("unmet_must_haves", []), turns)
            out[by] = {"verdict": "Not a fit" if unmet else "Fit", "read": r["read"], "unmet": unmet}
        return out


def joint_verdict(reads: dict, turns: list[dict]) -> dict:
    """Agree a verdict, or flag a split. A Fit always needs cited evidence for every must-have,
    and a Not a fit always names the must-haves it found unsupported."""
    rule = rule_verdict(turns)
    hm, eng = reads["hm"]["verdict"], reads["eng"]["verdict"]
    out = {**rule, "reads": {by: {"verdict": r["verdict"], "read": r["read"], "unmet": r.get("unmet", [])} for by, r in reads.items()}}
    if hm == eng == rule["verdict"]:
        out.update(by="team", split=False, summary=f"Both panel agents say {hm}. " + rule["summary"])
    elif hm == eng == "Not a fit":  # every must-have has a cited answer, but both judge some of it too weak
        unmet = list(dict.fromkeys(reads["hm"].get("unmet", []) + reads["eng"].get("unmet", [])))
        out.update(verdict="Not a fit", by="team", split=False, gaps=unmet,
                   summary="Both panel agents say Not a fit: the cited answers didn't show enough for " + ", ".join(unmet) + ".")
    elif hm == eng:
        out.update(by="team (evidence rule)", split=False,
                   summary=f"Both agents leaned {hm}, but a must-have has no cited evidence, so the panel can't call it a Fit. " + rule["summary"])
    else:
        out.update(by="team (split)", split=True,
                   summary=f"The panel split: the Hiring Manager says {hm}, the Senior Engineer says {eng}. The evidence decides: " + rule["summary"])
    return out

# ---------------------------------------------------------------- the interview
def _team_loop(job: dict, questions: list[dict], ask, answer_event, wait):
    """Live panel: each turn both agents propose a move and the floor rule picks one."""
    rows = [{**q, "asked": False, "status": None, "answer": "", "citations": [], "nearest": [], "followups": 0} for q in questions]
    log, asked = [], {by: 0 for by in PANEL}
    max_turns = MAX_TURNS or len(rows) + 3
    team_ok = True
    for n in range(max_turns):
        unasked = [i for i, r in enumerate(rows) if not r["asked"]]
        left = max_turns - n
        must_cover = left <= len(unasked)
        moves = {}
        if team_ok:
            yield {"type": "thinking", "who": "recruiter"}
            try:
                moves = propose_moves(job, rows, log, left, must_cover)
            except Exception as e:  # keep the interview going on the planned questions
                llm.log_error("panel, choosing the next question", e)
                team_ok = False
        move = pick_move(moves, rows, asked, must_cover)
        if move is None:
            if not unasked:
                break  # both agents passed and every topic is covered
            i = unasked[0]
            move = {"by": asker(rows[i]), "action": "ask", "topic": i, "question": rows[i]["q"], "why": ""}
        i, by, r = move["topic"], move["by"], rows[move["topic"]]
        follow = move["action"] == "follow_up"
        reply, resp, ms = yield from ask(i, by, move["question"], r, followup=follow, why=move["why"])
        asked[by] += 1
        if follow:
            r["followups"] += 1
            if reply["status"] == "answered":
                r["followup"] = (r.get("followup", "") + " " + reply["text"]).strip()
                r["citations"] = r["citations"] + [c for c in reply["citations"] if c not in r["citations"]]
                r["status"] = "answered"
        else:
            r.update(asked=True, asked_by=by, q=move["question"], answer=reply["text"], status=reply["status"],
                     citations=reply["citations"], nearest=reply["nearest"])
        log.append({"by": by, "i": i, "q": move["question"], "a": reply["text"], "status": reply["status"],
                    "notes": notes_of(reply["citations"]), "followup": follow})
        yield answer_event(i, reply, resp, ms, followup=follow)
        wait(0.9)
    turns = []
    for r in rows:
        t = {k: r[k] for k in ("q", "requirement", "must", "answer", "citations", "nearest")}
        t["status"] = r["status"] if r["asked"] else "idk"
        t["probed"] = r["followups"] > 0
        if r.get("followup"):
            t["followup"] = r["followup"]
        turns.append(t)
    return turns


def interview(job_id: str | dict, url: str, live: bool | None = None, pace: float = 1.0,
              team: bool | None = None) -> Iterator[dict]:
    """Yield UI events for one interview. pace=0 for no pauses.

    job_id is a key of demo/jobs.json, or a job dict with the same fields
    (title, description, questions); the evals pass their own sample jobs this way."""
    if live is None:
        live = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if team is None:  # live mode runs the panel as a team; FITCHECK_PANEL=script keeps the scripted order
        team = live and os.environ.get("FITCHECK_PANEL", "team") != "script"
    job = JOBS[job_id] if isinstance(job_id, str) else job_id
    if not isinstance(job_id, str):
        job_id = job.get("id", "custom")
    a2a = A2AClient(url)

    def wait(s):
        if pace:
            time.sleep(s * pace)

    card = a2a.card()
    yield {"type": "start", "job": {k: job.get(k, "") for k in ("label", "title", "company", "sample_note", "description")},
           "job_id": job_id, "recruiter_mode": (f"Claude {MODEL}, panel as a team" if team else f"Claude {MODEL}") if live else "scripted, no key",
           "panel": PANEL, "team": bool(team)}
    yield {"type": "card", "card": card, "url": url}
    wait(0.8)

    yield {"type": "thinking", "who": "recruiter"}
    questions, qsource = plan_questions(job, live)
    yield {"type": "plan", "questions": questions, "source": qsource}
    wait(0.6)

    def ask(i: int, by: str, text: str, item: dict, followup: bool = False, why: str = ""):
        req = a2a.build_request(text, {"from": PANEL[by]["name"]})
        ev = {"type": "question", "i": i, "by": by, "text": text, "requirement": item["requirement"], "must": item["must"], "rpc": req}
        if followup:
            ev["followup"] = True
        if why:
            ev["why"] = why
        yield ev
        wait(1.0)
        yield {"type": "thinking", "who": "candidate"}
        t0 = time.time()
        resp = a2a.send(req)
        ms = int((time.time() - t0) * 1000)
        wait(0.7)
        return parse_reply(resp), resp, ms

    def answer_event(i: int, reply: dict, resp: dict, ms: int, thin: bool = False, followup: bool = False) -> dict:
        ev = {"type": "answer", "i": i, "text": reply["text"], "status": reply["status"], "citations": reply["citations"],
              "nearest": reply["nearest"], "thin": thin, "mode": reply.get("mode"), "model": reply.get("model"),
              "disclosure": reply.get("disclosure"), "ms": ms, "rpc": resp}
        if followup:
            ev["followup"] = True
        return ev

    if team:
        turns = yield from _team_loop(job, questions, ask, answer_event, wait)
    else:
        turns, probes = [], 0
        for i, item in enumerate(questions):
            by = asker(item)
            reply, resp, ms = yield from ask(i, by, item["q"], item)
            turn = {**item, "answer": reply["text"], "status": reply["status"], "citations": reply["citations"],
                    "nearest": reply["nearest"]}
            thin = (by == "eng" and reply["status"] == "answered" and item["must"] and probes < MAX_PROBES and is_thin(reply["text"]))
            yield answer_event(i, reply, resp, ms, thin=thin)
            wait(0.9)

            if thin:
                # The Senior Engineer agent probes once: same story, asking for the numbers.
                probes += 1
                fq = f"Follow-up on {item['requirement'][0].lower() + item['requirement'][1:]}: what was the measurable result?"
                f, resp, ms = yield from ask(i, "eng", fq, item, followup=True)
                turn["probed"] = True
                if f["status"] == "answered":
                    turn["followup"] = f["text"]
                    turn["citations"] = turn["citations"] + [c for c in f["citations"] if c not in turn["citations"]]
                yield answer_event(i, f, resp, ms, followup=True)
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
    verdict = None
    if team:
        try:
            reads = panel_reads(job, turns)
            for by in PANEL:
                yield {"type": "panel_read", "by": by, "verdict": reads[by]["verdict"], "text": reads[by]["read"]}
                wait(1.0)
            verdict = joint_verdict(reads, turns)
        except Exception as e:
            llm.log_error("panel, verdict", e)
    if verdict is None:
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
                if ev.get("why"):
                    print(f"\n   ({who}: {ev['why']})", end="")
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
            elif t == "panel_read":
                print(f"\n{PANEL[ev['by']]['name'].replace(' Agent', '')}'s read > {ev['verdict']}. {ev['text']}")
            elif t == "verdict":
                print(f"\nPanel verdict: {ev['verdict']}\n  {ev['summary']}")
                for c in ev.get("scorecard", []):
                    mark = "met " if c["status"] == "met" else "gap "
                    src = ", ".join(c["evidence"]) or "no evidence in notes"
                    print(f"  [{mark}] {c['requirement']} ({'must' if c['must'] else 'nice'}): {src}")
    return 0
