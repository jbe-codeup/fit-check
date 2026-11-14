"""Eval runner. Run after every change: `python -m fitcheck eval`.

Retrieval stage (no API key needed):
  - hit@k:        an expected note is in the top-k chunks for answerable questions
  - pass rate:    answerable questions clear the "I don't know" threshold
  - IDK accuracy: unanswerable questions fall below it (no model call at all)
  - personal:     salary/family/etc. questions are redirected to the candidate
Generation stage (--generate, uses ANTHROPIC_API_KEY): status matches the
expectation and every citation points at a retrieved chunk of an expected note.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fitcheck import answer
from fitcheck.answer import SENSITIVE, ask, retrieve

QUESTIONS = Path(__file__).with_name("questions.json")


def run(generate: bool = False, questions: str | None = None) -> int:
    path = Path(questions) if questions else QUESTIONS
    print(f"eval set: {path.name}   notes: {answer.config.NOTES_DIR.name}")
    qs = json.loads(path.read_text(encoding="utf-8"))
    rows, stats = [], {"answer": [0, 0, 0], "idk": [0, 0], "personal": [0, 0], "gen": [0, 0]}
    for item in qs:
        q, exp = item["q"], item["expect"]
        if SENSITIVE.search(q):
            got, hits, ok = "personal", [], False
        else:
            hits, ok = retrieve(q)
            got = "answer" if ok else "idk"
        top_notes = [h["note"] for h in hits]
        hit = bool(set(item.get("notes", [])) & set(top_notes))
        if exp == "answer":
            stats["answer"][0] += 1
            stats["answer"][1] += hit
            stats["answer"][2] += got == "answer"
            passed = hit and got == "answer"
        else:
            stats[exp][0] += 1
            stats[exp][1] += got == exp
            passed = got == exp
        row = {"q": q, "expect": exp, "got": got, "top": hits[0]["score"] if hits else 0.0,
               "top_note": top_notes[0] if top_notes else "", "pass": passed}

        if generate:
            r = ask(q)
            want = {"answer": "answered", "idk": "idk", "personal": "personal"}[exp]
            cites_ok = all(c["note"] in item.get("notes", []) for c in r["citations"]) if r["citations"] else want != "answered"
            gpass = r["status"] == want and cites_ok
            stats["gen"][0] += 1
            stats["gen"][1] += gpass
            row.update(gen_status=r["status"], gen_pass=gpass, gen_answer=r["answer"][:160])
        rows.append(row)

    print(f"{'pass':5} {'expect':8} {'got':8} {'score':>6}  question  ->  top note")
    for r in rows:
        mark = "ok" if r["pass"] else "MISS"
        print(f"{mark:5} {r['expect']:8} {r['got']:8} {r['top']:6.3f}  {r['q']}  ->  {r['top_note']}")
        if generate:
            print(f"      generation: {r['gen_status']} {'ok' if r['gen_pass'] else 'MISS'} | {r['gen_answer']}")

    a, i, p = stats["answer"], stats["idk"], stats["personal"]
    print()
    print(f"threshold {answer.THRESHOLD}")
    print(f"answerable: hit@{answer.TOP_K} {a[1]}/{a[0]}, cleared threshold {a[2]}/{a[0]}")
    print(f"unanswerable: correct \"I don't know\" {i[1]}/{i[0]}")
    print(f"personal: redirected {p[1]}/{p[0]}")
    if generate:
        print(f"generation: {stats['gen'][1]}/{stats['gen'][0]} matched status with valid citations")
    total = sum(r["pass"] for r in rows)
    print(f"overall retrieval-stage pass: {total}/{len(rows)}")
    return 0 if total == len(rows) else 1


if __name__ == "__main__":
    sys.exit(run(generate="--generate" in sys.argv))
