"""Job 1 (Principal Backend) must be a clear Fit and Job 2 (iOS) a clear Not a fit, on the evidence.

Live mode writes its own questions, so besides the scripted ones this checks the wordings a
live panel tends to use: every Job 1 must-have has to retrieve a cited answer from the right note.
No key or network needed."""
import os
import unittest
from unittest import mock

os.environ.setdefault("FITCHECK_NOTES", "sample_notes")

from fitcheck import answer
from demo import recruiter

# requirement area -> (live-style wordings, the note that should answer it)
LIVE_WORDINGS = {
    "event-driven services in Go and Python on Lambda": ([
        "Has the candidate designed and built event-driven services in Go and Python on AWS Lambda?",
        "Describe the candidate's experience designing event-driven services in Go and Python on AWS Lambda.",
    ], "Serverless Tracking Pipeline on Lambda"),
    "design trade-offs": ([
        "What trade-offs did the candidate make when choosing Lambda?",
        "Why did the candidate pick Lambda over containers, and what was the trade-off?",
    ], "Serverless Tracking Pipeline on Lambda"),
    "cold starts and concurrency": ([
        "How has the candidate kept cold starts and concurrency under control in production?",
        "At peak, how did the candidate keep cold starts and concurrency limits in check?",
    ], "Serverless Tracking Pipeline on Lambda"),
    "failures and retries": ([
        "How does the candidate handle failures and retries safely in distributed systems?",
        "How did the candidate make retries safe so an update is never applied twice?",
    ], "Serverless Tracking Pipeline on Lambda"),
    "debugging under load": ([
        "Tell me about a time the candidate debugged a failure under production load.",
        "How does the candidate debug issues under load?",
    ], "Serverless Tracking Pipeline on Lambda"),
    "ownership including on-call": ([
        "What services has the candidate owned end to end, including on-call?",
        "Describe the candidate's on-call experience.",
    ], "Serverless Tracking Pipeline on Lambda"),
    "leading across teams": ([
        "Tell me about a project the candidate led across multiple teams.",
        "Describe a time the candidate led cross-team work.",
    ], "Cross-Team API Versioning and Mentoring"),
    "mentoring": ([
        "How has the candidate mentored other engineers?",
    ], "Cross-Team API Versioning and Mentoring"),
}


class FitJobCoverageTests(unittest.TestCase):
    def test_every_scripted_must_have_gets_a_cited_answer(self):
        for q in recruiter.JOBS["fit"]["questions"]:
            if q["must"]:
                r = answer.ask(q["q"], dry_run=True)
                self.assertEqual(r["status"], "answered", q["q"])
                self.assertTrue(r["citations"], q["q"])

    def test_live_style_wordings_retrieve_the_right_note(self):
        for area, (wordings, note) in LIVE_WORDINGS.items():
            for w in wordings:
                hits, ok = answer.retrieve(w)
                self.assertTrue(ok, f"{area}: no hit above the threshold for {w!r}")
                self.assertIn(note, [h["note"] for h in hits[:3]], f"{area}: {w!r}")
                r = answer.ask(w, dry_run=True)
                self.assertEqual(r["status"], "answered", w)

    def test_ios_must_haves_are_gaps(self):
        for q in recruiter.JOBS["notfit"]["questions"]:
            if q["must"]:
                self.assertNotEqual(answer.ask(q["q"], dry_run=True)["status"], "answered", q["q"])


class LiveVerdictTests(unittest.TestCase):
    """The live verdict follows from the must-haves Claude names as unmet."""
    turns = [
        {"q": "q1", "requirement": "Ownership end to end", "must": True, "status": "answered", "answer": "a",
         "citations": [{"note": "N", "section": "S", "chunk_id": "c"}]},
        {"q": "q2", "requirement": "LLM tooling", "must": False, "status": "idk", "answer": "I don't know.", "citations": []},
    ]

    def judge_with(self, unmet):
        reply = {"unmet_must_haves": unmet, "summary": "s", "matched": [], "gaps": []}
        with mock.patch.object(recruiter, "_claude", return_value=dict(reply)):
            return recruiter.judge({"description": "JD"}, self.turns, live=True)

    def test_no_unmet_must_haves_is_fit(self):
        self.assertEqual(self.judge_with([])["verdict"], "Fit")

    def test_a_missing_nice_to_have_does_not_decide(self):
        self.assertEqual(self.judge_with(["LLM tooling"])["verdict"], "Fit")

    def test_a_named_unmet_must_have_is_not_a_fit(self):
        v = self.judge_with(["ownership end-to-end"])
        self.assertEqual(v["verdict"], "Not a fit")
        self.assertEqual(v["unmet"], ["Ownership end to end"])


if __name__ == "__main__":
    unittest.main()
