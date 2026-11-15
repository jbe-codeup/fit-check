"""Offline tests for the live panel run as a team. The two Claude calls per turn are
replaced by a small scripted teammate, so no key or network beyond 127.0.0.1 is needed."""
import contextlib
import io
import os
import threading
import unittest
from unittest import mock

os.environ.setdefault("FITCHECK_NOTES", "sample_notes")

from fitcheck import a2a
from demo import recruiter


def scripted_team(job, rows, log, left, must_cover):
    """HM opens on its own topics; the SE follows up once on each answer the HM got, then asks its own."""
    def first(by):
        return next((i for i, r in enumerate(rows) if not r["asked"] and recruiter.asker(r) == by), None)

    moves = {}
    i = first("hm")
    moves["hm"] = ({"action": "ask", "topic": i, "question": rows[i]["q"], "why": "My area", "urgency": 3}
                   if i is not None else {"action": "pass", "topic": 0, "question": "", "why": "", "urgency": 1})
    last = log[-1] if log else None
    if last and last["by"] == "hm" and not last["followup"] and last["status"] == "answered":
        moves["eng"] = {"action": "follow_up", "topic": last["i"], "question": "What broke along the way, and how did you find it?",
                        "why": "Drilling into what the HM's question surfaced", "urgency": 5}
    else:
        j = first("eng")
        moves["eng"] = ({"action": "ask", "topic": j, "question": rows[j]["q"], "why": "Technical depth", "urgency": 4}
                        if j is not None else {"action": "pass", "topic": 0, "question": "", "why": "", "urgency": 1})
    return moves


def split_reads(job, turns):
    return {"hm": {"verdict": "Fit", "read": "Strong ownership, cited."},
            "eng": {"verdict": "Not a fit", "read": "I want more numbers on retries."}}


class TeamPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = a2a.make_server(port=0)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        a2a._limiter = a2a._RateLimiter(100000)  # many interviews in one test run

    def run_team(self, job_id="fit", **patches):
        # live=False keeps the planned questions and the candidate in no-key mode; team=True runs the team loop.
        with mock.patch.multiple(recruiter, **patches), contextlib.redirect_stderr(io.StringIO()):
            return list(recruiter.interview(job_id, self.url, live=False, pace=0, team=True))

    def test_engineer_follows_up_on_the_managers_topic(self):
        ev = self.run_team(propose_moves=scripted_team, panel_reads=split_reads)
        qs = [e for e in ev if e["type"] == "question"]
        follow = [e for e in qs if e.get("followup")]
        self.assertTrue(follow)
        self.assertTrue(all(e["by"] == "eng" for e in follow))
        hm_topics = {e["i"] for e in qs if e["by"] == "hm" and not e.get("followup")}
        self.assertTrue({e["i"] for e in follow} <= hm_topics)
        self.assertTrue(all(e.get("why") for e in qs))

    def test_every_topic_is_covered_and_cited(self):
        ev = self.run_team(propose_moves=scripted_team, panel_reads=split_reads)
        plan = next(e for e in ev if e["type"] == "plan")["questions"]
        opened = {e["i"] for e in ev if e["type"] == "question" and not e.get("followup")}
        self.assertEqual(opened, set(range(len(plan))))
        answered = [e for e in ev if e["type"] == "answer" and e["status"] == "answered"]
        self.assertTrue(answered and all(e["citations"] for e in answered))

    def test_split_is_flagged_and_the_evidence_decides(self):
        ev = self.run_team(propose_moves=scripted_team, panel_reads=split_reads)
        reads = [e for e in ev if e["type"] == "panel_read"]
        self.assertEqual({e["by"] for e in reads}, {"hm", "eng"})
        verdict = [e for e in ev if e["type"] == "verdict"][-1]
        self.assertTrue(verdict["split"])
        self.assertEqual(verdict["verdict"], "Fit")  # every must-have has a cited answer
        self.assertIn("split", verdict["summary"])

    def test_agreed_fit_without_evidence_is_not_a_fit(self):
        both_fit = lambda job, turns: {by: {"verdict": "Fit", "read": "Looks good."} for by in recruiter.PANEL}
        ev = self.run_team("notfit", propose_moves=scripted_team, panel_reads=both_fit)
        verdict = [e for e in ev if e["type"] == "verdict"][-1]
        self.assertEqual(verdict["verdict"], "Not a fit")

    def test_claude_failure_falls_back_to_the_plan(self):
        def boom(*a, **k):
            raise RuntimeError("no model")
        ev = self.run_team(propose_moves=boom, panel_reads=boom)
        plan = next(e for e in ev if e["type"] == "plan")["questions"]
        opened = [e["i"] for e in ev if e["type"] == "question" and not e.get("followup")]
        self.assertEqual(opened, list(range(len(plan))))
        self.assertEqual([e for e in ev if e["type"] == "verdict"][-1]["verdict"], "Fit")


class FloorRuleTests(unittest.TestCase):
    rows = [{"asked": True, "followups": 0}, {"asked": False, "followups": 0}]

    def test_urgent_follow_up_wins(self):
        m = recruiter.pick_move({"hm": {"action": "ask", "topic": 1, "question": "q1", "why": "", "urgency": 3},
                                 "eng": {"action": "follow_up", "topic": 0, "question": "q0", "why": "", "urgency": 5}},
                                self.rows, {"hm": 0, "eng": 0}, must_cover=False)
        self.assertEqual((m["by"], m["action"]), ("eng", "follow_up"))

    def test_near_the_cap_only_new_topics(self):
        m = recruiter.pick_move({"eng": {"action": "follow_up", "topic": 0, "question": "q0", "why": "", "urgency": 5}},
                                self.rows, {"hm": 0, "eng": 0}, must_cover=True)
        self.assertIsNone(m)

    def test_board_prompt_builds(self):
        rows = [{"q": "Q?", "requirement": "R", "must": True, "asked": False, "followups": 0, "status": None, "citations": []}]
        p = recruiter._move_prompt("eng", recruiter._board({"description": "JD"}, rows, []), 3, [0], False)
        self.assertIn("follow_up", p)
        self.assertIn("0: R (must-have)", p)


if __name__ == "__main__":
    unittest.main()
