"""Offline tests for the interview: honest gaps, the two-way question, follow-up
probes and the scorecard verdict. No API key or network beyond 127.0.0.1."""
import os
import threading
import unittest

os.environ.setdefault("FITCHECK_NOTES", "sample_notes")
os.environ.pop("ANTHROPIC_API_KEY", None)

from fitcheck import a2a, answer
from demo import recruiter


def send(text, data=None):
    parts = [{"kind": "text", "text": text}]
    if data:
        parts.append({"kind": "data", "data": data})
    req = {"jsonrpc": "2.0", "id": 1, "method": "message/send",
           "params": {"message": {"role": "user", "messageId": "m1", "parts": parts}}}
    return recruiter.parse_reply(a2a.handle_rpc(req))


class GapTests(unittest.TestCase):
    def test_missing_skill_is_a_gap_with_nearest_experience(self):
        r = answer.ask("Has the candidate shipped native iOS apps in Swift?")
        self.assertEqual(r["status"], "idk")
        self.assertIn("treat it as a gap", r["answer"])
        self.assertTrue(r.get("nearest_citations"))

    def test_gap_never_claims_the_skill(self):
        r = answer.ask("Has the candidate done mobile performance profiling with Instruments?")
        self.assertNotIn("Instruments", r["answer"].split("Nearest real experience")[-1])


class TwoWayTests(unittest.TestCase):
    def test_candidate_asks_about_the_role(self):
        r = send("Anything you'd like to ask?", {"intent": "ask_role"})
        self.assertEqual(r["status"], "question")
        self.assertIn("funded", r["text"])

    def test_role_answer_is_noted(self):
        r = send("Yes, it's funded.", {"intent": "role_answer"})
        self.assertEqual(r["status"], "noted")


class InterviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = a2a.make_server(port=0)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def run_job(self, job_id):
        return list(recruiter.interview(job_id, self.url, live=False, pace=0))

    def test_fit_job(self):
        ev = self.run_job("fit")
        verdict = [e for e in ev if e["type"] == "verdict"][-1]
        self.assertEqual(verdict["verdict"], "Fit")
        self.assertEqual(sum(1 for e in ev if e["type"] == "question" and e.get("followup")), 1)
        self.assertTrue(any(e["type"] == "role_question" for e in ev))
        self.assertTrue(all(row["status"] == "met" for row in verdict["scorecard"]))

    def test_not_fit_job_states_gaps(self):
        ev = self.run_job("notfit")
        verdict = [e for e in ev if e["type"] == "verdict"][-1]
        self.assertEqual(verdict["verdict"], "Not a fit")
        gaps = [row for row in verdict["scorecard"] if row["status"] == "gap"]
        self.assertTrue(gaps and all(row["nearest"] for row in gaps))


if __name__ == "__main__":
    unittest.main()
