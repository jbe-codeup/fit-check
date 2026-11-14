"""Offline tests for the answer path, run against the fictional sample notes.
No API key or network needed: `make test`."""
import json
import os
import unittest

os.environ.setdefault("FITCHECK_NOTES", "sample_notes")
from types import SimpleNamespace

from fitcheck import answer
from fitcheck.answer import AT_LIMIT, IDK, PERSONAL, ask

Q = "How did the candidate improve deploy reliability?"


class FakeError(Exception):
    def __init__(self, msg, status_code=None, type_=""):
        super().__init__(msg)
        self.status_code = status_code
        self.type = type_


class RaisingClient:
    def __init__(self, exc):
        self.messages = SimpleNamespace(create=self._create)
        self.exc = exc

    def _create(self, **kwargs):
        raise self.exc


class ReplyClient:
    def __init__(self, payload, stop_reason="end_turn"):
        self.calls = []
        self.payload = payload
        self.stop_reason = stop_reason
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        text = json.dumps(self.payload)
        return SimpleNamespace(stop_reason=self.stop_reason, content=[SimpleNamespace(type="text", text=text)])


class AtLimitTests(unittest.TestCase):
    def test_rate_limit_returns_friendly_message(self):
        r = ask(Q, client=RaisingClient(FakeError("rate limited", 429, "rate_limit_error")))
        self.assertEqual(r["status"], "at_limit")
        self.assertEqual(r["answer"], AT_LIMIT)

    def test_spend_cap_returns_friendly_message(self):
        r = ask(Q, client=RaisingClient(FakeError("Your credit balance is too low", 400, "invalid_request_error")))
        self.assertEqual(r["status"], "at_limit")

    def test_overloaded_returns_friendly_message(self):
        r = ask(Q, client=RaisingClient(FakeError("Overloaded", 529, "overloaded_error")))
        self.assertEqual(r["status"], "at_limit")

    def test_other_error_does_not_leak_details(self):
        r = ask(Q, client=RaisingClient(FakeError("internal stack trace secret", 500, "api_error")))
        self.assertEqual(r["status"], "error")
        self.assertNotIn("secret", r["answer"])


class AnswerTests(unittest.TestCase):
    def test_valid_citation_is_kept(self):
        hits, _ = answer.retrieve(Q)
        client = ReplyClient({"can_answer": True, "answer": "They quarantined flaky tests.", "cited_chunks": [hits[0]["id"]]})
        r = ask(Q, client=client)
        self.assertEqual(r["status"], "answered")
        self.assertEqual(r["citations"][0]["chunk_id"], hits[0]["id"])
        self.assertEqual(client.calls[0]["model"], answer.MODEL)

    def test_answer_without_valid_citation_becomes_idk(self):
        client = ReplyClient({"can_answer": True, "answer": "Made up.", "cited_chunks": ["Not A Note#1"]})
        self.assertEqual(ask(Q, client=client)["answer"], IDK)

    def test_refusal_becomes_idk(self):
        client = ReplyClient({}, stop_reason="refusal")
        self.assertEqual(ask(Q, client=client)["status"], "idk")

    def test_below_threshold_never_calls_model(self):
        client = ReplyClient({"can_answer": True, "answer": "x", "cited_chunks": []})
        r = ask("Does the candidate have Kubernetes experience?", client=client)
        self.assertEqual(r["status"], "idk")
        self.assertEqual(client.calls, [])

    def test_personal_never_calls_model(self):
        client = ReplyClient({"can_answer": True, "answer": "x", "cited_chunks": []})
        r = ask("What salary is the candidate looking for?", client=client)
        self.assertEqual(r["answer"], PERSONAL)
        self.assertEqual(client.calls, [])

    def test_dry_run_cites_top_note(self):
        r = ask(Q, dry_run=True)
        self.assertEqual(r["status"], "answered")
        self.assertTrue(r["citations"])
        self.assertIn("AI representative", r["disclosure"])


if __name__ == "__main__":
    unittest.main()
