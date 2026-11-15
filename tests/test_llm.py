"""Offline tests for the live-mode key check and error logs. No key or network needed."""
import contextlib
import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("FITCHECK_NOTES", "sample_notes")

from fitcheck import answer, llm

FAKE_KEY = "sk-ant-api03-THISISNOTAREALKEY1234567890"


class StatusError(Exception):
    def __init__(self, msg, status_code):
        super().__init__(msg)
        self.message = msg
        self.status_code = status_code


class AuthenticationError(StatusError):
    pass


class NotFoundError(StatusError):
    pass


def raising(exc):
    def create(**kwargs):
        raise exc
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def replying(text, stop_reason="end_turn"):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])
    return SimpleNamespace(messages=SimpleNamespace(create=create), calls=calls)


class KeyCheckTests(unittest.TestCase):
    def test_ok_uses_structured_output(self):
        client = replying(json.dumps({"ok": True}))
        ok, line = llm.key_check(client=client, model="claude-haiku-5-5")
        self.assertTrue(ok)
        self.assertIn("OK", line)
        self.assertEqual(client.calls[0]["output_config"]["format"]["type"], "json_schema")

    def test_rejected_key_says_so(self):
        ok, line = llm.key_check(client=raising(AuthenticationError("invalid x-api-key", 401)))
        self.assertFalse(ok)
        self.assertIn("API key was rejected", line)

    def test_unknown_model_says_so(self):
        ok, line = llm.key_check(client=raising(NotFoundError("model: claude-nope", 404)), model="claude-nope")
        self.assertFalse(ok)
        self.assertIn("isn't available", line)
        self.assertIn("FITCHECK_MODEL", line)

    def test_old_sdk_typeerror_says_upgrade(self):
        ok, line = llm.key_check(client=raising(TypeError("create() got an unexpected keyword argument 'output_config'")))
        self.assertFalse(ok)
        self.assertIn("anthropic>=", line)

    def test_old_installed_sdk_is_caught_before_calling(self):
        with mock.patch.object(llm, "sdk_version", return_value=("0.75.0", (0, 75, 0))):
            ok, line = llm.key_check()
        self.assertFalse(ok)
        self.assertIn("0.75.0 is too old", line)


class LoggingTests(unittest.TestCase):
    def test_describe_never_prints_the_key(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": FAKE_KEY}):
            line = llm.describe(AuthenticationError(f"bad key {FAKE_KEY} and sk-ant-other-secret", 401))
        self.assertIn("AuthenticationError 401", line)
        self.assertNotIn("THISISNOTAREALKEY", line)
        self.assertNotIn("other-secret", line)

    def test_answer_error_is_logged_to_stderr(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            r = answer.ask("How did the candidate improve deploy reliability?",
                           client=raising(AuthenticationError("invalid x-api-key", 401)))
        self.assertEqual(r["status"], "error")
        self.assertIn("AuthenticationError 401: invalid x-api-key", err.getvalue())


if __name__ == "__main__":
    unittest.main()
