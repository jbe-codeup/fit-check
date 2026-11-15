"""Offline tests for the fit-verdict evals (evals/fit_pipeline.py). No keys, no network, no LangSmith."""
import json
import os
import unittest
from types import SimpleNamespace

from evals import fit_pipeline as fp
from evals import langsmith_dataset, langsmith_experiment


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.data = fp.load()

    def test_pairs_are_complete_and_unique(self):
        ex = self.data["examples"]
        self.assertGreaterEqual(len(ex), 20)
        self.assertEqual(len({e["id"] for e in ex}), len(ex))
        verdicts = {e["outputs"]["verdict"] for e in ex}
        self.assertEqual(verdicts, {"Fit", "Not a fit"})
        self.assertTrue(any(e["metadata"]["borderline"] for e in ex))
        for e in ex:
            self.assertTrue((fp.ROOT / e["inputs"]["notes_dir"]).is_dir(), e["key"])
            self.assertTrue(e["inputs"]["job"]["questions"], e["key"])
            self.assertTrue(e["outputs"]["rationale"], e["key"])

    def test_example_ids_are_stable(self):
        self.assertEqual([e["id"] for e in fp.load()["examples"]], [e["id"] for e in self.data["examples"]])

    def test_every_candidate_is_marked_fictional(self):
        for notes_dir in self.data["candidates"].values():
            card = json.loads((fp.ROOT / notes_dir / "candidate.json").read_text(encoding="utf-8"))
            self.assertTrue(card.get("fictional"), notes_dir)


class EvaluatorTests(unittest.TestCase):
    inputs = {"notes_dir": "sample_notes"}

    def test_verdict_correct(self):
        self.assertEqual(fp.verdict_correct({"verdict": "Fit"}, {"verdict": "Fit"})["score"], 1)
        self.assertEqual(fp.verdict_correct({"verdict": "Fit"}, {"verdict": "Not a fit"})["score"], 0)

    def test_grounding_flags_missing_and_unknown_citations(self):
        good = {"chunk_id": "Deploy Reliability#1"}
        out = {"turns": [
            {"requirement": "a", "status": "answered", "citations": [good]},
            {"requirement": "b", "status": "answered", "citations": []},
            {"requirement": "c", "status": "answered", "citations": [{"chunk_id": "Made Up Note#1"}]},
            {"requirement": "d", "status": "idk", "citations": []},
        ]}
        r = fp.citation_grounding(self.inputs, out)
        self.assertAlmostEqual(r["score"], 0.333)
        self.assertIn("b", r["comment"])
        self.assertIsNone(fp.citation_grounding(self.inputs, {"turns": []})["score"])

    def test_summary_metrics(self):
        outs = [{"verdict": "Fit"}, {"verdict": "Fit"}, {"verdict": "Not a fit"}, {"verdict": "Not a fit"}]
        refs = [{"verdict": "Fit"}, {"verdict": "Not a fit"}, {"verdict": "Not a fit"}, {"verdict": "Fit"}]
        m = {r["key"]: r["score"] for r in fp.verdict_metrics(outs, refs)["results"]}
        self.assertEqual(m, {"accuracy": 0.5, "false_fit_rate": 0.5, "missed_fit_rate": 0.5})

    def test_judge_with_fake_client(self):
        calls = []

        def create(**kw):
            calls.append(kw)
            return SimpleNamespace(stop_reason="end_turn",
                                   content=[SimpleNamespace(type="text", text='{"score": 5, "reason": "ok"}')])

        judge = fp.make_reasoning_judge("test-model", client=SimpleNamespace(messages=SimpleNamespace(create=create)))
        r = judge({"job": {"description": "d"}}, {"verdict": "Fit", "summary": "s", "turns": []},
                  {"verdict": "Fit", "rationale": "r"})
        self.assertEqual(r["score"], 1.0)
        self.assertEqual(calls[0]["model"], "test-model")


class MockPipelineTests(unittest.TestCase):
    """Runs real interviews over A2A in mock mode on a few pairs from each candidate."""

    def test_mock_run_scores(self):
        ex = [e for e in fp.load()["examples"]
              if e["key"] in ("sample-candidate--principal-backend-go", "jordan-lee--senior-ios", "casey-nguyen--senior-ios")]
        self.assertEqual(len(ex), 3)
        with fp.AgentPool(live=False, model="test-model") as pool:
            target = fp.make_target(pool, live=False, model="test-model")
            for e in ex:
                out = target(e["inputs"])
                self.assertEqual(fp.verdict_correct(out, e["outputs"])["score"], 1, e["key"])
                g = fp.citation_grounding(e["inputs"], out)["score"]
                self.assertIn(g, (None, 1.0), e["key"])


class CliTests(unittest.TestCase):
    def test_dataset_dry_run_needs_no_key(self):
        self.assertEqual(langsmith_dataset.main(["--dry-run"]), 0)

    def test_experiment_without_langsmith_key_stops_early(self):
        saved = os.environ.pop("LANGSMITH_API_KEY", None)
        try:
            self.assertEqual(langsmith_experiment.main(["--mode", "mock"]), 2)
        finally:
            if saved is not None:
                os.environ["LANGSMITH_API_KEY"] = saved


if __name__ == "__main__":
    unittest.main()
