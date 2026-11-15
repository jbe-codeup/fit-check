"""Run the fitcheck pipeline over the fit-verdict dataset as a LangSmith experiment.

    make eval-experiment                                   # Haiku, live if ANTHROPIC_API_KEY is set
    make eval-experiment ARGS="--model claude-sonnet-5-5"  # same dataset, another model
    make eval-experiment ARGS="--recruiter-prompt prompts/strict.txt --prefix strict-prompt"
    make eval-fit                                          # --dry-run: local, no keys, mock mode

Every run is one experiment on the same dataset, so LangSmith can put runs side by side
(Datasets & Experiments > the dataset > select experiments > Compare).

Modes: mock = scripted questions, quoted answers and the rule verdict (no Anthropic calls);
live = Claude writes the questions, the answers and the verdict. auto picks live when
ANTHROPIC_API_KEY is set.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from evals import fit_pipeline as fp


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=fp.ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except OSError:
        return None


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m evals.langsmith_experiment", description=__doc__.splitlines()[0])
    p.add_argument("--model", default=os.environ.get("FITCHECK_MODEL", "claude-haiku-5-5"),
                   help="model for both agents (default: FITCHECK_MODEL or claude-haiku-5-5)")
    p.add_argument("--mode", choices=["auto", "mock", "live"], default="auto")
    p.add_argument("--recruiter-prompt", help="file with a replacement recruiter system prompt, to compare prompt variants")
    p.add_argument("--prefix", help="experiment name prefix (default: fitcheck-<mode>-<model>[-<prompt file>])")
    p.add_argument("--judge", action="store_true", help="add the LLM judge of reasoning quality (Anthropic calls)")
    p.add_argument("--judge-model", default=fp.JUDGE_MODEL)
    p.add_argument("--split", choices=["clear", "borderline"], help="only clear-cut or only borderline pairs")
    p.add_argument("--limit", type=int, help="only the first N pairs")
    p.add_argument("--repetitions", type=int, default=1, help="run each pair N times to see variance")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--dataset", help="LangSmith dataset name (default: FITCHECK_LS_DATASET or the name in pairs.json)")
    p.add_argument("--dry-run", action="store_true", help="run locally and print scores; no LangSmith")
    a = p.parse_args(argv)

    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    live = a.mode == "live" or (a.mode == "auto" and has_key)
    if live and not has_key:
        print("--mode live needs ANTHROPIC_API_KEY. Fill it in .env (see .env.example) and export it.")
        return 2
    if a.judge and not has_key:
        print("--judge needs ANTHROPIC_API_KEY.")
        return 2
    if not a.dry_run and not os.environ.get("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY is not set. Fill it in .env (see .env.example) and export it, or use --dry-run.")
        return 2
    prompt = Path(a.recruiter_prompt).read_text(encoding="utf-8") if a.recruiter_prompt else None
    mode = "live" if live else "mock"
    prefix = a.prefix or "-".join(filter(None, ["fitcheck", mode, a.model if live else None,
                                                _slug(Path(a.recruiter_prompt).stem) if prompt else None]))

    data = fp.load()
    examples = [e for e in data["examples"] if not a.split or e["split"] == a.split][: a.limit]
    judge = fp.make_reasoning_judge(a.judge_model) if a.judge else None
    print(f"{prefix}: {len(examples)} pairs, mode {mode}" + (f", model {a.model}" if live else "")
          + (", LLM judge on" if judge else ""))

    with fp.AgentPool(live=live, model=a.model) as pool:
        target = fp.make_target(pool, live=live, model=a.model, recruiter_prompt=prompt)
        if a.dry_run:
            fp.run_local(examples, target, judge)
            return 0

        from langsmith import Client

        client = Client()
        name = a.dataset or os.environ.get("FITCHECK_LS_DATASET") or data["name"]
        if not client.has_dataset(dataset_name=name):
            print(f"No LangSmith dataset {name!r} yet. Run `make eval-dataset` first.")
            return 2
        wanted = {e["id"] for e in examples}
        rows = [e for e in client.list_examples(dataset_name=name) if str(e.id) in wanted]
        evaluators = [fp.verdict_correct, fp.citation_grounding] + ([judge] if judge else [])
        results = client.evaluate(
            target,
            data=rows,
            evaluators=evaluators,
            summary_evaluators=[fp.verdict_metrics],
            experiment_prefix=prefix,
            description=f"fitcheck fit verdicts, {mode} mode" + (f", {a.model}" if live else ""),
            metadata={"mode": mode, "model": a.model if live else None, "recruiter_prompt": a.recruiter_prompt or "default",
                      "dataset_version": data["version"], "git_sha": _git_sha(), "split": a.split or "all"},
            max_concurrency=a.concurrency,
            num_repetitions=a.repetitions,
        )
        print(f"experiment: {results.experiment_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
