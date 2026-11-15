"""Command line: python -m fitcheck <prep|ingest|ask|eval|serve|demo> ..."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fitcheck")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prep", help="copy notes allowlisted in private/manifest.json from a vault into private/notes/")
    s.add_argument("--vault", default=os.environ.get("FITCHECK_VAULT"), required=os.environ.get("FITCHECK_VAULT") is None)

    sub.add_parser("ingest", help="split the notes by heading and build build/index.json")

    sub.add_parser("check", help="check that ANTHROPIC_API_KEY, the model and the anthropic package work for live mode")

    s = sub.add_parser("ask", help="ask one question")
    s.add_argument("question")
    s.add_argument("--json", action="store_true")
    s.add_argument("--dry-run", action="store_true", help="never call the API")

    s = sub.add_parser("eval", help="run the eval set")
    s.add_argument("--generate", action="store_true", help="also run answer generation (uses ANTHROPIC_API_KEY)")
    s.add_argument("--questions", default=os.environ.get("FITCHECK_EVALS"), help="eval file (default: evals/questions.json)")

    s = sub.add_parser("serve", help="run the candidate agent (A2A) locally")
    s.add_argument("--port", type=int, default=8787)

    s = sub.add_parser("demo", help="a recruiter agent interviews the candidate agent over A2A (opens a local page)")
    s.add_argument("--url", default=None, help="a candidate agent A2A base URL to use (default: start one locally)")
    s.add_argument("--port", type=int, default=8788, help="port for the demo page")
    s.add_argument("--agent-port", type=int, default=8787)
    s.add_argument("--fast", action="store_true", help="no pauses between steps")
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--cli", choices=["fit", "notfit", "both"], help="print the interview in the terminal instead")

    a = p.parse_args(argv)

    if a.cmd == "prep":
        from .prep import prep

        prep(Path(a.vault).expanduser())
    elif a.cmd == "ingest":
        from .retrieval import build_index

        print(json.dumps(build_index(), indent=2))
    elif a.cmd == "ask":
        from .answer import ask

        r = ask(a.question, dry_run=True if a.dry_run else None)
        if a.json:
            print(json.dumps(r, indent=2, ensure_ascii=False))
        else:
            print(r["answer"])
            for c in r["citations"]:
                print(f"  source: {c['note']} > {c['section']}")
    elif a.cmd == "check":
        from . import llm

        ok, line = llm.key_check()
        print(line)
        return 0 if ok else 1
    elif a.cmd == "eval":
        from evals.run_evals import run

        return run(generate=a.generate, questions=a.questions)
    elif a.cmd == "serve":
        from .a2a import serve

        serve(port=a.port)
    elif a.cmd == "demo":
        from demo.server import run_demo

        jobs = None if not a.cli else (["fit", "notfit"] if a.cli == "both" else [a.cli])
        return run_demo(url=a.url, port=a.port, agent_port=a.agent_port, fast=a.fast,
                        open_browser=not a.no_browser, cli_jobs=jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
