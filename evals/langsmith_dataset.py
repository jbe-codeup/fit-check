"""Upload evals/fit/pairs.json to a LangSmith dataset. Safe to re-run.

    make eval-dataset                      # or: python -m evals.langsmith_dataset
    python -m evals.langsmith_dataset --dry-run   # show what would change, no key needed

Each pair keeps a fixed example id (derived from its pair id), so a re-run only creates new
pairs, updates changed ones and, with --prune, deletes pairs that were removed from the file.
Needs LANGSMITH_API_KEY (and LANGSMITH_ENDPOINT for a self-hosted or EU instance).
"""
from __future__ import annotations

import argparse
import os
import sys

from evals.fit_pipeline import PAIRS, load


def _changed(ex: dict, current) -> bool:
    meta = {k: v for k, v in (current.metadata or {}).items() if k in ex["metadata"]}
    return current.inputs != ex["inputs"] or (current.outputs or {}) != ex["outputs"] or meta != ex["metadata"]


def sync(name: str | None = None, prune: bool = False, dry_run: bool = False) -> int:
    data = load(PAIRS)
    name = name or os.environ.get("FITCHECK_LS_DATASET") or data["name"]
    examples = data["examples"]
    fits = sum(e["outputs"]["verdict"] == "Fit" for e in examples)
    print(f"dataset {name!r} v{data['version']}: {len(examples)} pairs ({fits} Fit, {len(examples) - fits} Not a fit)")
    if dry_run:
        for e in examples:
            print(f"  {e['id']}  {e['key']:<45} {e['outputs']['verdict']}{'  (borderline)' if e['metadata']['borderline'] else ''}")
        print("dry run: nothing uploaded")
        return 0
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY is not set. Fill it in .env (see .env.example) and export it, or use --dry-run.")
        return 2

    from langsmith import Client

    client = Client()
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(name, description=data["description"],
                                        metadata={"source": "evals/fit/pairs.json", "version": data["version"]})
        print(f"created dataset {name!r}")

    current = {str(e.id): e for e in client.list_examples(dataset_id=dataset.id)}
    new = [e for e in examples if e["id"] not in current]
    changed = [e for e in examples if e["id"] in current and _changed(e, current[e["id"]])]
    gone = [i for i in current if i not in {e["id"] for e in examples}]

    if new:
        client.create_examples(dataset_id=dataset.id, examples=[
            {"id": e["id"], "inputs": e["inputs"], "reference_outputs": e["outputs"], "metadata": e["metadata"],
             "split": e["split"]} for e in new])
    if changed:
        client.update_examples(dataset_id=dataset.id, updates=[
            {"id": e["id"], "inputs": e["inputs"], "outputs": e["outputs"], "metadata": e["metadata"],
             "split": e["split"]} for e in changed])
    if gone and prune:
        client.delete_examples(gone)
    print(f"created {len(new)}, updated {len(changed)}, unchanged {len(examples) - len(new) - len(changed)}"
          + (f", deleted {len(gone)}" if gone and prune else f", {len(gone)} extra in LangSmith (use --prune to delete)" if gone else ""))
    print(f"open it in LangSmith: Datasets & Experiments > {name}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m evals.langsmith_dataset", description=__doc__.splitlines()[0])
    p.add_argument("--name", help="dataset name (default: FITCHECK_LS_DATASET or the name in pairs.json)")
    p.add_argument("--prune", action="store_true", help="delete examples that are no longer in pairs.json")
    p.add_argument("--dry-run", action="store_true", help="print the pairs, upload nothing")
    a = p.parse_args(argv)
    return sync(a.name, prune=a.prune, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
