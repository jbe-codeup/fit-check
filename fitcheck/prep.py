"""Prepare your own notes (optional; the demo ships with fictional sample notes).

Copies ONLY the notes allowlisted in private/manifest.json out of an Obsidian
vault into private/notes/, and strips interview-coaching material on the way:
frontmatter, [Fill]/[Add] placeholders, "Don't say" and similar sections,
and per-note wording fixes listed in the manifest. Nothing outside the
allowlist is ever opened.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from .config import ROOT, _path

# Both live under private/, which is gitignored.
MANIFEST = _path(os.environ.get("FITCHECK_MANIFEST", "private/manifest.json"))
CORPUS = _path(os.environ.get("FITCHECK_PREP_OUT", "private/notes"))

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
PLACEHOLDER = re.compile(r"\(?\s*\*{0,2}\[(?:Fill|Add|Strongly)[^\]]*\]\*{0,2}\s*\)?\.?", re.I)
WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4 :].lstrip("\n")
    return text


def clean(text: str, manifest: dict, title: str) -> str:
    text = strip_frontmatter(text)
    for old, new in manifest.get("replace", []):
        text = text.replace(old, new)
    text = WIKILINK.sub(lambda m: m.group(2) or m.group(1), text)

    drop_sections = [re.compile(p, re.I) for p in manifest.get("drop_sections", [])]
    drop_lines = [re.compile(p, re.I) for p in manifest.get("drop_lines", [])]

    out: list[str] = []
    skip_level = 0  # >0 while inside a dropped section of that heading level
    for line in text.splitlines():
        h = HEADING.match(line)
        if h:
            level, name = len(h.group(1)), h.group(2).strip()
            if skip_level and level > skip_level:
                continue
            skip_level = 0
            if any(p.search(name) for p in drop_sections):
                skip_level = level
                continue
            out.append(line)
            continue
        if skip_level:
            continue
        # Obsidian callouts: keep the text, drop the "[!tip]" marker.
        line = re.sub(r"^>\s*\[![a-z]+\]\s*", "> ", line)
        if any(p.search(line.strip()) for p in drop_lines):
            continue
        had_placeholder = bool(PLACEHOLDER.search(line))
        line = PLACEHOLDER.sub("", line).rstrip()
        if had_placeholder:
            line = re.sub(r"\s+([,.])", r"\1", line)
            line = re.sub(r"\.(?=\S)", ". ", line) if re.search(r"\*\*\.[A-Za-z]", line) else line
            line = re.sub(r":$", ".", line)
        if had_placeholder and re.fullmatch(r"\s*[-*>]?\s*(\*\*)?\s*", line):
            continue  # line was only a placeholder
        out.append(line)

    # Drop headings left with no body (e.g. follow-up questions whose answer was a [Fill]).
    result: list[str] = []
    for i, line in enumerate(out):
        h = HEADING.match(line)
        if h:
            level = len(h.group(1))
            body = []
            for nxt in out[i + 1 :]:
                hn = HEADING.match(nxt)
                if hn and len(hn.group(1)) <= level:
                    break
                body.append(nxt)
            if not any(b.strip() and b.strip() != "---" for b in body):
                continue
        result.append(line)

    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(result)).strip() + "\n"
    if not cleaned.lstrip().startswith("# "):
        cleaned = f"# {title}\n\n{cleaned}"
    return cleaned


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9 ()+.%-]", "", name).strip() + ".md"


def prep(vault: Path) -> list[dict]:
    manifest = load_manifest()
    if CORPUS.exists():
        shutil.rmtree(CORPUS)
    CORPUS.mkdir(parents=True)
    written = []
    for entry in manifest["notes"]:
        src = vault / entry["path"]
        if not src.exists():
            print(f"  MISSING (skipped): {entry['path']}")
            continue
        text = clean(src.read_text(encoding="utf-8"), manifest, entry["name"])
        dst = CORPUS / safe_filename(entry["name"])
        dst.write_text(text, encoding="utf-8")
        written.append({"name": entry["name"], "file": dst.name, "chars": len(text)})
    for entry in manifest.get("manual_notes", []):
        src = ROOT / entry["path"]
        if not src.exists():
            print(f"  MISSING manual note (skipped): {entry['path']}")
            continue
        dst = CORPUS / safe_filename(entry["name"])
        shutil.copyfile(src, dst)
        written.append({"name": entry["name"], "file": dst.name, "chars": dst.stat().st_size})
    if manifest.get("candidate"):
        (CORPUS / "candidate.json").write_text(json.dumps(manifest["candidate"], indent=2), encoding="utf-8")
    for w in written:
        print(f"  + {w['file']}  ({w['chars']} chars)")
    return written
