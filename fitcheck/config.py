"""Settings, all overridable by environment variables.

FITCHECK_NOTES points at a folder of Markdown notes about one candidate. It defaults
to the fictional sample candidate in sample_notes/. Point it at your own notes to
run the demo about yourself; keep that folder out of git.

The folder may hold a candidate.json:
  {"name": "...", "pronoun": "he|she|they", "role": "...", "contact": "reach out to ... on LinkedIn",
   "questions_for_role": ["What does on-call look like for this team?"]}
questions_for_role is what the candidate agent asks a recruiter's agent when invited to ask about the role.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(*files: Path) -> None:
    """Load KEY=value lines from .env files into os.environ. Variables already set in
    the shell win, and the first file that sets a key wins. No dependency needed."""
    for f in files:
        if not f.is_file():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            key, value = key.strip(), value.strip()
            if value[:1] in "\"'" and value[-1:] == value[:1] and len(value) > 1:
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            if key and value and key not in os.environ:
                os.environ[key] = value


load_dotenv(ROOT / ".env", ROOT / "demo" / ".env")


def _path(value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


NOTES_DIR = _path(os.environ.get("FITCHECK_NOTES", "sample_notes"))
BUILD = _path(os.environ.get("FITCHECK_BUILD", "build"))
INDEX = _path(os.environ.get("FITCHECK_INDEX", BUILD / "index.json"))

MODEL = os.environ.get("FITCHECK_MODEL", "claude-haiku-5-5")
THRESHOLD = float(os.environ.get("FITCHECK_THRESHOLD", "0.10"))
TOP_K = int(os.environ.get("FITCHECK_TOP_K", "5"))

PRONOUNS = {
    "he": {"subj": "he", "obj": "him", "pos": "his"},
    "she": {"subj": "she", "obj": "her", "pos": "her"},
    "they": {"subj": "they", "obj": "them", "pos": "their"},
}


@lru_cache(maxsize=1)
def candidate() -> dict:
    data = {}
    f = NOTES_DIR / "candidate.json"
    if f.exists():
        data = json.loads(f.read_text(encoding="utf-8"))
    name = os.environ.get("FITCHECK_CANDIDATE") or data.get("name") or "the candidate"
    pronoun = PRONOUNS.get(data.get("pronoun", "they"), PRONOUNS["they"])
    contact = os.environ.get("FITCHECK_CONTACT") or data.get("contact") or f"reach out to {name} directly"
    return {
        "name": name,
        "first": name.split()[0] if name != "the candidate" else name,
        "role": data.get("role", ""),
        "contact": contact,
        "fictional": bool(data.get("fictional")),
        "questions_for_role": data.get("questions_for_role") or ["How big is the team, and what does on-call look like?"],
        **pronoun,
    }
