"""Ingest + retrieval. Standard library only.

Notes are split by heading into chunks, each chunk is turned into a sparse
TF-IDF vector (a local "embedding" that needs no model download and no API
key), and everything is written to one JSON index file that ships with the
function. Anthropic has no embeddings endpoint and it is the only hosted AI
provider this project uses, so retrieval stays local.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import INDEX, NOTES_DIR

MAX_CHUNK_CHARS = 1800

STOPWORDS = set(
    """a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have having
    he her here hers him himself his how i if in into is it its itself just me more most my myself no nor not
    now of off on once only or other our ours out over own same she should so some such than that the their
    theirs them then there these they this those through to too under until up very was we were what when where
    which while who whom why will with would you your yours tell give describe explain talk example share
    candidate candidates anything something time ever also any much many
    experience experienced know knows known work worked working use used using written write wrote shipped
    production professional good great well really his him he's has have had done
    """.split()
)

# Small query-side synonym map so recruiter phrasing lands on note phrasing.
SYNONYMS = {
    "accomplishment": ["impact", "proudest", "result"],
    "achievement": ["impact", "proudest", "result"],
    "proud": ["proudest", "impact"],
    "mistake": ["failure", "learn"],
    "fail": ["failure", "mistake"],
    "weakness": ["failure", "gap"],
    "conflict": ["friction", "skeptic", "disagree"],
    "disagreement": ["disagree", "convince", "skeptic"],
    "golang": ["go"],
    "k8s": ["kubernetes"],
    "llm": ["ai", "mcp", "rag"],
    "genai": ["ai", "rag", "llm"],
    "ci": ["pipeline", "github", "actions", "jenkins"],
    "cd": ["pipeline", "deploy"],
    "cicd": ["pipeline", "github", "actions", "jenkins"],
    "lead": ["leadership", "led", "owned"],
    "leadership": ["led", "owned", "lead"],
    "mentor": ["mentoring", "intern"],
    "school": ["education", "university", "degree"],
    "college": ["education", "university", "degree"],
    "degree": ["education", "university"],
    "study": ["education", "university", "degree"],
    "studi": ["education", "university", "degree"],
    "migrate": ["migration", "cutover", "move"],
    "migrat": ["migration", "cutover", "move"],
    "cost": ["cost", "saving", "cheaper"],
    "cheaper": ["cost"],
    "scale": ["throughput", "scale", "scaled", "requests", "peak"],
    "scal": ["scale", "throughput", "peak"],  # "scaled" / "scaling" stem to "scal"
    "system": ["service", "platform"],
    "retry": ["retries", "idempotent", "duplicate", "twice"],
    "idempotency": ["idempotent", "duplicate", "twice"],
    "monitor": ["metrics", "alarms", "dashboard", "traces", "logs"],
    "observability": ["metrics", "alarms", "dashboard", "traces", "logs"],
    "alert": ["alarms", "page", "paging"],
    "coach": ["mentor", "mentored", "pairing"],
    "incident": ["outage", "postmortem", "incident"],
    "outage": ["incident", "postmortem"],
    "own": ["owned", "ownership"],
    "ownership": ["owned", "own"],
    "test": ["testing", "tests"],
    "language": ["go", "python", "ruby", "java", "csharp", "javascript", "typescript", "elixir", "skills"],
    "languages": ["go", "python", "ruby", "java", "csharp", "javascript", "typescript", "elixir", "skills"],
    "stack": ["skills", "go", "python", "ruby", "aws"],
    "tech": ["skills"],
    "skill": ["skills"],
    "iac": ["terraform"],
    "evaluat": ["eval", "evaluation"],
    "evaluate": ["eval", "evaluation"],
    "evaluation": ["eval"],
    "infrastructure": ["terraform", "aws"],
    "debug": ["debugged", "traced", "cause", "x-ray"],
    "debugg": ["debug", "traced", "cause", "x-ray"],  # "debugged" / "debugging" stem to "debugg"
    "troubleshoot": ["debug", "traced", "cause"],
    "led": ["lead", "leadership"],
    "on-call": ["rotation", "pages", "runbook"],
}


def stem(tok: str) -> str:
    """Tiny suffix stripper, consistent for singular/plural ("language"/"languages")."""
    if len(tok) <= 4:
        return tok
    if tok.endswith(("ies", "ied")):
        return tok[:-3] + "y"
    if tok.endswith(("sses", "xes", "zes", "ches", "shes")):
        return tok[:-2]
    for suf in ("ing", "ed", "s"):
        if tok.endswith(suf) and not tok.endswith("ss") and len(tok) > len(suf) + 3:
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("c#", "csharp").replace("node.js", "nodejs")
    raw = re.findall(r"[a-z0-9]+(?:[-+][a-z0-9]+)*", text)
    return [stem(t) for t in raw if t not in STOPWORDS and len(t) > 1]


def expand(tokens: list[str]) -> list[str]:
    out = list(tokens)
    for t in tokens:
        out += [stem(s) for s in SYNONYMS.get(t, [])]
    return out


@dataclass
class Chunk:
    id: str
    note: str
    heading: str
    text: str


def split_note(name: str, text: str) -> list[Chunk]:
    """Split one note by heading; carry the heading path into each chunk."""
    chunks: list[Chunk] = []
    path: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if len(re.sub(r"[\s#>*_-]", "", body)) < 25:
            return
        heading = " > ".join(h for _, h in path) or name
        # Keep chunks small enough to cite precisely.
        parts, cur = [], ""
        for para in re.split(r"\n\s*\n", body):
            if cur and len(cur) + len(para) > MAX_CHUNK_CHARS:
                parts.append(cur)
                cur = para
            else:
                cur = f"{cur}\n\n{para}" if cur else para
        if cur:
            parts.append(cur)
        for p in parts:
            chunks.append(Chunk(id=f"{name}#{len(chunks) + 1}", note=name, heading=heading, text=p.strip()))

    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            buf = []
            level = len(m.group(1))
            path = [(lv, h) for lv, h in path if lv < level] + [(level, m.group(2).strip())]
        else:
            buf.append(line)
    flush()
    return chunks


def _notes(corpus_dir: Path) -> list[Path]:
    return sorted(f for f in corpus_dir.glob("*.md") if f.name.lower() != "readme.md")


def ensure_index(corpus_dir: Path = NOTES_DIR, out: Path = INDEX) -> None:
    """Rebuild the index when it's missing, built from another folder, or older than a note."""
    if out.exists():
        try:
            meta = json.loads(out.read_text(encoding="utf-8")).get("source")
        except (OSError, json.JSONDecodeError):
            meta = None
        newest = max((f.stat().st_mtime for f in _notes(corpus_dir)), default=0)
        if meta == str(corpus_dir.resolve()) and out.stat().st_mtime >= newest:
            return
    build_index(corpus_dir, out)


def build_index(corpus_dir: Path = NOTES_DIR, out: Path = INDEX) -> dict:
    chunks: list[Chunk] = []
    for f in _notes(corpus_dir):
        chunks += split_note(f.stem, f.read_text(encoding="utf-8"))
    if not chunks:
        raise SystemExit(f"No notes found in {corpus_dir}. Set FITCHECK_NOTES to a folder of Markdown notes.")

    docs = [tokenize(f"{c.note} {c.heading} {c.heading} {c.text}") for c in chunks]
    df = Counter(t for d in docs for t in set(d))
    n = len(docs)
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}

    entries = []
    for c, toks in zip(chunks, docs):
        vec = _vector(Counter(toks), idf)
        entries.append({"id": c.id, "note": c.note, "heading": c.heading, "text": c.text, "vec": vec})

    index = {"version": 1, "source": str(corpus_dir.resolve()), "idf": idf, "chunks": entries}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    return {"chunks": n, "notes": len({c.note for c in chunks}), "vocab": len(idf), "path": str(out)}


def _vector(tf: Counter, idf: dict) -> dict:
    vec = {t: (1 + math.log(c)) * idf[t] for t, c in tf.items() if t in idf}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {t: round(v / norm, 5) for t, v in vec.items()}


class Retriever:
    def __init__(self, index_path: Path = INDEX, ignore: str = ""):
        data = json.loads(Path(index_path).read_text(encoding="utf-8"))
        self.idf: dict = data["idf"]
        self.chunks: list[dict] = data["chunks"]
        # The candidate's own name says nothing about the topic of a question.
        self.ignore = {stem(t) for t in re.findall(r"[a-z0-9]+", ignore.lower())}
        self.ignore |= {t + "s" for t in self.ignore}

    def search(self, question: str, k: int = 5) -> list[dict]:
        toks = [t for t in expand(tokenize(question)) if t not in self.ignore]
        unseen = math.log(len(self.chunks) + 1) + 1.0
        qvec = _vector(Counter(toks), {**{t: unseen for t in toks}, **self.idf})
        scored = []
        for c in self.chunks:
            cv = c["vec"]
            s = sum(w * cv.get(t, 0.0) for t, w in qvec.items())
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{**c, "score": round(s, 4)} for s, c in scored[:k]]

    def known_terms(self, question: str) -> tuple[list[str], list[str]]:
        toks = tokenize(question)
        return [t for t in toks if t in self.idf], [t for t in toks if t not in self.idf]
