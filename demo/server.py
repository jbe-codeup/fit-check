"""Local demo: starts the candidate agent (A2A) as its own process, then serves the
interview page. The page streams each step (question, A2A request, answer,
citations, verdict) from /api/run over Server-Sent Events.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from fitcheck import config

from . import recruiter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Live runs are saved here so a good take can be replayed for a screen recording
# without spending tokens: open the page with ?replay=1. Contains note text, so it's gitignored.
RECORDINGS = config.BUILD / "recordings"
CANDIDATE = config.candidate()
SLUG = "".join(ch if ch.isalnum() else "-" for ch in CANDIDATE["name"].lower()).strip("-") or "candidate"

DELAYS = {"card": 0.8, "plan": 0.6, "question": 1.0, "thinking": 0.7, "answer": 0.9}


def _replay(events: list[dict], pace: float):
    for ev in events:
        yield {**ev, "replay": True} if ev["type"] == "start" else ev
        if pace:
            time.sleep(DELAYS.get(ev["type"], 0.3) * pace)


def start_agent(port: int) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-m", "fitcheck", "serve", "--port", str(port)], cwd=ROOT)
    url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            urllib.request.urlopen(url + "/health", timeout=1)
            return proc
        except OSError:
            time.sleep(0.2)
    proc.terminate()
    raise SystemExit(f"Candidate agent did not start on port {port}")


def make_handler(agent_url: str, pace: float):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self._send(200, (HERE / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/jobs":
                jobs = {k: {f: v[f] for f in ("label", "title", "company", "sample_note", "description")}
                        for k, v in recruiter.JOBS.items()}
                live = bool(os.environ.get("ANTHROPIC_API_KEY"))
                body = {"jobs": jobs, "mode": f"Live: Claude {recruiter.MODEL}" if live else "No key: answers quote the notes", "agent": agent_url,
                        "candidate": {k: CANDIDATE[k] for k in ("name", "first", "role", "fictional")}}
                return self._send(200, json.dumps(body).encode(), "application/json")
            if u.path == "/api/run":
                qs = parse_qs(u.query)
                job = (qs.get("job") or ["fit"])[0]
                replay = (qs.get("replay") or ["0"])[0] == "1"
                if job not in recruiter.JOBS:
                    return self._send(404, b'{"error":"unknown job"}', "application/json")
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.end_headers()
                rec_path = RECORDINGS / f"{SLUG}-{job}.json"
                live = bool(os.environ.get("ANTHROPIC_API_KEY"))
                try:
                    if replay and rec_path.exists():
                        events = _replay(json.loads(rec_path.read_text(encoding="utf-8")), pace)
                    else:
                        events = recruiter.interview(job, agent_url, pace=pace)
                    recorded = []
                    for ev in events:
                        recorded.append(ev)
                        self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
                        self.wfile.flush()
                    if live and not replay and recorded and recorded[-1]["type"] == "done":
                        RECORDINGS.mkdir(parents=True, exist_ok=True)
                        rec_path.write_text(json.dumps(recorded, ensure_ascii=False), encoding="utf-8")
                except (BrokenPipeError, ConnectionResetError):
                    return
                except Exception as e:  # show it on the page instead of a silent stall
                    msg = json.dumps({"type": "error", "message": f"{type(e).__name__}: {e}"})
                    self.wfile.write(f"data: {msg}\n\n".encode())
                return
            self._send(404, b"not found", "text/plain")

        def log_message(self, *a):
            pass

    return Handler


def run_demo(url: str | None = None, port: int = 8788, agent_port: int = 8787, fast: bool = False,
             open_browser: bool = True, cli_jobs: list[str] | None = None) -> int:
    proc = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        from fitcheck import llm

        ok, line = llm.key_check()
        print(line + ("" if ok else "\n  Live answers will fail until this is fixed. Unset ANTHROPIC_API_KEY to run the no-key demo."),
              file=sys.stderr if not ok else sys.stdout, flush=True)
    if not url:
        proc = start_agent(agent_port)
        url = f"http://127.0.0.1:{agent_port}"
    try:
        if cli_jobs:
            return recruiter.run_cli(cli_jobs, url, fast=fast)
        srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(url, 0 if fast else 1.0))
        page = f"http://127.0.0.1:{port}"
        mode = f"live, Claude {recruiter.MODEL}" if os.environ.get("ANTHROPIC_API_KEY") else "mock, no API key"
        print(f"\nfitcheck demo ({mode}), candidate: {CANDIDATE['name']}\n  page:             {page}\n  candidate agent:  {url}  (A2A)\nCtrl+C to stop.")
        if open_browser:
            webbrowser.open(page)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    finally:
        if proc:
            proc.terminate()
