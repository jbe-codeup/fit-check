"""The candidate agent as an A2A agent (Agent2Agent protocol, JSON-RPC transport).

  GET  /.well-known/agent-card.json   the agent card other agents discover
  POST /                               JSON-RPC 2.0, method "message/send"

The reply is an A2A Message with a text part (the answer) and a data part
(status, citations, disclosure), so a calling agent gets sources it can show.

Standard library only. `serve()` runs it locally; `lambda_handler` is the same
logic for a serverless function (for example an Amplify / Lambda function URL),
with the index file shipped next to the code and ANTHROPIC_API_KEY set as a
server-side secret.

Guards, all optional and set by env var:
  FITCHECK_A2A_TOKEN   if set, callers must send "Authorization: Bearer <token>"
  FITCHECK_RATE_LIMIT  requests per minute per caller (default 30)
  FITCHECK_LOG         path of a JSON-lines log of questions and statuses (default build/questions.log)
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import answer, config

VERSION = "0.1.0"
LOG_PATH = Path(os.environ.get("FITCHECK_LOG", config.BUILD / "questions.log"))
RATE_LIMIT = int(os.environ.get("FITCHECK_RATE_LIMIT", "30"))
MAX_BODY = 16_000
C = config.candidate()


def agent_card(base_url: str) -> dict:
    card = {
        "protocolVersion": "0.3.0",
        "name": f"{C['name']} (candidate agent)",
        "description": (
            f"AI representative of {C['name']}{', ' + C['role'] if C['role'] else ''}. "
            f"Answers recruiter questions only from {C['pos']} own career notes, cites the note it used, "
            "and says \"I don't know\" when the notes don't cover something."
            + (" This is a fictional sample candidate." if C["fictional"] else "")
        ),
        "url": base_url.rstrip("/") + "/",
        "preferredTransport": "JSONRPC",
        "version": VERSION,
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "skills": [
            {
                "id": "career-qa",
                "name": "Career questions",
                "description": f"Experience, projects and skills, answered with citations to {C['first']}'s notes.",
                "tags": ["recruiting", "career", "backend", "citations"],
                "examples": [
                    f"What has {C['first']} built with Go and AWS?",
                    f"Tell me about a production reliability problem {C['subj']} fixed.",
                    f"Has {C['first']} mentored other engineers?",
                ],
            }
        ],
    }
    if os.environ.get("FITCHECK_A2A_TOKEN"):
        card["securitySchemes"] = {"bearer": {"type": "http", "scheme": "bearer"}}
        card["security"] = [{"bearer": []}]
    return card


def _text_of(message: dict) -> str:
    parts = message.get("parts") or []
    return " ".join(p.get("text", "") for p in parts if p.get("kind", p.get("type")) == "text").strip()


def _rpc_error(rid, code: int, msg: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}


def handle_rpc(req: dict, caller: str = "local") -> dict:
    """One JSON-RPC request in, one JSON-RPC response out."""
    rid = req.get("id")
    if req.get("jsonrpc") != "2.0" or "method" not in req:
        return _rpc_error(rid, -32600, "Invalid Request")
    if req["method"] != "message/send":
        return _rpc_error(rid, -32601, f"Method not found: {req['method']}")
    message = (req.get("params") or {}).get("message") or {}
    question = _text_of(message)
    if not question:
        return _rpc_error(rid, -32602, "params.message needs a text part")

    started = time.time()
    r = answer.ask(question)
    _log({"t": round(started), "caller": caller, "q": question[:500], "status": r["status"],
          "ms": int((time.time() - started) * 1000)})

    text = r["answer"]
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "result": {
            "kind": "message",
            "role": "agent",
            "messageId": str(uuid.uuid4()),
            "contextId": message.get("contextId") or str(uuid.uuid4()),
            "parts": [
                {"kind": "text", "text": text},
                {
                    "kind": "data",
                    "data": {
                        "status": r["status"],
                        "citations": r["citations"],
                        "disclosure": r["disclosure"],
                        "mode": r.get("mode") or ("claude" if r.get("model") else "rule"),
                        "model": r.get("model"),
                    },
                },
            ],
        },
    }


_log_lock = threading.Lock()


def _log(entry: dict) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _log_lock, LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


class _RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True


_limiter = _RateLimiter(RATE_LIMIT)


def _authorized(headers) -> bool:
    token = os.environ.get("FITCHECK_A2A_TOKEN")
    return not token or headers.get("authorization", headers.get("Authorization", "")) == f"Bearer {token}"


def _dispatch(method: str, path: str, headers, body: bytes, caller: str, base_url: str) -> tuple[int, dict]:
    if method == "GET" and path in ("/.well-known/agent-card.json", "/.well-known/agent.json"):
        return 200, agent_card(base_url)
    if method == "GET" and path in ("/", "/health"):
        return 200, {"ok": True, "agent": C["name"], "card": "/.well-known/agent-card.json"}
    if method != "POST" or path not in ("/", "/a2a"):
        return 404, {"error": "not found"}
    if not _authorized(headers):
        return 401, _rpc_error(None, -32001, "Unauthorized")
    if not _limiter.allow(caller):
        return 429, _rpc_error(None, -32002, "Too many requests, try again in a minute")
    if len(body) > MAX_BODY:
        return 413, _rpc_error(None, -32600, "Request too large")
    try:
        req = json.loads(body or b"{}")
    except json.JSONDecodeError:
        return 400, _rpc_error(None, -32700, "Parse error")
    return 200, handle_rpc(req, caller)


class _Handler(BaseHTTPRequestHandler):
    server_version = "fitcheck/" + VERSION

    def _go(self, method: str):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(min(n, MAX_BODY + 1)) if n else b""
        host = self.headers.get("host") or f"127.0.0.1:{self.server.server_port}"
        code, payload = _dispatch(method, self.path.split("?")[0], self.headers, body, self.client_address[0], f"http://{host}")
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._go("GET")

    def do_POST(self):
        self._go("POST")

    def log_message(self, fmt, *args):  # quiet; questions go to the JSON-lines log instead
        if os.environ.get("FITCHECK_VERBOSE"):
            super().log_message(fmt, *args)


def make_server(port: int = 8787, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    answer.retriever()  # load the index before the first request
    return ThreadingHTTPServer((host, port), _Handler)


def serve(port: int = 8787, host: str = "127.0.0.1") -> None:
    srv = make_server(port, host)
    mode = "Claude " + answer.MODEL if os.environ.get("ANTHROPIC_API_KEY") else "no-key mode (quotes the notes, no model call)"
    print(f"Candidate agent for {C['name']} on http://{host}:{srv.server_port}  [{mode}]")
    print(f"Agent card: http://{host}:{srv.server_port}/.well-known/agent-card.json")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def lambda_handler(event, context=None):
    """API Gateway / Lambda function URL entry point (payload format 2.0)."""
    import base64

    http = (event.get("requestContext") or {}).get("http") or {}
    method = http.get("method") or event.get("httpMethod", "GET")
    path = event.get("rawPath") or event.get("path") or "/"
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    body = event.get("body") or ""
    body = base64.b64decode(body) if event.get("isBase64Encoded") else body.encode()
    caller = http.get("sourceIp") or "unknown"
    base = "https://" + (headers.get("host") or (event.get("requestContext") or {}).get("domainName", "localhost"))
    code, payload = _dispatch(method, path, headers, body, caller, base)
    return {"statusCode": code, "headers": {"content-type": "application/json"}, "body": json.dumps(payload, ensure_ascii=False)}
