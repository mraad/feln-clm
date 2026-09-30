"""Stdlib HTTP server for the single-page app: GET / (the page), GET /api/info, POST /api/ask."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).with_name("static") / "index.html"
EXAMPLES = [
    "Find gas/condensate wells.",
    "Show oil discoveries within 5 kilometers of injection pipelines with a diameter between 8.0 and 26.0 inches.",
    "List gas pipelines where the current phase is either 'IN SERVICE' or 'DECOMMISSIONED'.",
    "Which dry wells completed between 1982 and 2000 lie within gas discoveries?",
    "Get condensate discoveries in Denmark more than 10 miles from all oil wells.",
]
MAX_TEXT = 2000
IDLE = 5  # seconds a connection may sit without sending, so shutdown never waits on it


def serve(translator, port: int) -> None:
    lock = threading.Lock()  # one encoder, one request at a time
    info = {
        "catalog_sha": translator.cat.sha,
        "base": translator.cfg["base"],
        "threshold": translator.threshold,
        "layers": {n: {"description": ly.description, "columns": len(ly.columns)} for n, ly in translator.cat.layers.items()},
        "examples": EXAMPLES,
    }  # fmt: skip

    class Handler(BaseHTTPRequestHandler):
        timeout = IDLE  # socket reads/writes; inference itself is not bounded

        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            if self.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._json(200, info)
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/api/ask":
                return self._json(404, {"error": "not found"})
            try:
                size = int(self.headers.get("Content-Length", 0))
                text = str(
                    json.loads(self.rfile.read(min(size, 64 * MAX_TEXT)) or b"{}").get("text", "")
                ).strip()
            except (ValueError, AttributeError):
                return self._json(400, {"error": 'body must be JSON {"text": ...}'})
            if not text or len(text) > MAX_TEXT:
                return self._json(422, {"error": f"text must be 1..{MAX_TEXT} characters"})
            with lock:
                res = translator.ask(text)
            res.pop("probabilities", None)
            self._json(200, res)

        def log_message(self, format, *args):  # noqa: A002
            print(f"[serve] {self.address_string()} {format % args}", flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = False  # server_close() then waits for requests in flight
    print(f"[serve] http://127.0.0.1:{port}/", flush=True)
    try:
        server.serve_forever()
    finally:  # Ctrl-C / SIGTERM / SIGHUP (see cli.main): stop accepting, finish, close
        print("[serve] shutting down: finishing requests in flight", flush=True)
        server.server_close()
