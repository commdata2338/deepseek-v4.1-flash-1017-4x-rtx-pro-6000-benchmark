"""Streaming, fixed-destination loopback relay. Never rewrites model requests."""
import argparse
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import threading
import time

from sandbox import GATEWAY, PORT, verify


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.proxy()

    def do_POST(self):
        self.proxy()

    def proxy(self):
        allowed = (self.command, self.path) in {
            ("GET", "/v1/models"), ("POST", "/v1/chat/completions")}
        if not allowed:
            self.send_error(403, "Only model discovery and chat completions are allowed")
            return
        if self.headers.get("Transfer-Encoding"):
            self.send_error(400, "Chunked request bodies are unsupported")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 32 * 1024 * 1024:
                raise ValueError()
            self.connection.settimeout(30)
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError()
            parsed = json.loads(body) if body else {}
        except (ValueError, OSError):
            self.send_error(400, "Invalid request body")
            return
        audit = {"time": time.time(), "method": self.command, "path": self.path,
                 "body_sha256": hashlib.sha256(body).hexdigest(),
                 "request": {k: parsed[k] for k in (
                     "model", "temperature", "reasoning_effort", "max_tokens", "stream",
                     "stream_options", "chat_template_kwargs") if k in parsed}}
        start = time.monotonic()
        conn = http.client.HTTPConnection("127.0.0.1", self.server.engine_port, timeout=1200)
        sent = False
        try:
            headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
            if self.headers.get("Authorization"):
                headers["Authorization"] = self.headers["Authorization"]
            conn.request(self.command, self.path, body=body or None, headers=headers)
            upstream = conn.getresponse()
            audit["status"] = upstream.status
            self.send_response(upstream.status)
            self.send_header("Content-Type", upstream.getheader("Content-Type", "application/json"))
            self.send_header("Connection", "close")
            self.end_headers()
            sent = True
            self.connection.settimeout(30)
            while chunk := upstream.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException) as exc:
            audit["error"] = type(exc).__name__
            if not sent:
                self.send_error(502, "Model upstream unavailable")
        finally:
            conn.close()
            audit["seconds"] = time.monotonic() - start
            with self.server.audit_lock:
                self.server.audit.write(json.dumps(audit) + "\n")
                self.server.audit.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine-port", required=True, type=int)
    p.add_argument("--log", required=True, type=Path)
    args = p.parse_args()
    if not 1 <= args.engine_port <= 65535 or args.engine_port == PORT:
        p.error("invalid engine port")
    verify()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", buffering=1) as log, ThreadingHTTPServer((GATEWAY, PORT), Handler) as server:
        server.engine_port = args.engine_port
        server.audit = log
        server.audit_lock = threading.Lock()
        def stop(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        print(f"Relay ready on {GATEWAY}:{PORT}; upstream loopback:{args.engine_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Relay stopped", flush=True)


if __name__ == "__main__":
    main()
