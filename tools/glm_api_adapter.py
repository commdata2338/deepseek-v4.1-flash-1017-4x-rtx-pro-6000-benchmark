#!/usr/bin/env python3
"""Local adapter so the GLM paper's unchanged test tools can measure the DeepSeek 1017 test server (stdlib).

    python3 -B glm_api_adapter.py [--listen 8021] [--upstream 8010]

Listens on 127.0.0.1 only. For each request it:
  - sends it to the DeepSeek server on 127.0.0.1:<upstream>, with the server's API key added (read from
    state/api-key; never printed or logged);
  - renames the model "glm-5.3-flash" to "deepseek-v4.1-flash" in a JSON request body;
  - maps GLM's thinking switch to DeepSeek's: chat_template_kwargs.enable_thinking becomes
    chat_template_kwargs.thinking. GLM thinks unless told not to, so a request that does not say turns
    DeepSeek's thinking on too. reasoning_effort passes as it is (DeepSeek accepts low, high, xhigh, max);
  - in the answer to GET /v1/models, lists the model as "glm-5.3-flash" as well;
  - drops return_token_ids from a streamed request (SGLang refuses the pair; vLLM ignored the field).
Responses, including streams, pass through unchanged and unbuffered. It logs counts only, never content.
"""
import argparse
import http.client
import http.server
import json
from pathlib import Path
import socketserver
import sys
import threading
import time

WORK = Path(__file__).resolve().parent
GLM_NAME = 'glm-5.3-flash'
DS_NAME = 'deepseek-v4.1-flash'
HOP = {'host', 'content-length', 'authorization', 'connection', 'keep-alive', 'transfer-encoding', 'te',
       'trailer', 'upgrade', 'proxy-authorization', 'proxy-connection'}
COUNTS = {'requests': 0, 'rewritten': 0, 'thinking_defaulted_on': 0, 'errors': 0}
LOCK = threading.Lock()


def rewrite(body):
    if not isinstance(body, dict):
        return body, False
    changed = False
    if body.get('model') == GLM_NAME:
        body['model'] = DS_NAME
        changed = True
    kwargs = body.get('chat_template_kwargs')
    kwargs = dict(kwargs) if isinstance(kwargs, dict) else {}
    if 'enable_thinking' in kwargs:
        kwargs.setdefault('thinking', bool(kwargs['enable_thinking']))
        del kwargs['enable_thinking']
        changed = True
    if 'thinking' not in kwargs and ('messages' in body):
        kwargs['thinking'] = True
        changed = True
        with LOCK:
            COUNTS['thinking_defaulted_on'] += 1
    if kwargs or 'chat_template_kwargs' in body:
        body['chat_template_kwargs'] = kwargs
    if body.get('stream') and 'return_token_ids' in body:
        # SGLang refuses return_token_ids with a stream (HTTP 400); vLLM ignored the field, and the paper's burst
        # probe runs with --no-token-ids on a vLLM server, so dropping it sends what GLM's server acted on.
        del body['return_token_ids']
        changed = True
    return body, changed


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    upstream = 8010
    key = ''

    def log_message(self, *args):
        pass

    def forward(self):
        with LOCK:
            COUNTS['requests'] += 1
        length = int(self.headers.get('Content-Length') or 0)
        raw = self.rfile.read(length) if length else b''
        if raw and 'json' in (self.headers.get('Content-Type') or ''):
            try:
                body, changed = rewrite(json.loads(raw))
                if changed:
                    raw = json.dumps(body).encode()
                    with LOCK:
                        COUNTS['rewritten'] += 1
            except ValueError:
                pass
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        headers['Authorization'] = 'Bearer ' + self.key
        if raw:
            headers['Content-Length'] = str(len(raw))
        connection = http.client.HTTPConnection('127.0.0.1', self.upstream, timeout=3600)
        try:
            connection.request(self.command, self.path, body=raw or None, headers=headers)
            response = connection.getresponse()
            models = self.command == 'GET' and self.path.rstrip('/') == '/v1/models' and response.status == 200
            if models:
                data = json.loads(response.read())
                ids = [m.get('id') for m in data.get('data', [])]
                if DS_NAME in ids and GLM_NAME not in ids:
                    alias = dict(next(m for m in data['data'] if m.get('id') == DS_NAME))
                    alias['id'] = GLM_NAME
                    data['data'].append(alias)
                payload = json.dumps(data).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(response.status, response.reason)
            chunked = response.getheader('Content-Length') is None
            for k, v in response.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            if chunked:
                self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            while True:
                piece = response.read1(65536)
                if not piece:
                    break
                if chunked:
                    self.wfile.write(b'%x\r\n%s\r\n' % (len(piece), piece))
                else:
                    self.wfile.write(piece)
                self.wfile.flush()
            if chunked:
                self.wfile.write(b'0\r\n\r\n')
                self.wfile.flush()
        except (ConnectionError, OSError, http.client.HTTPException):
            with LOCK:
                COUNTS['errors'] += 1
            try:
                self.send_error(502, 'Upstream error')
            except OSError:
                pass
        finally:
            connection.close()

    do_GET = forward
    do_POST = forward


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--listen', type=int, default=8021)
    parser.add_argument('--upstream', type=int, default=8010)
    args = parser.parse_args()
    Handler.upstream = args.upstream
    Handler.key = (WORK / 'state' / 'api-key').read_text().strip()
    server = Server(('127.0.0.1', args.listen), Handler)
    print(f'adapter on 127.0.0.1:{args.listen} to 127.0.0.1:{args.upstream}', flush=True)

    def report():
        while True:
            time.sleep(300)
            with LOCK:
                print(time.strftime('%H:%M:%S'), json.dumps(COUNTS), flush=True)
    threading.Thread(target=report, daemon=True).start()
    server.serve_forever()


if __name__ == '__main__':
    sys.exit(main())
