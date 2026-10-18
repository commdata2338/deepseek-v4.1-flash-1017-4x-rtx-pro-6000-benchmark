"""Offline integration: real pi CLI, mock SSE endpoint, no benchmark model calls."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
import time

from common import ROOT, container_flags, image_id, unique_name, write_json
from probe_sandbox import PROBE
from relay import Handler as RelayHandler
from run_tasks import FLAGS, INSTRUCTION, run_one, summarize_events
from sandbox import GATEWAY, PORT, NETWORK, verify


class Mock(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "glm-5.3-flash"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock:
            self.server.requests.append(request)
        assert request["reasoning_effort"] == "max", request.get("reasoning_effort")
        assert request["temperature"] == 0
        assert request["max_tokens"] == 32768
        assert request["stream_options"]["include_usage"] is True
        assert {t["function"]["name"] for t in request["tools"]} == {"read", "bash", "edit", "write"}
        has_tool = any(m["role"] == "tool" for m in request["messages"])
        if self.server.behavior == "timeout":
            time.sleep(5)
            self.close_connection = True
            return
        if self.server.behavior == "missing":
            has_tool = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        def send(choices, usage=None):
            chunk = {"id": "fixture", "object": "chat.completion.chunk", "created": 0,
                     "model": "glm-5.3-flash", "choices": choices}
            if usage is not None:
                chunk["usage"] = usage
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.flush()
        send([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}])
        if not has_tool:
            send([{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "fixture_write", "type": "function",
                "function": {"name": "write", "arguments": json.dumps({"path": "solution.py", "content": "def square(x):\n    return x * x\n"})}}]}, "finish_reason": None}])
        else:
            send([{"index": 0, "delta": {"content": "Complete."}, "finish_reason": None}])
        send([{"index": 0, "delta": {}, "finish_reason": "stop" if has_tool else "tool_calls"}])
        send([], {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
                  "prompt_tokens_details": {"cached_tokens": 20}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    verify()
    with ThreadingHTTPServer(("127.0.0.1", 0), Mock) as mock, args.output.joinpath("relay.jsonl").open("x") as audit:
        mock.requests, mock.lock, mock.behavior = [], threading.Lock(), "write"
        with ThreadingHTTPServer((GATEWAY, PORT), RelayHandler) as relay:
            relay.engine_port = mock.server_port
            relay.audit, relay.audit_lock = audit, threading.Lock()
            threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (mock, relay)]
            for thread in threads:
                thread.start()
            try:
                work = args.output / "probe-work"
                work.mkdir()
                script = PROBE.replace("GATEWAY", GATEWAY).replace("PORT", str(PORT))
                # A known-live host loopback listener must also be unreachable directly.
                script = script.replace('(\"host_ssh\", \"' + GATEWAY + '\", 22)',
                                        '("known_live_loopback_mock", "' + GATEWAY + '", ' + str(mock.server_port) + ')')
                cmd = container_flags(unique_name("probe"), NETWORK, work)
                result = subprocess.run(cmd + ["--entrypoint", "python", "bench-pi-agent:0.86.1", "-c", script],
                                        check=True, capture_output=True, text=True, timeout=45)
                (args.output / "sandbox.json").write_text(result.stdout)
                for provider in ("tensorfold", "vllm"):
                    folder = args.output / provider
                    fixture = {"task_id": "Fixture/0", "prompt": "Implement square(x), returning x multiplied by itself.\n"}
                    options = argparse.Namespace(output=folder, endpoint=f"http://{GATEWAY}:{PORT}/v1",
                                provider=provider, wall_limit=60, task_datasets={"Fixture/0": "fixture"})
                    metrics = run_one(fixture, options, image_id("bench-pi-agent:0.86.1"))
                    assert metrics["solution"] == "def square(x):\n    return x * x\n", metrics
                    assert metrics["model_calls"] == 2, metrics
                    assert metrics["total_input_tokens"] == 200 and metrics["output_tokens"] == 20, metrics
                    assert not metrics["agent_errors"], metrics
                    write_json(folder / "metrics.json", metrics)
                for behavior in ("missing", "timeout"):
                    mock.behavior = behavior
                    options.output = args.output / behavior
                    options.wall_limit = 2 if behavior == "timeout" else 60
                    metrics = run_one(fixture, options, image_id("bench-pi-agent:0.86.1"))
                    assert not metrics["solution_present"], metrics
                    assert metrics["limit_hit"] == (behavior == "timeout"), metrics
                    assert metrics["status"] == ("timeout" if behavior == "timeout" else "failed"), metrics
                    write_json(options.output / "metrics.json", metrics)
                write_json(args.output / "requests.json", mock.requests)
                write_json(args.output / "result.json", {"status": "passed", "mock_model_calls": len(mock.requests),
                           "live_model_calls": 0, "providers": ["tensorfold", "vllm"]})
                print("Offline integration passed: isolation, both providers, max effort, temperature, tool write, token accounting.")
            finally:
                relay.shutdown()
                mock.shutdown()


if __name__ == "__main__":
    main()
