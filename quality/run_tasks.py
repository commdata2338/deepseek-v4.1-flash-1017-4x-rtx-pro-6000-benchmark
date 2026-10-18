"""Run one upstream pi process per task, retaining every attempt and snapshot."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import threading
import time

from common import ROOT, container_flags, image_id, sha, stop_owned, unique_name, write_json
from sandbox import GATEWAY, PORT, NETWORK, verify

INSTRUCTION = "Read PROBLEM.md. Implement the requested function in solution.py. You may run Python to check your work. Stop when solution.py is complete."
FLAGS = ["--mode", "json", "--print", "--thinking", "max", "--no-session", "--no-extensions",
         "--no-skills", "--no-prompt-templates", "--no-context-files", "--offline", "--no-approve",
         "--tools", "read,bash,edit,write"]
CANCEL = threading.Event()


def summarize_events(path):
    counts = {"model_calls": 0, "completed_model_calls": 0, "input_tokens": 0,
              "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0,
              "usage_records": 0, "malformed_event_lines": 0, "agent_errors": []}
    with path.open() as f:
        for line in f:
            try:
                event = json.loads(line)
            except ValueError:
                counts["malformed_event_lines"] += 1
                continue
            if not isinstance(event, dict):
                counts["malformed_event_lines"] += 1
                continue
            msg = event.get("message") or {}
            if not isinstance(msg, dict):
                counts["malformed_event_lines"] += 1
                continue
            if msg.get("role") != "assistant":
                continue
            if event.get("type") == "message_start":
                counts["model_calls"] += 1
            elif event.get("type") == "message_end":
                counts["completed_model_calls"] += 1
                usage = msg.get("usage")
                if usage:
                    counts["usage_records"] += 1
                    for field, target in [("input", "input_tokens"), ("output", "output_tokens"),
                                          ("cacheRead", "cache_read_tokens"), ("cacheWrite", "cache_write_tokens")]:
                        counts[target] += usage.get(field, 0)
                if msg.get("stopReason") in ("error", "aborted"):
                    counts["agent_errors"].append(msg.get("errorMessage", msg["stopReason"]))
    counts["model_calls"] = max(counts["model_calls"], counts["completed_model_calls"])
    counts["total_input_tokens"] = counts["input_tokens"] + counts["cache_read_tokens"] + counts["cache_write_tokens"]
    counts["usage_complete"] = (counts["model_calls"] > 0 and
                               counts["usage_records"] == counts["model_calls"] and
                               not counts["agent_errors"])
    return counts


def snapshot(output, records):
    folder = output / "snapshots" / unique_name("snapshot")
    folder.mkdir(parents=True)
    ordered = sorted(records.values(), key=lambda r: (r["task_id"].split("/")[0], int(r["task_id"].split("/")[1])))
    samples = "".join(json.dumps({"task_id": r["task_id"], "solution": r["solution"]}) + "\n" for r in ordered)
    summary = "".join(json.dumps({k: v for k, v in r.items() if k != "solution"}) + "\n" for r in ordered)
    for name, data in (("samples.jsonl", samples), ("records.jsonl", summary)):
        (folder / name).write_text(data)
        # Every earlier top-level value remains in a uniquely named snapshot.
        temp = output / ("." + unique_name("publish"))
        temp.write_text(data)
        os.replace(temp, output / name)


def run_one(task, args, immutable_image):
    task_root = args.output / "tasks" / task["task_id"].replace("/", "_")
    attempt = task_root / unique_name("attempt")
    work = attempt / "work"
    work.mkdir(parents=True)
    (work / "PROBLEM.md").write_text(task["prompt"])
    name = unique_name("agent")
    cmd = container_flags(name, NETWORK, work)
    cmd += ["--env", "QUALITY_RELAY_URL=" + args.endpoint, immutable_image,
            "--provider", args.provider, "--model", "glm-5.3-flash", *FLAGS, INSTRUCTION]
    write_json(attempt / "started.json", {"task_id": task["task_id"], "container_name": name,
               "started_utc": datetime.now(timezone.utc).isoformat(), "wall_limit": args.wall_limit})
    start = time.monotonic()
    hit_limit = False
    exit_code = None
    infrastructure_error = None
    with (attempt / "events.jsonl").open("x") as events, (attempt / "stderr.log").open("x") as errors:
        try:
            if CANCEL.is_set():
                raise RuntimeError("run cancelled before container launch")
            proc = subprocess.Popen(cmd, stdout=events, stderr=errors)
            # Wait on an Event rather than busy-polling the Docker process.
            done = threading.Event()
            waiter = threading.Thread(target=lambda: (proc.wait(), done.set()), daemon=True)
            waiter.start()
            deadline = start + args.wall_limit
            while not done.wait(min(1, max(0, deadline - time.monotonic()))):
                if CANCEL.is_set() or time.monotonic() >= deadline:
                    hit_limit = not CANCEL.is_set()
                    stop_owned(name)
                    break
            try:
                exit_code = proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.terminate()
                exit_code = proc.wait(timeout=10)
        except Exception as exc:
            infrastructure_error = type(exc).__name__ + ": " + str(exc)
        finally:
            stop_owned(name)
    wall = time.monotonic() - start
    solution = ""
    solution_present = False
    # Do not follow a solution symlink written by generated code into the host.
    try:
        fd = os.open(work / "solution.py", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "r") as f:
            st = os.fstat(f.fileno())
            if not stat.S_ISREG(st.st_mode) or st.st_size > 8 * 1024 * 1024:
                raise ValueError("solution is not a regular file of at most 8 MiB")
            solution = f.read()
            solution_present = bool(solution.strip())
    except (OSError, ValueError, UnicodeError) as exc:
        infrastructure_error = infrastructure_error or ("no readable solution: " + type(exc).__name__)
    metrics = summarize_events(attempt / "events.jsonl")
    status = ("cancelled" if CANCEL.is_set() else "timeout" if hit_limit else
              "failed" if exit_code != 0 or not solution_present or metrics["agent_errors"] else "completed")
    record = {"task_id": task["task_id"], "dataset": args.task_datasets[task["task_id"]],
              "attempt": str(attempt.relative_to(args.output)), "wall_seconds": wall,
              "exit_status": exit_code, "status": status, "limit_hit": hit_limit,
              "solution_present": solution_present, "solution_sha256": hashlib.sha256(solution.encode()).hexdigest(),
              "infrastructure_error": infrastructure_error, "solution": solution, **metrics}
    write_json(attempt / "record.json", record)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default=f"http://{GATEWAY}:{PORT}/v1", help="sandbox relay URL")
    p.add_argument("--provider", choices=["vllm", "tensorfold", "endpoint"], default="endpoint")
    p.add_argument("--dataset", choices=["humaneval", "mbpp", "both"], required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--tasks", nargs="+", help="explicit IDs, e.g. HumanEval/0 Mbpp/2")
    group.add_argument("--limit", type=int)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--wall-limit", type=float, default=1200)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image", default="bench-pi-agent:0.86.1")
    p.add_argument("--rerun", action="store_true", help="new attempts; retain all old bytes and snapshots")
    args = p.parse_args()
    if args.concurrency < 1 or args.wall_limit <= 0 or (args.limit is not None and args.limit < 1):
        p.error("limits and concurrency must be positive")
    if args.endpoint != f"http://{GATEWAY}:{PORT}/v1":
        p.error("endpoint must be the sandbox relay; select the upstream port with relay.py")
    verify()
    datasets = ["humaneval", "mbpp"] if args.dataset == "both" else [args.dataset]
    dataset_manifest = json.loads((ROOT / "datasets/manifest.json").read_text())
    tasks, args.task_datasets = {}, {}
    for dataset in datasets:
        if sha(ROOT / "datasets" / (dataset + ".jsonl")) != dataset_manifest["datasets"][dataset]["prompts_sha256"]:
            p.error("prompt export hash differs from the dataset manifest")
        for line in (ROOT / "datasets" / (dataset + ".jsonl")).read_text().splitlines():
            task = json.loads(line)
            tasks[task["task_id"]] = task
            args.task_datasets[task["task_id"]] = dataset
    selected = args.tasks if args.tasks else list(tasks)[:args.limit]
    if len(set(selected)) != len(selected) or set(selected) - tasks.keys() or not selected:
        p.error("task list is empty, has duplicates, or contains unknown IDs")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        immutable_image = image_id(args.image)
        manifest = {"schema": 1, "endpoint": args.endpoint, "provider": args.provider,
                    "image_id": immutable_image, "pi_version": "0.86.1", "task_ids": selected,
                    "dataset_hashes": {d: sha(ROOT / "datasets" / (d + ".jsonl")) for d in datasets},
                    "instruction": INSTRUCTION, "flags": FLAGS, "wall_limit": args.wall_limit,
                    "concurrency": args.concurrency,
                    "models_sha256": sha(ROOT / "pi-config/models.json"),
                    "settings_sha256": sha(ROOT / "pi-config/settings.json")}
        path = args.output / "run.json"
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                p.error("run configuration changed; use a new output folder")
        else:
            write_json(path, manifest)
        records = {}
        # Records are outside the agent's only writable bind mount.
        for task_id in selected:
            attempts = sorted((args.output / "tasks" / task_id.replace("/", "_")).glob("*/record.json"),
                              key=lambda file: file.stat().st_mtime_ns)
            if attempts:
                records[task_id] = json.loads(attempts[-1].read_text())
        pending = [task_id for task_id in selected if args.rerun or task_id not in records
                   or records[task_id]["status"] == "cancelled"]
        def cancel(signum, frame):
            CANCEL.set()
        signal.signal(signal.SIGINT, cancel)
        signal.signal(signal.SIGTERM, cancel)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(run_one, tasks[tid], args, immutable_image): tid for tid in pending}
            for future in as_completed(futures):
                record = future.result()
                records[record["task_id"]] = record
                snapshot(args.output, records)
                print(json.dumps({k: record[k] for k in ("task_id", "status", "wall_seconds", "model_calls",
                                                          "input_tokens", "output_tokens")}), flush=True)
        if not pending:
            snapshot(args.output, records)
            print(f"Skipped {len(records)} finished tasks")
        if CANCEL.is_set():
            raise SystemExit(130)


if __name__ == "__main__":
    main()
