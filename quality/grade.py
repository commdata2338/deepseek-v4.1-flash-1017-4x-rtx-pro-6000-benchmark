"""Run EvalPlus grading in a capped container with no network."""
import argparse
from pathlib import Path
import subprocess
import time

from common import ROOT, container_flags, image_id, sha, stop_owned, unique_name, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="new directory; never reuse an old grade")
    p.add_argument("--wall-limit", type=float, default=14400)
    p.add_argument("--image", default="bench-evalplus:0.3.1")
    args = p.parse_args()
    if args.wall_limit <= 0:
        p.error("wall limit must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    # Read-only samples mount; only this invocation's output folder is writable.
    name = unique_name("grader")
    image = image_id(args.image)
    cmd = container_flags(name, "none", args.output, memory="4g", cpus="2", pids=128)
    cmd += ["--mount", f"type=bind,src={args.samples.resolve()},dst=/input/samples.jsonl,readonly",
            image, "--samples", "/input/samples.jsonl", "--output", "/work/graded.json"]
    start = time.monotonic()
    limit_hit = False
    exit_code = None
    try:
        with (args.output / "grader.log").open("x") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            try:
                exit_code = proc.wait(timeout=args.wall_limit)
            except subprocess.TimeoutExpired:
                limit_hit = True
                stop_owned(name)
                exit_code = proc.wait(timeout=20)
            except KeyboardInterrupt:
                stop_owned(name)
                exit_code = proc.wait(timeout=20)
                raise
    finally:
        stop_owned(name)
        write_json(args.output / "grade-run.json", {
            "image_id": image, "samples_sha256": sha(args.samples), "network": "none",
            "wall_seconds": time.monotonic() - start, "limit_hit": limit_hit, "exit_status": exit_code})
    if exit_code != 0 or limit_hit:
        raise SystemExit("Grading incomplete; inspect the preserved grader.log")
    print(args.output / "graded.json")


if __name__ == "__main__":
    main()
