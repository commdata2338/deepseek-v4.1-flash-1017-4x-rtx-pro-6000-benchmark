"""Subset-capable orchestration of EvalPlus 0.3.1's own execution and oracles.

The evaluator CLI requires every dataset task. Calling its public execution
functions permits pilots without adding fake samples for unattempted tasks.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from evalplus.config import DEFAULT_GT_TIME_LIMIT_FACTOR, DEFAULT_MIN_TIME_LIMIT
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.eval import PASS, untrusted_check
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.gen.util import trusted_exec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("output already exists; use a new output folder")
    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    ids = [s["task_id"] for s in samples]
    if not ids or len(set(ids)) != len(ids):
        p.error("require one sample per task, and at least one task")
    problems = {**get_human_eval_plus(), **get_mbpp_plus()}
    if set(ids) - problems.keys():
        p.error("unknown task IDs")
    manifest = json.loads(Path("/opt/tasks/manifest.json").read_text())
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".progress.jsonl").open("x") as progress:
        for sample in samples:
            start = time.monotonic()
            task = problems[sample["task_id"]]
            dataset = "humaneval" if task["task_id"].startswith("HumanEval/") else "mbpp"
            solution = sample.get("solution", task["prompt"] + sample.get("completion", ""))
            row = {"task_id": task["task_id"], "dataset": dataset,
                   "solution_sha256": hashlib.sha256(solution.encode()).hexdigest()}
            for suite in ("base", "plus"):
                inputs = task[suite + "_input"]
                # Same reference generation and special-oracle flag as evaluate.get_groundtruth.
                expected, ref_time = trusted_exec(
                    task["prompt"] + task["canonical_solution"], inputs, task["entry_point"],
                    record_time=True,
                    output_not_none=dataset == "mbpp" and task["entry_point"] in MBPP_OUTPUT_NOT_NONE_TASKS,
                )
                status, details = untrusted_check(
                    dataset, solution, inputs, task["entry_point"], expected=expected,
                    atol=task["atol"], ref_time=ref_time, fast_check=True,
                    min_time_limit=DEFAULT_MIN_TIME_LIMIT,
                    gt_time_limit_factor=DEFAULT_GT_TIME_LIMIT_FACTOR,
                )
                row[suite + "_status"] = status
                row[suite + "_pass"] = status == PASS
                row[suite + "_tests_executed"] = len(details)
                row[suite + "_tests_total"] = len(inputs)
            # Published plus scoring requires BOTH base and additional tests.
            row["plus_pass"] = row["base_pass"] and row["plus_pass"]
            row["grade_seconds"] = time.monotonic() - start
            rows.append(row)
            progress.write(json.dumps(row) + "\n")
            progress.flush()
            print(json.dumps(row), flush=True)
    report = {"schema": 1, "evalplus_version": importlib.metadata.version("evalplus"),
              "datasets": manifest["datasets"], "samples_sha256": hashlib.sha256(args.samples.read_bytes()).hexdigest(),
              "min_time_limit": DEFAULT_MIN_TIME_LIMIT, "gt_time_limit_factor": DEFAULT_GT_TIME_LIMIT_FACTOR,
              "fast_check": True, "tasks": rows}
    with args.output.open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
