"""Dataset-weighted extrapolation, explicitly conditional on unchanged latency."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=Path, nargs="+", required=True)
    args = p.parse_args()
    rows = [json.loads(line) for path in args.records for line in path.read_text().splitlines()]
    if len({r["task_id"] for r in rows}) != len(rows):
        p.error("duplicate tasks")
    total_seconds, input_tokens, output_tokens = 0, 0, 0
    print("Conditional arithmetic only: pilot task latency held fixed as concurrency rises.")
    print("This is not a measured throughput or engine comparison. Grading time is separate.\n")
    print("| Dataset | Pilot N | Mean task seconds | Full tasks | Summed task hours |")
    print("|---|---:|---:|---:|---:|")
    for dataset, count in (("humaneval", 164), ("mbpp", 378)):
        group = [r for r in rows if r["dataset"] == dataset]
        if len(group) < 2 or any(r["status"] != "completed" or not r["usage_complete"] for r in group):
            p.error("need two completed tasks with usage per dataset for this pilot extrapolation")
        mean = statistics.mean(r["wall_seconds"] for r in group)
        total_seconds += count * mean
        input_tokens += count * statistics.mean(r["total_input_tokens"] for r in group)
        output_tokens += count * statistics.mean(r["output_tokens"] for r in group)
        print(f"| {dataset} | {len(group)} | {mean:.2f} | {count} | {count * mean / 3600:.2f} |")
    print(f"\nProjected reported input tokens (including cache): {input_tokens:,.0f}; output: {output_tokens:,.0f}.")
    print("\n| Concurrency | Ideal generation hours | Minutes |")
    print("|---|---:|---:|")
    for concurrency in (8, 16):
        print(f"| {concurrency} | {total_seconds / concurrency / 3600:.3f} | {total_seconds / concurrency / 60:.1f} |")
    print("\nTensorFold was not timed. If its task latency is r times the pilot latency, multiply these times by r.")
    print("The first two tasks per dataset are a smoke sample; no confidence interval for full-run time is justified.")


if __name__ == "__main__":
    main()
