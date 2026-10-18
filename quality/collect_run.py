"""Put one graded run into one folder, in the layout that scripts/build_data.py of the paper reads.

  python3 -B collect_run.py --run runs/tensorfold-full-c8 --grade runs/tensorfold-full-c8-grade \
      --relay runs/window/relay.jsonl --out collected/tensorfold-full-c8

The output folder gets copies of records.jsonl, samples.jsonl, run.json, graded.json, grade-run.json and the relay
log, and three files that this script makes from the task folders:

  timing.json        the start of the first task and the end of the last task (epoch seconds). The builder uses
                     them to select the model calls of this run in the relay log.
  first_calls.jsonl  for each task: the prompt tokens and the output tokens of its first model call
  exceptions.jsonl   for each task that did not complete or has no solution file: the stop reason and the output
                     tokens of each model call, and the last tool call

An existing file in the output folder is not replaced.
"""
import argparse
import datetime
import json
import shutil
from pathlib import Path


def events(path):
    for line in path.read_text(errors="replace").splitlines():
        try:
            yield json.loads(line)
        except ValueError:
            continue


def write_once(path, text):
    if not path.exists():
        path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--grade", required=True, type=Path)
    parser.add_argument("--relay", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    copies = [(args.run / name, name) for name in ("records.jsonl", "samples.jsonl", "run.json")]
    copies += [(args.grade / name, name) for name in ("graded.json", "grade-run.json")]
    copies.append((args.relay, "relay.jsonl"))
    for source, name in copies:
        if not (args.out / name).exists():
            shutil.copy2(source, args.out / name)

    records = [json.loads(line) for line in (args.run / "records.jsonl").read_text().splitlines()]
    starts, ends, first_calls, exceptions = [], [], [], []
    for record in records:
        attempt = args.run / record["attempt"]
        started = json.loads((attempt / "started.json").read_text())["started_utc"]
        start = datetime.datetime.fromisoformat(started).timestamp()
        starts.append(start)
        ends.append(start + record["wall_seconds"])
        calls, tool_starts, tool_ends = [], [], 0
        for event in events(attempt / "events.jsonl"):
            message = event.get("message") or {}
            if event.get("type") == "message_end" and message.get("role") == "assistant":
                usage = message.get("usage") or {}
                calls.append({"prompt_tokens": sum(usage.get(k) or 0 for k in ("input", "cacheRead", "cacheWrite")),
                              "output_tokens": usage.get("output"), "thought_tokens": usage.get("reasoning"),
                              "stop_reason": message.get("rawStopReason") or message.get("stopReason")})
            elif event.get("type") == "tool_execution_start":
                tool_starts.append({"tool": event.get("toolName"), "arguments": json.dumps(event.get("args"))[:300]})
            elif event.get("type") == "tool_execution_end":
                tool_ends += 1
        if calls:
            first_calls.append({"task_id": record["task_id"], "prompt_tokens": calls[0]["prompt_tokens"],
                                "output_tokens": calls[0]["output_tokens"]})
        if record["status"] != "completed" or not record["solution_present"] or record["limit_hit"]:
            exceptions.append({"task_id": record["task_id"], "status": record["status"], "time_limit_hit": record["limit_hit"],
                               "solution_file_present": record["solution_present"], "wall_seconds": round(record["wall_seconds"], 1),
                               "model_calls": calls, "tool_calls_started": len(tool_starts), "tool_calls_ended": tool_ends,
                               "last_tool_call": tool_starts[-1] if tool_starts else None})
    timing = {"first_task_start": min(starts), "last_task_end": max(ends), "tasks": len(records)}
    write_once(args.out / "timing.json", json.dumps(timing, indent=1) + "\n")
    write_once(args.out / "first_calls.jsonl", "".join(json.dumps(r) + "\n" for r in first_calls))
    write_once(args.out / "exceptions.jsonl", "".join(json.dumps(r) + "\n" for r in exceptions))
    print(f"{args.out}: {len(records)} tasks, {len(exceptions)} exceptions, "
          f"run of {(timing['last_task_end'] - timing['first_task_start']) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
