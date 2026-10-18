"""Regression checks. All created fixtures are retained under evidence/test-work."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common import ROOT, write_json
from compare import comparison, mcnemar, wilson
import run_tasks


class HarnessTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / "evidence/test-work"
        base.mkdir(exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix="fixture-", dir=base))

    def test_mcnemar_exact(self):
        self.assertEqual(mcnemar(0, 0), 1)
        self.assertEqual(mcnemar(6, 0), 0.03125)
        self.assertEqual(mcnemar(0, 6), 0.03125)
        self.assertEqual(mcnemar(3, 3), 1)
        self.assertAlmostEqual(mcnemar(9, 1), 22 / 1024)

    def test_wilson_boundaries(self):
        lo, hi = wilson(4, 4)
        self.assertAlmostEqual(lo, 0.5101091635454027)
        self.assertAlmostEqual(hi, 1)
        lo, hi = wilson(0, 4)
        self.assertAlmostEqual(lo, 0)
        self.assertAlmostEqual(hi, 0.4898908364545973)
        with self.assertRaises(ValueError):
            wilson(0, 0)

    def test_event_usage_counted_once(self):
        msg = {"role": "assistant", "usage": {"input": 80, "output": 10, "cacheRead": 20, "cacheWrite": 0}, "stopReason": "stop"}
        events = [{"type": "message_start", "message": {"role": "assistant"}},
                  {"type": "message_update", "message": msg},
                  {"type": "message_end", "message": msg},
                  {"type": "agent_end", "messages": [msg]}]
        path = self.work / "events.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in events))
        got = run_tasks.summarize_events(path)
        self.assertEqual(got["model_calls"], 1)
        self.assertEqual(got["total_input_tokens"], 100)
        self.assertEqual(got["output_tokens"], 10)
        self.assertTrue(got["usage_complete"])

    def test_partial_event_stream_usage_is_incomplete(self):
        path = self.work / "events.jsonl"
        path.write_text(json.dumps({"type": "message_start", "message": {"role": "assistant"}}) + '\n{"partial"')
        got = run_tasks.summarize_events(path)
        self.assertEqual(got["model_calls"], 1)
        self.assertFalse(got["usage_complete"])
        self.assertEqual(got["malformed_event_lines"], 1)

    def test_snapshots_preserve_old_solutions_and_failures(self):
        rows = {"HumanEval/0": {"task_id": "HumanEval/0", "solution": "old"}}
        run_tasks.snapshot(self.work, rows)
        old = (self.work / "samples.jsonl").read_bytes()
        rows["HumanEval/0"]["solution"] = "new"
        rows["HumanEval/1"] = {"task_id": "HumanEval/1", "solution": ""}
        run_tasks.snapshot(self.work, rows)
        files = list(self.work.glob("snapshots/*/samples.jsonl"))
        self.assertEqual(len(files), 2)
        self.assertIn(old, [f.read_bytes() for f in files])
        current = [json.loads(s) for s in (self.work / "samples.jsonl").read_text().splitlines()]
        self.assertEqual(current[-1], {"task_id": "HumanEval/1", "solution": ""})

    def test_nonobject_event_is_recorded_as_malformed(self):
        path = self.work / "events.jsonl"
        path.write_text('null\n[]\n{"message":[]}\n')
        got = run_tasks.summarize_events(path)
        self.assertFalse(got["usage_complete"])
        self.assertEqual(got["malformed_event_lines"], 2)

    def reports(self):
        meta = {"schema": 1, "evalplus_version": "fixture", "datasets": {}, "min_time_limit": 1,
                "gt_time_limit_factor": 4, "fast_check": True}
        rows = [{"task_id": f"HumanEval/{n}", "dataset": "humaneval", "base_pass": n < 2, "plus_pass": n < 2} for n in range(4)]
        a, b = self.work / "a.json", self.work / "b.json"
        a.write_text(json.dumps({**meta, "tasks": rows}))
        b.write_text(json.dumps({**meta, "tasks": [{**r, "base_pass": False, "plus_pass": False} for r in rows]}))
        return a, b

    def test_comparison_paired_counts(self):
        a, b = self.reports()
        table = comparison([a], [b])
        self.assertIn("| 0 | 2 | 0 | 2 | 0.5 |", table)
        self.assertIn("overall", table)

    def test_comparison_rejects_missing_tasks(self):
        a, b = self.reports()
        obj = json.loads(b.read_text())
        obj["tasks"].pop()
        b.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, "identical"):
            comparison([a], [b])

    def test_comparison_rejects_duplicate_tasks(self):
        a, b = self.reports()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            comparison([a, a], [b])

    def test_comparison_rejects_different_grading(self):
        a, b = self.reports()
        obj = json.loads(b.read_text())
        obj["evalplus_version"] = "different"
        b.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, "differ"):
            comparison([a], [b])

    def test_resume_skips_failures_and_rerun_keeps_attempts(self):
        def fake(task, args, image):
            folder = args.output / "tasks" / task["task_id"].replace("/", "_") / f"attempt-{mock.call_count}"
            folder.mkdir(parents=True)
            record = {"task_id": task["task_id"], "solution": "", "status": "failed",
                      "wall_seconds": 1, "model_calls": 0, "input_tokens": 0, "output_tokens": 0}
            write_json(folder / "record.json", record)
            return record
        argv = ["run_tasks.py", "--dataset", "humaneval", "--limit", "1", "--output", str(self.work)]
        with patch.object(run_tasks, "verify"), patch.object(run_tasks, "image_id", return_value="fixture"), patch.object(run_tasks, "run_one", side_effect=fake) as mock:
            with patch("sys.argv", argv):
                run_tasks.main()
                run_tasks.main()
            self.assertEqual(mock.call_count, 1)
            with patch("sys.argv", [*argv, "--rerun"]):
                run_tasks.main()
            self.assertEqual(mock.call_count, 2)
            self.assertEqual(len(list(self.work.glob("tasks/*/*/record.json"))), 2)


if __name__ == "__main__":
    unittest.main()
