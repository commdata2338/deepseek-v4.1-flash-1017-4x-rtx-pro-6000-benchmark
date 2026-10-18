"""Export prompts only, and hash the complete upstream datasets separately."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.data.humaneval import HUMANEVAL_PLUS_VERSION, _ready_human_eval_plus_path
from evalplus.data.mbpp import MBPP_PLUS_VERSION, _ready_mbpp_plus_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"evalplus_version": importlib.metadata.version("evalplus"), "datasets": {}}
    for name, getter, path_getter, version, count in [
        ("humaneval", get_human_eval_plus, _ready_human_eval_plus_path, HUMANEVAL_PLUS_VERSION, 164),
        ("mbpp", get_mbpp_plus, _ready_mbpp_plus_path, MBPP_PLUS_VERSION, 378),
    ]:
        problems = getter()
        assert len(problems) == count, (name, len(problems))
        output = args.output / (name + ".jsonl")
        with output.open("x") as f:
            for task in sorted(problems.values(), key=lambda t: int(t["task_id"].split("/")[1])):
                f.write(json.dumps({k: task[k] for k in ("task_id", "prompt", "entry_point")}) + "\n")
        raw = Path(path_getter()).read_bytes()
        manifest["datasets"][name] = {
            "version": version, "count": count,
            "upstream_sha256": hashlib.sha256(raw).hexdigest(),
            "evalplus_md5": hashlib.md5(raw).hexdigest(),
            "prompts_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
