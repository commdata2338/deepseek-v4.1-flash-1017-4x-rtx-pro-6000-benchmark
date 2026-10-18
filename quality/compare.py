"""Paired pass@1 comparison with Wilson intervals and exact McNemar tests."""
import argparse
import json
import math
from pathlib import Path


def wilson(k, n):
    if not n:
        raise ValueError("empty sample")
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0, center - radius), min(1, center + radius)


def mcnemar(b, c):
    # Conditional two-sided Binomial(b+c, 1/2), no continuity approximation.
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n) if n else 1.0


def load(paths):
    rows, metadata = {}, {}
    for path in paths:
        report = json.loads(Path(path).read_text())
        if report.get("schema") != 1:
            raise ValueError("unsupported graded report")
        for field in ("evalplus_version", "datasets", "min_time_limit", "gt_time_limit_factor", "fast_check"):
            if field in metadata and metadata[field] != report[field]:
                raise ValueError("inconsistent grader or dataset metadata")
            metadata[field] = report[field]
        for row in report["tasks"]:
            if row["task_id"] in rows:
                raise ValueError("duplicate task ID")
            if type(row["base_pass"]) is not bool or type(row["plus_pass"]) is not bool:
                raise ValueError("missing pass/fail result")
            expected = "humaneval" if row["task_id"].startswith("HumanEval/") else "mbpp"
            if row["dataset"] != expected:
                raise ValueError("inconsistent dataset label")
            rows[row["task_id"]] = row
    return rows, metadata


def comparison(a_paths, b_paths):
    a, am = load(a_paths)
    b, bm = load(b_paths)
    if am != bm:
        raise ValueError("datasets or grading settings differ")
    if a.keys() != b.keys() or not a:
        raise ValueError("paired comparison requires identical, nonempty task sets")
    out = ["| Dataset | Suite | N | A pass (95% CI) | B pass (95% CI) | Both | A only | B only | Neither | Exact p |",
           "|---|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for dataset in ("humaneval", "mbpp", "overall"):
        ids = [tid for tid in a if dataset == "overall" or a[tid]["dataset"] == dataset]
        if not ids:
            continue
        for suite in ("base", "plus"):
            pairs = [(a[tid][suite + "_pass"], b[tid][suite + "_pass"]) for tid in ids]
            both, only_a, only_b, neither = [pairs.count(pair) for pair in [(True, True), (True, False), (False, True), (False, False)]]
            def rate(k):
                low, high = wilson(k, len(ids))
                return f"{k}/{len(ids)} = {k/len(ids):.1%} [{low:.1%}, {high:.1%}]"
            out.append(f"| {dataset} | {suite} | {len(ids)} | {rate(both+only_a)} | {rate(both+only_b)} | {both} | {only_a} | {only_b} | {neither} | {mcnemar(only_a, only_b):.6g} |")
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=Path, nargs="+", required=True)
    p.add_argument("--b", type=Path, nargs="+", required=True)
    args = p.parse_args()
    try:
        print(comparison(args.a, args.b), end="")
    except (ValueError, KeyError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
