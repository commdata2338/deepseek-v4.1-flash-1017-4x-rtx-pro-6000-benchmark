"""Record or compare seeded real-model replies using a frozen JSONL request corpus.

Each corpus line: {"name": "case", "body": {OpenAI chat request}}. Require explicit
seed/temperature/top_k/top_p/min_p, stream=false and return_token_ids=true. Outputs
are exclusively created; existing evidence is never overwritten. No service is started.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def checked_request(case):
    body = case["body"]
    for key in ("seed", "temperature", "top_k", "top_p", "min_p"):
        if key not in body:
            raise ValueError(f"{case['name']}: explicit {key} required")
    if body.get("stream", False) or body.get("return_token_ids") is not True:
        raise ValueError("stream=false and return_token_ids=true required")
    return canonical(body)


def reply_bits(response):
    ids = response.get("tensorfold", {}).get("token_ids")
    if not isinstance(ids, list) or not ids or any(type(x) is not int for x in ids):
        raise ValueError("response must include nonempty tensorfold.token_ids")
    choices = [{key: choice[key] for key in ("index", "message", "text", "finish_reason") if key in choice}
               for choice in response["choices"]]
    return {"token_ids": ids, "choices": choices}


def compare(actual, expected):
    return (actual["name"] == expected["name"] and actual["request_sha256"] == expected["request_sha256"]
            and actual["reply"] == expected["reply"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.requests.read_text().splitlines() if line.strip()]
    bodies = [checked_request(case) for case in cases]
    expected = [json.loads(line) for line in args.reference.read_text().splitlines()] if args.reference else None
    if expected is not None and len(expected) != len(cases):
        raise ValueError("reference and corpus lengths differ")
    def run(pair):
        case, body = pair
        request = urllib.request.Request(args.base_url.rstrip("/") + "/v1/chat/completions", body,
                                         {"Content-Type": "application/json"})
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=1800) as reply:
            response = json.load(reply)
        return {"name": case["name"], "request_sha256": hashlib.sha256(body).hexdigest(),
                "reply": reply_bits(response), "seconds": time.monotonic() - start}
    bad = []
    with args.output.open("x") as output, ThreadPoolExecutor(args.concurrency) as pool:
        for i, result in enumerate(pool.map(run, zip(cases, bodies))):
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            if expected is not None and not compare(result, expected[i]):
                bad.append(result["name"])
    print(f"{len(cases)} replies {'compared' if expected is not None else 'recorded'}; {len(bad)} mismatches")
    if bad:
        raise SystemExit("mismatched cases: " + ", ".join(bad))


if __name__ == "__main__":
    main()
