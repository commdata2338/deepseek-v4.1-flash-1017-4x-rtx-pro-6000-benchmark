"""Short-context probe: does the server read the last tokens of a short prompt?

Release r38 of Jovian Judgement contains a change to the token selection of the sparse attention (pull request 715
of the fork, "keep partial pool tails inside the active selection prefix"). Releases r24 and r28.1 do not contain
it. Its description says that a short prompt can lose the last tokens of a pool that is not complete. A pool has 4
tokens, and the change applies to a prompt of less than 2,048 tokens. This probe examines a server for that
pattern. A comparison of two releases with it does not isolate the change.

Method. Each prompt is a sequence with one correct subsequent token: numbers in ascending order, a cycle of words,
a cycle of letters, numbered lines of code, a JSON list of ids in ascending order. The probe cuts each sequence to
an exact number of tokens and sends it as token ids to /v1/completions, with max_tokens 1, temperature 0, and
logprobs 20. Each request has its own cache salt, so the server does the prefill of the full prompt. The probe
records the log probability of the correct subsequent token, if it is one of the 20 most probable tokens, and the
token that the server selected. The summary uses the probability 0 for a correct token that is not in that list. The lengths come
in groups of 8 consecutive values. Thus each group has two lengths for each remainder of the length divided by 4.

How to read the result. For a length of less than 2,048 tokens that is not a multiple of 4, the last tokens are in
a pool that is not complete. If the server does not read them, the correct token gets less probability at these
lengths than at the multiples of 4 of the same group. The groups from 2,056 tokens are the control.

  python3 short_context.py run --base-url http://127.0.0.1:8000 --out shortctx.jsonl [--model glm-5.3-flash]
  python3 short_context.py summary shortctx.jsonl [other.jsonl]

The probe uses the Python standard library only. The server must give /tokenize and /v1/completions with token ids
as the prompt (vLLM does).
"""
import argparse
import concurrent.futures
import json
import math
import statistics
import sys
import urllib.request

RUN_STARTS = (8, 96, 508, 1020, 2040, 2048, 2056, 3000, 6000)     # each group: 8 consecutive lengths from here
SEEDS = (0, 1, 2)
PREFIX = "[gMASK]<sop>"


def sequences(seed):
    """Texts with one correct continuation, long enough for the longest group."""
    start = 1000 + 137 * seed
    words = ["red", "green", "blue", "yellow", "black", "white", "brown"][seed:] + ["red", "green"][:seed]
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[seed * 3:] + "ABCDEFGHIJ"[:seed * 3]
    return {
        "integers": " ".join(str(start + i) for i in range(9000)),
        "word cycle": " ".join(words[i % len(words)] for i in range(9000)),
        "letter cycle": " ".join(letters[i % len(letters)] for i in range(9000)),
        "code lines": "".join(f"x{start + i} = {start + i}\n" for i in range(4000)),
        "json ids": "[" + ", ".join(f'{{"id": {start + i}}}' for i in range(4000)),
    }


def post(url, body, timeout=300):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def cases(tokenize):
    """Each (kind, seed, group, length, prompt ids, correct subsequent id)."""
    prefix = tokenize(PREFIX)
    out = []
    for seed in SEEDS:
        for kind, text in sequences(seed).items():
            ids = prefix + tokenize(text)
            assert len(ids) > RUN_STARTS[-1] + 8, (kind, len(ids))
            for run in RUN_STARTS:
                for length in range(run, run + 8):
                    part = length % 4            # tokens in the pool that is not complete
                    out.append({"kind": kind, "seed": seed, "run": run, "length": length,
                                "prompt": ids[:length], "expected": ids[length],
                                "stale": ids[length - part] if part else None})
    return out


def request_body(model, case, number):
    return {"model": model, "prompt": case["prompt"], "max_tokens": 1, "temperature": 0, "logprobs": 20,
            "return_tokens_as_token_ids": True, "return_token_ids": True,
            "cache_salt": f"shortctx-{number}-{case['length']}"}


def parse_reply(reply, case):
    """One result row from a /v1/completions reply. A reply with no logprobs still gives the selected token."""
    choice = reply["choices"][0]
    top = ((choice.get("logprobs") or {}).get("top_logprobs") or [None])[0] or {}   # {"token_id:123": logprob}
    ranked = sorted(top.items(), key=lambda item: -item[1])
    generated = (choice.get("token_ids") or [None])[0]
    if generated is None and ranked:
        generated = int(ranked[0][0].split(":")[1])

    def logprob(token_id):
        return None if token_id is None else top.get(f"token_id:{token_id}")
    key = f"token_id:{case['expected']}"
    row = {k: case[k] for k in ("kind", "seed", "run", "length", "expected", "stale")}
    row.update(prompt_tokens=reply["usage"]["prompt_tokens"], generated=generated, has_logprobs=bool(top),
               top1_logprob=ranked[0][1] if ranked else None, expected_logprob=logprob(case["expected"]),
               expected_rank=next((i for i, (k, _) in enumerate(ranked) if k == key), None),
               stale_logprob=logprob(case["stale"]))
    return row


def ask(base_url, model, case, number):
    return parse_reply(post(base_url + "/v1/completions", request_body(model, case, number)), case)


def run(args):
    def tokenize(text):
        return post(args.base_url + "/tokenize", {"model": args.model, "prompt": text,
                                                  "add_special_tokens": False})["tokens"]
    todo = cases(tokenize)
    print(f"{len(todo)} requests, lengths {RUN_STARTS[0]} to {RUN_STARTS[-1] + 7}", flush=True)
    with open(args.out, "w") as out, concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
        futures = [pool.submit(ask, args.base_url, args.model, case, n) for n, case in enumerate(todo)]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            assert row["prompt_tokens"] == row["length"], row
            out.write(json.dumps(row) + "\n")
            if done % 200 == 0:
                print(f"  {done} done", flush=True)
    summary([args.out])


def table(path):
    """{(group, the length is a multiple of 4): [...]} for the probability of the correct token and two counts."""
    prob, hit, stale, missing = {}, {}, {}, 0
    for line in open(path):
        row = json.loads(line)
        key = (row["run"], row["length"] % 4 == 0)
        logprob = row["expected_logprob"]
        missing += not row.get("has_logprobs", True)
        prob.setdefault(key, []).append(math.exp(logprob) if logprob is not None else 0.0)
        hit.setdefault(key, []).append(row["generated"] == row["expected"])
        stale.setdefault(key, []).append(row["stale"] is not None and row["generated"] == row["stale"]
                                         and row["stale"] != row["expected"])
    return prob, hit, stale, missing


def summary(paths):
    tables = [table(path) for path in paths]
    print()
    print("p = mean probability of the correct subsequent token (0 if it is not in the 20 most probable tokens).")
    print("first = share of the prompts where the correct token is the first choice.")
    print("stale = share of the prompts where the first choice is the token that is correct with the last 1-3 tokens gone.")
    print("'whole pools' = the length is a multiple of 4; 'part pool' = remainder 1, 2 or 3.")
    for path, (prob, hit, stale, missing) in zip(paths, tables):
        print()
        print(f"{path}" + (f"   ({missing} replies had no logprobs)" if missing else ""))
        print(f"{'lengths':>12s} | whole pools: p, first | part pool: p, first, stale | difference in p")
        for run in RUN_STARTS:
            whole, part = prob.get((run, True), []), prob.get((run, False), [])
            if not whole or not part:
                print(f"{run:5d}-{run + 7:<6d} | no data")
                continue
            p0, p1 = statistics.mean(whole), statistics.mean(part)
            h0, h1 = statistics.mean(hit[(run, True)]), statistics.mean(hit[(run, False)])
            s1 = statistics.mean(stale[(run, False)])
            print(f"{run:5d}-{run + 7:<6d} | {p0:14.3f}, {100 * h0:3.0f}% | {p1:13.3f}, {100 * h1:3.0f}%, {100 * s1:3.0f}% "
                  f"| {p1 - p0:+.3f}")
    print()
    print("Lengths 8 to 2,047 are less than 2,048 tokens; 2,056 and more are the control.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base-url", default="http://127.0.0.1:8000")
    r.add_argument("--model", default="glm-5.3-flash")
    r.add_argument("--out", required=True)
    r.add_argument("--parallel", type=int, default=4)
    s = sub.add_parser("summary")
    s.add_argument("files", nargs="+")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        summary(args.files)


if __name__ == "__main__":
    sys.exit(main())
