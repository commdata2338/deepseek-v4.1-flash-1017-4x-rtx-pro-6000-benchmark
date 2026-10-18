"""Prompt reuse: does a second request find the context that an earlier request computed?

  python3 probes/prefix_reuse_check.py OUT.jsonl [LABEL] [user|system] [--base-url http://127.0.0.1:8801]

The probe sends one request at a time, with no other load on the server. Each request has the shared context of the
long-context test (about 56K tokens) and a short question.

user (the default): the context and the question are one user message.

  1. the context and a question; 16 output tokens
  2. the same request again
  3. the same context and a different question
  4. the request of step 1 with 300 output tokens
  5. the request of step 1 again
  6. the same context and a third question

system: the context is a system message, and the question is the user message. The probe sends steps 1, 2, 3, and 6.

For each step, the probe records the seconds to the full reply. A step that takes as long as step 1 computed the
full context again. A step that takes much less time reused the context.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import longctx  # noqa: E402 - the long-context probe of this folder builds the shared context


def ask(base, model, messages, max_tokens):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "ignore_eos": True, "temperature": 0}
    request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=900) as reply:
        usage = json.load(reply).get("usage") or {}
    return {"seconds": round(time.monotonic() - started, 2), "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("label", nargs="?", default="check")
    parser.add_argument("place", nargs="?", default="user", choices=("user", "system"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8801")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--context-tokens", type=int, default=56000)
    args = parser.parse_args()
    context, tokens = longctx.shared_context(args.base_url, args.model, args.context_tokens)
    # A new stamp for each run. On a server that reuses a shared context, an earlier run makes step 1 fast:
    # start the server again before the check.
    stamp = f"{time.time():.0f}"

    def messages(topic):
        question = f"Check {stamp} {args.label}: review the code above for {topic}. Give three observations."
        if args.place == "system":
            return [{"role": "system", "content": context}, {"role": "user", "content": question}]
        return [{"role": "user", "content": context + "\n\n---\n" + question}]

    first, other, third = messages("naming"), messages("error handling"), messages("tests")
    steps = [("first request, 16 tokens", first, 16), ("the same request again", first, 16),
             ("same context, other question", other, 16)]
    if args.place == "user":
        steps += [("the same request, 300 tokens", first, 300), ("the same request again, after 300 tokens", first, 16)]
    steps.append(("same context, third question", third, 16))
    with open(args.out, "a") as handle:
        for name, content, max_tokens in steps:
            row = {"label": args.label, "context_in": f"{args.place} message", "step": name,
                   "shared_context_tokens": tokens, **ask(args.base_url, args.model, content, max_tokens)}
            handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
