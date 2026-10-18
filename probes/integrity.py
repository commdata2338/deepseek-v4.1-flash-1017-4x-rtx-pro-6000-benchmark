"""Does the engine read short and long prompts correctly?

A passphrase is hidden about 60% of the way through filler text of roughly N tokens, and the model is asked for it
(thinking off, greedy). A benchmark of an engine that returns wrong text means nothing, and serving recipes have
been known to corrupt long prompts, so this runs before the speed tests.

  python3 integrity.py --base-url http://127.0.0.1:8020 --sizes 2000,30000,200000 --out integrity.jsonl

The answer is limited to 40 tokens. If a server writes a sentence before the passphrase and the limit cuts it, send
the test again with more room: --max-tokens 300. Each run makes new filler text and a new passphrase (the seed is
the time), so a second run is not the same prompt. The passphrase counts if it is in the answer or in the thoughts.

Exit code 0 only if every size passes.
"""
import argparse
import json
import random
import sys
import time
import urllib.request

WORDS = ("river stone signal harbor lantern copper meadow engine ledger orbit canvas thistle quartz beacon "
         "willow saddle prism falcon anchor garnet summit cobalt ember glacier velvet compass tundra").split()
NO_THINK = {"thinking": False, "enable_thinking": False}


def filler(rng, n):
    return " ".join(rng.choice(WORDS) + str(rng.randrange(1000)) for _ in range(n))


def ask(base, model, size, seed, max_tokens=40):
    rng = random.Random(seed)
    phrase = f"{rng.choice(WORDS)}-{rng.randrange(10000, 99999)}-{rng.choice(WORDS)}"
    words = size // 3  # about three tokens a word (measured: 1,000 words made 3,049 prompt tokens)
    text = (filler(rng, int(words * 0.6)) + f"\nThe passphrase is {phrase}.\n" + filler(rng, int(words * 0.4)))
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "top_p": 1.0, "chat_template_kwargs": NO_THINK,
            "messages": [{"role": "user", "content": "Notes:\n" + text + "\n\nWhat is the passphrase stated in the "
                          "notes above? Reply with the passphrase only."}]}
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as reply:
        result = json.load(reply)
    message = result["choices"][0]["message"]
    answer = (message.get("content") or "") + (message.get("reasoning_content") or "")
    return {"target_tokens": size, "prompt_tokens": result["usage"].get("prompt_tokens"),
            "completion_tokens": result["usage"].get("completion_tokens"), "max_tokens": max_tokens,
            "finish_reason": result["choices"][0].get("finish_reason"), "passphrase": phrase,
            "answer": answer.strip()[:120 if max_tokens <= 40 else 700], "ok": phrase in answer,
            "seconds": round(time.perf_counter() - started, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8801")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--sizes", default="2000,30000,200000")
    parser.add_argument("--max-tokens", type=int, default=40, help="limit for the answer (default 40)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    passed = True
    with open(args.out, "a") as out:
        for index, size in enumerate(int(s) for s in args.sizes.split(",")):
            row = ask(args.base_url, args.model, size, seed=int(time.time()) + index, max_tokens=args.max_tokens)
            passed &= row["ok"]
            out.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
