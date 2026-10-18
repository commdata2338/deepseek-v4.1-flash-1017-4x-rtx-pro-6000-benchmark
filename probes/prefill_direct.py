"""Prefill speed of one cold prompt, measured directly: prompt tokens divided by the time to the first token.

One request for each size, with new random text each time, so that a server cannot have the prompt in its cache.
The prompt size comes from the usage that the server reports. Thinking is off and the answer is 8 tokens.

  python3 prefill_direct.py --base-url http://127.0.0.1:8020 --out prefill-direct.jsonl

The paper used this probe to compare two prefill chunk sizes of TensorFold modified (Appendix E.5). The prefill
numbers of the other sections come from sparkDash, which uses a different prompt text. Compare the results of this
probe only with other results of this probe.
"""
import argparse
import json
import random
import time
import urllib.request

WORDS = ("river stone signal harbor lantern copper meadow engine ledger orbit canvas thistle quartz beacon "
         "willow saddle prism falcon anchor garnet summit cobalt ember glacier velvet compass tundra").split()


def measure(base, model, size):
    rng = random.Random(time.time_ns())
    text = " ".join(rng.choice(WORDS) + str(rng.randrange(1000)) for _ in range(size // 3))
    body = {"model": model, "max_tokens": 8, "temperature": 0, "top_p": 1.0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
            "messages": [{"role": "user", "content": f"Nonce {rng.randrange(10 ** 12)}.\n" + text + "\n\nReply OK."}]}
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    started, first, usage = time.perf_counter(), None, {}
    with urllib.request.urlopen(request, timeout=1800) as reply:
        for raw in reply:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            delta = (event["choices"][0].get("delta") or {}) if event.get("choices") else {}
            if first is None and delta.get("content") is not None:
                first = time.perf_counter() - started
            usage = event.get("usage") or usage
    tokens = usage.get("prompt_tokens")
    return {"target_tokens": size, "prompt_tokens": tokens, "ttft_s": round(first or 0, 3),
            "prefill_tok_s": round(tokens / first, 1) if tokens and first else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8020")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--sizes", default="8192,16384,32768,65536,131072,262144")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.out, "a") as out:
        for size in (int(s) for s in args.sizes.split(",")):
            row = measure(args.base_url, args.model, size)
            out.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
