"""Prompt reuse over the chat API, for engines without raw /v1/completions (TensorFold) and for vLLM alike.

For each size: a cold request, the identical request again, the same conversation plus a new user turn, and the
agent pattern (the model's own reply appended, then a new user turn). Time to first token is measured with
thinking off and 16 output tokens, greedy, so it is prompt processing plus one step. Fresh random text per size
and round keeps the cold request cold.

  python3 warm_chat.py --base-url http://127.0.0.1:8801 --out warm-chat.jsonl
"""
import argparse
import json
import random
import time
import urllib.request

WORDS = ("river stone signal harbor lantern copper meadow engine ledger orbit canvas thistle quartz beacon "
         "willow saddle prism falcon anchor garnet summit cobalt ember glacier velvet compass tundra").split()
NO_THINK = {"thinking": False, "enable_thinking": False}


def post(base, path, body, timeout=900):
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=timeout)


def words(seed, n):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) + str(rng.randrange(1000)) for _ in range(n))


def ttft(base, model, messages):
    body = {"model": model, "messages": messages, "max_tokens": 16, "temperature": 0, "top_p": 1.0, "top_k": -1,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": NO_THINK}
    started, first, usage = time.perf_counter(), None, None
    with post(base, "/v1/chat/completions", body) as reply:
        for raw in reply:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            event = json.loads(line[5:])
            delta = (event.get("choices") or [{}])[0].get("delta") or {}
            if first is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")):
                first = time.perf_counter() - started
            if event.get("usage"):
                usage = event["usage"]
    return first, usage


def reply_text(base, model, messages, tokens):
    body = {"model": model, "messages": messages, "max_tokens": tokens, "temperature": 0, "top_p": 1.0,
            "top_k": -1, "chat_template_kwargs": NO_THINK}
    with post(base, "/v1/chat/completions", body) as reply:
        return json.load(reply)["choices"][0]["message"].get("content") or ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8801")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--sizes", default="9000,56000", help="approximate prompt tokens (about 3 words a token)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.out, "a") as out:
        for size in (int(s) for s in args.sizes.split(",")):
            for round_ in (1, 2):
                base = [{"role": "user", "content": "Notes:\n" + words(size * 100 + round_, size // 2) +
                         "\n\nSummarise the notes in one sentence."}]
                turn = {"role": "user", "content": "Now list three words from the notes. " + words(round_ + 7, 300)}
                rows = []
                first, usage = ttft(args.base_url, args.model, base)
                rows.append(("cold", first, usage))
                first, usage = ttft(args.base_url, args.model, base)
                rows.append(("identical resend", first, usage))
                first, usage = ttft(args.base_url, args.model, base + [turn])
                rows.append(("resend + new turn", first, usage))
                answer = reply_text(args.base_url, args.model, base, 400)
                first, usage = ttft(args.base_url, args.model, base + [{"role": "assistant", "content": answer}, turn])
                rows.append(("own reply + new turn", first, usage))
                for case, first_s, use in rows:
                    row = {"size": size, "round": round_, "case": case,
                           "ttft_s": round(first_s, 3) if first_s else None,
                           "prompt_tokens": (use or {}).get("prompt_tokens"),
                           "cached_tokens": ((use or {}).get("prompt_tokens_details") or {}).get("cached_tokens")}
                    print(json.dumps(row), flush=True)
                    out.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
