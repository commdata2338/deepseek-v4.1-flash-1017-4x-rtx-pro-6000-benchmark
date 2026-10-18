"""Concurrent decode over a shared long context.

One request primes the prefix cache with a shared context (Python standard-library source, about 56K tokens by
default). Then C requests, each the context plus a short distinct question, start together, read only their new
tokens, and decode side by side. Total decode tok/s is taken over the window in which every stream is decoding.
Output length is forced with ignore_eos, so this measures speed, not quality.

Token counts over time come from per-chunk usage where the engine streams it (vLLM). An engine that reports usage
only on the last chunk (TensorFold) gets its streamed text scaled to the final count.

  python3 longctx.py --base-url http://127.0.0.1:8801 --concurrency 4,8,16 --raw raw.jsonl \
      --temperature 1.0 --top-p 0.95 --top-k -1 --thinking-on max
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path

TOPICS = ["error handling", "naming", "concurrency", "performance", "API design", "testing", "documentation",
          "security", "resource cleanup", "edge cases"]
LIBRARY = Path("/usr/lib/python3.12")
PACKAGES = ["asyncio", "email", "http", "json", "logging", "concurrent/futures", "importlib", "unittest"]


def post(base, path, body, timeout=900):
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=timeout)


def count(base, model, text):
    with post(base, "/tokenize", {"model": model, "prompt": text}) as reply:
        return json.load(reply)["count"]


def shared_context(base, model, target):
    text = ""
    for package in PACKAGES:
        for path in sorted((LIBRARY / package).glob("*.py")):
            text += f"### File: {path.relative_to(LIBRARY)}\n{path.read_text(errors='replace')}\n"
            if len(text) > target * 3 and count(base, model, text) >= target:
                break
        else:
            continue
        break
    tokens = count(base, model, text)
    while tokens > target * 1.01:  # trim proportionally, keeping whole lines
        text = text[:int(len(text) * target / tokens)].rsplit("\n", 1)[0] + "\n"
        tokens = count(base, model, text)
    return text, tokens


SYSTEM_CONTEXT = False   # --context-as-system: the shared context is a system message, the task is the user message


def messages(context, task):
    if SYSTEM_CONTEXT:
        return [{"role": "system", "content": context}, {"role": "user", "content": task}]
    return [{"role": "user", "content": context + "\n\n---\n" + task}]


def stream(base, model, content, max_tokens, origin, out, extra=None):
    body = {"model": model, "messages": content, "max_tokens": max_tokens,
            "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}, **(extra or {})}
    events, usage, text_events, chars, engine_stats = [], None, [], 0, None
    started = time.monotonic() - origin
    with post(base, "/v1/chat/completions", body) as reply:
        for raw in reply:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            now = time.monotonic() - origin
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            piece = "".join(delta.get(k) or "" for k in ("content", "reasoning_content", "reasoning"))
            if piece:
                chars += len(piece)
                text_events.append((now, chars))
            if chunk.get("tensorfold"):  # TensorFold's own per-reply statistics, on the last chunk
                engine_stats = chunk["tensorfold"]
            if chunk.get("usage"):
                usage = chunk["usage"]
                if usage.get("completion_tokens"):
                    events.append((now, usage["completion_tokens"]))
    # Engines without per-chunk usage (TensorFold) report only the final count: scale the streamed text to it.
    method = "usage"
    if len(events) < 3 and text_events and usage and usage.get("completion_tokens"):
        total = usage["completion_tokens"]
        events = [(time_, max(1, round(total * c / chars))) for time_, c in text_events]
        method = "text-scaled"
    out.update(started=started, events=events, usage=usage, count_method=method, engine_stats=engine_stats)


def at(events, t):
    """Completion tokens reached by time t, interpolated between chunks."""
    previous = (events[0][0], 0)
    for time_, tokens in events:
        if time_ >= t:
            span = time_ - previous[0]
            return previous[1] + (tokens - previous[1]) * ((t - previous[0]) / span if span > 0 else 1)
        previous = (time_, tokens)
    return events[-1][1]


def wave(base, model, context, concurrency, max_tokens, raw, extra=None):
    stamp = f"{time.time():.0f}"
    # The prime has the same shape as the wave's requests and generates a few tokens; a one-token prime left no
    # reusable cache on vLLM with this hybrid model, and the first wave then read the whole context again.
    prime = messages(context, f"Run {stamp}, request 0 of {concurrency}: review the code above for naming. "
                     "For each file, list concrete observations with line references. Be exhaustive.")
    with post(base, "/v1/chat/completions", {"model": model, "messages": prime,
                                            "max_tokens": 16, "ignore_eos": True, **(extra or {})}) as reply:
        json.load(reply)
    origin, origin_epoch = time.monotonic(), time.time()
    results = [{} for _ in range(concurrency)]
    threads = []
    for index in range(concurrency):
        topic = TOPICS[index % len(TOPICS)]
        content = messages(context, f"Run {stamp}, request {index + 1} of {concurrency}: review the code above for "
                           f"{topic}. For each file, list concrete observations with line references. Be exhaustive.")
        thread = threading.Thread(target=stream, args=(base, model, content, max_tokens, origin, results[index], extra))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    firsts = [r["events"][0][0] for r in results]
    ends = [r["events"][-1][0] for r in results]
    lo, hi = max(firsts), min(ends)
    row = {"concurrency": concurrency, "context_tokens": (results[0]["usage"] or {}).get("prompt_tokens"),
           "output_tokens": max_tokens, "median_ttft_s": round(statistics.median(firsts), 3)}
    if hi > lo + 1:
        rates = [(at(r["events"], hi) - at(r["events"], lo)) / (hi - lo) for r in results]
        row.update(status="measured", shared_decode_s=round(hi - lo, 2), aggregate_decode_tps=round(sum(rates), 1),
                   median_decode_tps=round(statistics.median(rates), 1))
    else:
        row.update(status="no_overlap")
    with open(raw, "a") as handle:
        handle.write(json.dumps({"row": row, "origin_epoch": origin_epoch, "streams": results}) + "\n")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8801")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--context-tokens", type=int, default=56000)
    parser.add_argument("--output-tokens", type=int, default=2000)
    parser.add_argument("--concurrency", default="1,2,4,6")
    parser.add_argument("--raw", required=True, help="JSONL file for per-stream events")
    parser.add_argument("--temperature", type=float, help="omit for the server default; 0 for greedy")
    parser.add_argument("--no-thinking", action="store_true", help="turn GLM thinking off for these requests")
    parser.add_argument("--thinking-on", metavar="EFFORT", help="ask for thinking explicitly at this reasoning effort "
                        "(for a server whose default is thinking off, such as TensorFold's recipe)")
    parser.add_argument("--top-p", type=float, help="send top_p explicitly (TensorFold and vLLM defaults differ)")
    parser.add_argument("--top-k", type=int, help="send top_k explicitly; -1 disables it")
    parser.add_argument("--context-as-system", action="store_true", help="send the shared context as a system "
                        "message and the task as the user message (the default is one user message)")
    args = parser.parse_args()
    globals()["SYSTEM_CONTEXT"] = args.context_as_system
    extra = {}
    if args.temperature is not None:
        extra["temperature"] = args.temperature
    if args.top_p is not None:
        extra["top_p"] = args.top_p
    if args.top_k is not None:
        extra["top_k"] = args.top_k
    if args.no_thinking:
        extra["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
    if args.thinking_on:
        extra["chat_template_kwargs"] = {"thinking": True, "enable_thinking": True, "reasoning_effort": args.thinking_on}
    context, tokens = shared_context(args.base_url, args.model, args.context_tokens)
    print(json.dumps({"shared_context_tokens": tokens}), flush=True)
    for concurrency in (int(value) for value in args.concurrency.split(",")):
        print(json.dumps(wave(args.base_url, args.model, context, concurrency, args.output_tokens, args.raw, extra)),
              flush=True)


if __name__ == "__main__":
    main()
