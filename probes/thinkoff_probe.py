"""What does "thinking off" cost or gain on an engine? One stream per prompt, greedy, 400 forced tokens.

sparkDash's decode benchmark asks for thinking off through chat_template_kwargs. TensorFold honours that by rendering
an empty think block with no effort line. The stock GLM-5.3 chat template that vLLM uses has no such switch: the
prompt is unchanged, the model still thinks, and only the reasoning parser is turned off. This probe measures both
forms on one server:

  chat-flags : the chat request sparkDash sends (whatever the server makes of the flags);
  raw-off    : a raw completion whose prompt is TensorFold's thinking-off rendering, token for token.

Decode tok/s is sparkDash's formula: (completion tokens - 1) / (last token time - first token time).

  python3 thinkoff_probe.py --base-url http://127.0.0.1:8801 --out thinkoff.jsonl
"""
import argparse
import json
import time
import urllib.request

PROMPTS = {
    "prose": "Write a detailed step-by-step explanation of how a hash map works, including collision handling, "
             "resizing, and time complexity. Be thorough.",
    "code": "binary_search\ndef binary_search(nums, target) -> int: index of target in a sorted list, or -1.\n"
            "Output only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers "
            "this needs. Keep writing code.",
    "structured": "Count from 1 to 200. Output only the numbers, separated by spaces. No other text.",
}
OFF_FLAGS = {"enable_thinking": False, "thinking": False, "thinking_mode": "disabled"}


def stream(base, path, body):
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    started, first, last, usage, text = time.perf_counter(), None, None, None, ""
    with urllib.request.urlopen(request, timeout=600) as reply:
        for raw in reply:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            piece = choice.get("text") or "".join(delta.get(k) or "" for k in ("content", "reasoning_content", "reasoning"))
            if piece:
                now = time.perf_counter()
                first = first if first is not None else now
                last = now
                text += piece
            if chunk.get("usage"):
                usage = chunk["usage"]
    tokens = (usage or {}).get("completion_tokens") or 0
    tps = (tokens - 1) / (last - first) if first is not None and last > first else None
    return {"ttft_s": round(first - started, 3) if first else None, "completion_tokens": tokens,
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "decode_tps": round(tps, 1) if tps else None,
            "head": text[:140], "think_close_at_char": text.find("</think>")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8801")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--tokens", type=int, default=400)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    common = {"model": args.model, "max_tokens": args.tokens, "min_tokens": args.tokens, "ignore_eos": True,
              "temperature": 0, "top_p": 1, "stream": True, "stream_options": {"include_usage": True}}
    with open(args.out, "a") as out:
        for round_ in range(1, args.rounds + 1):
            for kind, prompt in PROMPTS.items():
                rows = {
                    "chat-flags": stream(args.base_url, "/v1/chat/completions",
                                         {**common, "messages": [{"role": "user", "content": prompt}],
                                          "chat_template_kwargs": OFF_FLAGS}),
                    "raw-off": stream(args.base_url, "/v1/completions",
                                      {**common, "add_special_tokens": False,
                                       "prompt": f"[gMASK]<sop><|user|>{prompt}<|assistant|><think></think>"}),
                }
                for form, row in rows.items():
                    row = {"round": round_, "type": kind, "form": form, **row}
                    out.write(json.dumps(row) + "\n")
                    print(json.dumps({k: row[k] for k in ("round", "type", "form", "decode_tps", "completion_tokens",
                                                          "prompt_tokens", "think_close_at_char")}), flush=True)


if __name__ == "__main__":
    main()
