"""Check the benchmark chat template against the stock GLM-5.3 template.

With thinking on (the default, or enable_thinking true), every rendering must equal the stock template's.
With enable_thinking false, it must equal TensorFold's thinking-off rendering of the stock text: the
"Reasoning Effort: Max" line removed and the opened think block closed (tensorfold/families/glm5_next/prompts.py).

  python3 template_check.py <model dir> <patched template>
"""
import sys
from pathlib import Path

from transformers import AutoTokenizer

EFFORT_LINE = "<|system|>Reasoning Effort: Max"
OPENED = "<|assistant|><think>"


def tensorfold_off(text):
    text = text.replace(EFFORT_LINE, "", 1)
    return text + "</think>" if text.endswith(OPENED) else text


TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.",
                                           "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                          "required": ["path"]}}}]
CHATS = {
    "one turn": [{"role": "user", "content": "Say hello."}],
    "system + turn": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Say hello."}],
    "multi-turn with reasoning": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "Simple sum."},
        {"role": "user", "content": "And 3+3?"}],
    "tool call history": [
        {"role": "user", "content": "Read a.txt"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "read_file", "arguments": {"path": "a.txt"}}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "file body"},
        {"role": "user", "content": "Summarise it."}],
}


def main():
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
    patched = Path(sys.argv[2]).read_text()
    failures = 0
    for name, messages in CHATS.items():
        tools = TOOLS if "tool" in name else None
        for kwargs in ({}, {"enable_thinking": True}, {"reasoning_effort": "low"}, {"reasoning_effort": "high"},
                       {"enable_thinking": True, "reasoning_effort": "max"}, {"thinking": False}):
            stock = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True, **kwargs)
            mine = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True,
                                                 chat_template=patched, **kwargs)
            ok = stock == mine
            failures += not ok
            print(f"{'ok  ' if ok else 'FAIL'} thinking on  | {name} | {kwargs}")
        stock = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        for kwargs in ({"enable_thinking": False}, {"enable_thinking": False, "thinking": False, "thinking_mode": "disabled"}):
            mine = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True,
                                                 chat_template=patched, **kwargs)
            ok = mine == tensorfold_off(stock)
            failures += not ok
            print(f"{'ok  ' if ok else 'FAIL'} thinking off | {name} | tail {mine[-48:]!r}")
    print("FAILURES", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
