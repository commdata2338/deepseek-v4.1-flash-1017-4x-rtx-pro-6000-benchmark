# Test commands

`$URL` is the base URL of the engine: `http://127.0.0.1:8020` for TensorFold and `http://127.0.0.1:8801` for Jovian
Judgement and the official vLLM. Each engine serves the model with the name `glm-5.3-flash`. Do each test with no other load on the server.

## Before you start an engine

Do the GPU copy test for all 12 ordered pairs, with 1 GiB for each pair:

    docker run --rm --gpus all --network none -v $PWD/probes:/tools:ro \
      --entrypoint python <an image with CUDA PyTorch> /tools/p2p_test.py 1024
    journalctl -k --since "5 minutes ago" | grep -E 'IO_PAGE_FAULT|NVRM: Xid'

The second command must give no lines.

Compare a direct transfer with a staged transfer:

    docker run --rm --gpus '"device=0,1,2"' --network none -v $PWD/probes:/probes:ro \
      --entrypoint python <image> /probes/gpu_copy_paths.py 256

## For each engine, in this sequence

**1. Correctness.**

    python3 probes/integrity.py --base-url $URL --sizes 2000,30000,200000 --out integrity.jsonl

**2. Decode test and prefill test of sparkDash.** Start sparkDash v1.8.9 as a local server. Add the port of the
engine as an LLM port of a local unit. Then send these requests to the sparkDash API:

    POST /api/sparks/<unit>/llm/bench
      {"port": <port>, "modelId": "glm-5.3-flash", "concurrencies": [1, 2, 4, 8, 16],
       "maxTokens": 400, "promptType": "prose" | "code" | "structured" | "json"}
    POST /api/sparks/<unit>/llm/prefill-bench
      {"port": <port>, "modelId": "glm-5.3-flash",
       "contextSizes": [8192, 16384, 32768, 65536, 131072, 262144]}

Poll the same path with `GET ...?port=<port>` until the job is not active. Keep the JSON that the API gives, in the
files `sparkdash-decode-<type>.json` and `sparkdash-prefill.json`.

**3. Thinking-off prompt.** This probe shows if the two engines get the same thinking-off prompt:

    python3 probes/thinkoff_probe.py --base-url $URL --out thinkoff.jsonl

The chat form and the raw form must show the same prompt token count for each prompt type (33, 60, and 27 in our
tests). On Jovian Judgement with the standard template, the chat form shows 6 more tokens. These tokens are the "Reasoning
Effort" line. The output then starts with the thoughts of the model.

**4. Greedy waves with a short context.**

    python3 probes/longctx.py --base-url $URL --concurrency 1,8,12,16 --context-tokens 1000 \
      --output-tokens 400 --temperature 0 --top-p 1.0 --top-k -1 --no-thinking --raw greedy-raw.jsonl > greedy.log

**5. Reuse of a conversation.**

    python3 probes/warm_chat.py --base-url $URL --sizes 9000,56000 --out warm-chat.jsonl

The sizes are nominal. The prompts had 13.5K and 83.6K tokens.

**6. Long shared context with thinking on.**

    python3 probes/longctx.py --base-url $URL --concurrency 4,8,16 --output-tokens 2000 \
      --temperature 1.0 --top-p 0.95 --top-k -1 --thinking-on max --raw longctx-p95-raw.jsonl > longctx-sampled-p95.log
    python3 probes/longctx.py --base-url $URL --concurrency 4,8,16 --output-tokens 2000 \
      --temperature 1.0 --top-p 0.95 --top-k 20 --thinking-on max --raw longctx-topk20-raw.jsonl > longctx-p95-topk20.log
    python3 probes/longctx.py --base-url $URL --concurrency 4,8,12,16 --output-tokens 2000 \
      --temperature 1.0 --top-p 1.0 --top-k -1 --thinking-on max --raw longctx-raw.jsonl > longctx-sampled.log

With `--top-p 1.0`, TensorFold decodes this test at approximately 10 tokens/s for each stream. The probe prints one
JSON line for each number of concurrent requests. `scripts/build_data.py` reads these lines from the `.log` files.
The `--raw` file contains the events of each stream.

**7. Sampler settings (TensorFold only).** Use four streams, a 1K context, and 400 tokens. Do one run for each
group of settings:

    python3 probes/longctx.py --base-url $URL --concurrency 4 --context-tokens 1000 --output-tokens 400 \
      --raw isolate-raw.jsonl <settings> >> isolate.log

These are the groups of settings:

    --temperature 0   --top-p 1.0  --top-k -1 --no-thinking
    --temperature 1.0 --top-p 1.0  --top-k -1 --no-thinking
    --temperature 0   --top-p 1.0  --top-k -1 --thinking-on max
    --temperature 1.0 --top-p 1.0  --top-k -1 --thinking-on max
    --temperature 1.0 --top-p 0.95 --top-k -1 --thinking-on max
    --temperature 1.0 --top-p 0.99 --top-k -1 --thinking-on max
    --temperature 1.0 --top-p 1.0  --top-k 50 --thinking-on max
    --temperature 1.0                          --thinking-on max

**8. Reuse of a shared context.** Do this check directly after the start of the server, before a different test
puts the context into the cache:

    python3 probes/prefix_reuse_check.py cache-check.jsonl <label> user --base-url $URL
    python3 probes/prefix_reuse_check.py cache-check-system.jsonl <label> system --base-url $URL

The first form puts the 56K context and the question in one user message. The second form puts the context in a
system message. A step that uses as much time as the first step did the prefill of the context again.

For the long-context test with the context in a system message, add `--context-as-system` to a command of step 6:

    python3 probes/longctx.py --base-url $URL --concurrency 4,8,16 --output-tokens 2000 --temperature 1.0 \
      --top-p 0.95 --top-k 20 --thinking-on max --context-as-system \
      --raw longctx-system-p95-topk20-raw.jsonl > longctx-system-p95-topk20.log

**9. Four cold requests with a shared prefix (the vLLM inference stacks).** [`tensorfold-modified.md`](tensorfold-modified.md) gives
this test for the two TensorFold inference stacks. A vLLM server does not accept the priority field as text, and it gives no
token ids. Make the requests with a new first line for each start of a server, and record the times only:

    python3 probes/burst_reuse.py make --workload burst --lines 2800 --streams 4 --output-tokens 128 \
      --temperature 1 --top-k 20 --top-p .95 --thinking-on max --tag B --no-priority --output burst-requests-vllm.json
    python3 probes/burst_reuse.py run --fixtures burst-requests-vllm.json --base-url $URL --concurrency 4 \
      --no-token-ids --output burst-<server>.jsonl

**10. The last tokens of a short prompt.** This probe is for the change of Appendix C.9 of the paper. It sends 1,080
prompts of 8 to 6,007 tokens as token ids:

    python3 probes/short_context.py run --base-url $URL --out shortctx.jsonl > shortctx.log
    python3 probes/short_context.py summary shortctx.jsonl <the file of a different server>

## The check of the chat template

    python3 probes/template_check.py <checkpoint dir> <patched template>

The script uses `transformers`. We ran it in the image of Jovian Judgement with no GPU and no network.

## What each probe sends

| Probe | Thinking | Sampler settings | Output length |
|---|---|---|---|
| sparkDash decode test | off, through `chat_template_kwargs` | temperature 0, top_p 1 | 400 tokens, `min_tokens` = `max_tokens`, `ignore_eos` |
| sparkDash prefill test | off | not applicable: the test measures only the first token | 8 tokens |
| `longctx.py` | `--no-thinking` sends `enable_thinking: false` and `thinking: false`. `--thinking-on max` sends `enable_thinking: true` and `reasoning_effort: max`. | from the command line | fixed with `ignore_eos` |
| `warm_chat.py` | off | temperature 0 | 16 tokens |
| `prefix_reuse_check.py` | the default of the server | temperature 0 | 16 tokens, and 300 tokens in one step, fixed with `ignore_eos` |
| `integrity.py` | off | temperature 0 | 40 tokens maximum |
| `short_context.py` | not applicable: the prompt is a list of token ids | temperature 0 | 1 token |
| `thinkoff_probe.py` | off, in two forms | temperature 0, top_p 1 | 400 tokens, fixed |

TensorFold does not use `min_tokens`, and it obeys `ignore_eos`. It reads `top_k: -1` as "no top_k". If a request
gives no `top_k`, TensorFold uses top_k 20. The last group of settings in step 7 gets this default.

## Tests of TensorFold modified, and the quality run

[`tensorfold-modified.md`](tensorfold-modified.md) gives the equality tests and the speed tests of TensorFold modified. Do the
equality tests before the speed tests.

[`../quality/README.md`](../quality/README.md) gives the commands of the quality run through the pi agent. Use the
same commands and the same images for each engine.

## More steps for the official vLLM

**Second run of sparkDash.** Do step 2 a second time on the same server. In our tests, the first run was slower
in some cells at 8 and 16 streams. We did not find the cause.

**Passphrase with more output.** If the 200K case of `integrity.py` gives no passphrase, send the same type of
prompt with a limit of 300 tokens. Then make sure that the answer contains the passphrase:

    python3 probes/integrity.py --base-url $URL --sizes 200000,200000 --max-tokens 300 --out integrity-recheck.jsonl

The probe makes a new prompt for each run. Thus this is not the same prompt as in step 1.

## The files that `scripts/build_data.py` reads

Put the output of each server into one directory. `scripts/build_data.py` reads these file names. The table in
`scripts/collect_raw.py` gives the directory names of our runs.

| File | Test |
|---|---|
| `integrity.jsonl`, `integrity-recheck.jsonl` | step 1, and the passphrase with more output |
| `sparkdash-decode-prose.json`, `-code.json`, `-structured.json`, `-json.json`, `sparkdash-prefill.json` | step 2 |
| `thinkoff.jsonl` | step 3 |
| `greedy.log` | step 4 |
| `warm-chat.jsonl` | step 5 |
| `longctx-sampled-p95.log`, `longctx-sampled.log` | step 6 with no top_k: top_p 0.95 and top_p 1.0 |
| `longctx-p95-topk20.log`, `longctx-p95-topk-off.log`, `longctx-p95-topk20-pass2.log` | step 6 in the third session and in the fifth session |
| `cache-check.jsonl`, `cache-check-system.jsonl` | step 8, the two forms of the check |
| `longctx-system-p95-topk20.log`, `longctx-system-p95-topk-off.log`, `longctx-system-p1.log`, `longctx-system-p95-topk20-pass2.log` | step 8, the long-context test with the context in a system message |
| `isolate.log` | step 7 |
| `rank0.log` | the request log of a TensorFold server, for the time of a round |

The directories of TensorFold modified and of the quality run have other file names.
[`tensorfold-modified.md`](tensorfold-modified.md) and [`../quality/README.md`](../quality/README.md) give them.
