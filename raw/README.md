# The raw output of the tests

`scripts/collect_raw.py` copied these files from the result folders of the test host. It replaced host paths with
placeholders such as `<results-dir>`, and it did not keep a file that still had a private detail
([`NOT_INCLUDED.txt`](NOT_INCLUDED.txt) lists such files: there are none). Before publication, the same rules also
replaced the API key of the server with `<removed>` (in the `server-info.json` files of SGLang) and the serial UUID
of each GPU with `<gpu-uuid>` (in the `resources.jsonl` files). No result changed.

| Directory | Contents |
|---|---|
| `deepseek-v4.1-flash-1017/16-slots/` | The tests of the GLM paper on DeepSeek-V4.1-Flash-1017 on Karmic Kraken with 16 slots: the prompt-reuse check, the long-context test with top_k 20, off, and 20 again, the speed suite, and the test with the context in a system message |
| `deepseek-v4.1-flash-1017/8-slots/` | The same server with 8 slots, round 1: the prompt-reuse check, the speed suite, the long-context test with top_k 20, and the test with the context in a system message |
| `deepseek-v4.1-flash-1017/8-slots-round2/` | Round 2 with 8 slots: the speed suite, the long-context test with top_k 20, off, and 20 again, and the test with the context in a system message with top_k 20 and off |
| `deepseek-v4.1-flash-1017/cold-requests/` | The test with four cold requests, with 16 and with 8 slots, and its requests |
| `deepseek-v4.1-flash-1017/more-tests/` | The tests of the benchmark procedure for local models (Appendix D): the summary files only |
| `deepseek-v4.1-flash-1017/` (files) | The command and the environment that the launcher made, the numbers from the start log and the draft lines of the server log (no request text), the batch test with strict JSON schemas, the first checks, the check of the weights against the hashes of Hugging Face, and the memory and GPU values of every 5 seconds |
| `deepseek-v4.1-flash-1017-sglang/` | The same weights on SGLang (Appendix F): the speed suite, the prompt-reuse check, the long-context passes, the start log, the resident part of the Engram tables, the numbers from the server log, the cold requests, and the tests of the benchmark procedure |
| `glm-5.3-flash-r38/` | The raw files of release r38 of Jovian Judgement from the GLM paper, version 1.5, with no change |
| `quality-pi/` | The quality runs through the pi agent: the records, the files that the agent wrote, the grades, and the relay log of each run |

The per-request files of the `more-tests/` directories stay on the test host. Each summary file gives the values of
its test.
