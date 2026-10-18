# The data of the paper

`scripts/build_data.py` makes each CSV file in this directory from [`../raw/`](../raw/). Do not change a CSV file by
hand. Change a number only with a new measurement.

The run names are these:

| Run | Inference stack |
|---|---|
| `deepseek_1017_kk_16` | DeepSeek-V4.1-Flash-1017 on Karmic Kraken (beta of 16 October 2026), 16 slots, Engram tables in RAM |
| `deepseek_1017_kk_8` | The same server with 8 slots, round 1 |
| `deepseek_1017_kk_8_round2` | The same server with 8 slots, round 2 |
| `deepseek_1017_sglang` | DeepSeek-V4.1-Flash-1017 on SGLang with the community recipe, 8 slots, Engram tables in RAM (Appendix F) |
| `glm_r38_16_aligned` | GLM-5.3-Flash on Jovian Judgement r38, 16 slots, checkpoint policy `aligned` (the GLM paper, session 6) |
| `glm_r38_8` | GLM-5.3-Flash on Jovian Judgement r38, 8 slots (the GLM paper, session 6) |

| File | Test | Section |
|---|---|---|
| `decode_sparkdash.csv` | The decode test of sparkDash: 4 output types, 1 to 16 requests | 2, 9, Appendix C, F |
| `prefill_sparkdash.csv` | The prefill test of sparkDash: one cold prompt of 8K to 256K tokens | 3, Appendix F |
| `concurrent_waves.csv` | The greedy test, the long-context test with three sampler settings, and the long-context test with the context in a system message (the column `pass` numbers the passes of one setting) | 2, 4, 9 |
| `chat_reuse_ttft.csv` | The conversation test | 5 |
| `prefix_reuse_check.csv` | The prompt-reuse check | 5 |
| `cold_burst.csv` | Four cold requests with a shared prefix | 5 |
| `server_facts.csv` | The KV cache, the context, and the start time of the Karmic Kraken server, and the memory check of the Engram tables at its start, from its start log and its launcher | 6, Appendix A |
| `decode_log_summary.csv` | The accepted draft tokens for each step, from the log of each server | 6 |
| `quality_pi_tasks.csv`, `quality_pi_summary.csv`, `quality_pi_paired.csv`, `quality_pi_calls.csv` | The quality run through the pi agent | 7, Appendix D |
| `integrity.csv` | A passphrase in a long prompt | 8 |
| `batch_schema_check.csv` | Strict JSON schemas and forced tool calls in a batch with long chats | 8 |
| `more_qualification.csv`, `more_speed.csv`, `chat_reply_under_load.csv`, `api_feature_probes.csv` | The tests of the benchmark procedure for local models | Appendix D |
| `engram_ram_kk.csv` | The Engram tables in host memory on Karmic Kraken with 8 slots: the reads of the drive with the weights, the resident shared memory of each worker, and the locked memory of the host | Appendix A |
| `engram_residency.csv` | The resident part of the Engram tables on SGLang, at four points of its test | Appendix F |
| `single_request_recipe_matrix.csv` | The benchmark of the recipe with one request, on SGLang | Appendix F |

The GLM-5.3-Flash rows come from the raw files of the GLM paper at version 1.5, which this repository copies with no
change.
