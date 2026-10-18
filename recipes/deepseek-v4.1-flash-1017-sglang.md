# The DeepSeek-V4.1-Flash-1017 server on SGLang (Appendix F)

This file gives the SGLang server of Appendix F of the paper. Its tests ran on 17 October 2026, before the tests on
Karmic Kraken. The inference stack of the main results is in [`deepseek-v4.1-flash-1017.md`](deepseek-v4.1-flash-1017.md).

## The weights

| Item | Value |
|---|---|
| Model | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), MIT license |
| Revision | `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Files | 83 files, 510,311,045,295 bytes. We compared each file with the SHA-256 (LFS files) or the Git blob ID (other files) that the Hugging Face API gives for this revision. |
| Precision | As published: the dense layers in FP8, the routed experts in FP4, the embeddings and the vision encoder in BF16 |
| Engram tables | Shards 47 and 48, 189.1 GiB, for layers 1 and 14 |

The weights are new for each of the 48 shards. The configuration, the tensor index and the tokenizer are the same as
in the September release of DeepSeek-V4.1-Flash. The prompt encoding (`encoding/encoding.py`) and the reference
code (`inference/model.py`) are different. Refer to "Known differences" below.

## The engine

The engine is SGLang with the community recipe for four RTX PRO 6000 GPUs:
[`sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000`](https://github.com/sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000).

- The base image is `lmsysorg/sglang@sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5`.
- The recipe code is commit `45f538a`. Commit `ce69105` of 17 October changes only the model name and the revision
  that the recipe downloads. It does not change the server.
- Our image `dsv41-sglang:20260927-rowread` (image ID `sha256:dccb3cda381e…`) adds three changes to the recipe:
  1. The fix of recipe issue #1: the attention kernel masks for prompts of more than 64 tokens.
  2. A fix of the Responses API path for this model.
  3. A batched reader for Engram rows on NVMe. The RAM mode of this test does not use it.
- The image checks the weights against a list of files at the start. For this test, the list of the 1017 files
  replaces the list in the image (a read-only file mount). We did not build a new image.

## The settings of the server

| Setting | Value |
|---|---|
| Parallelism | 4 GPUs: tensor parallel 4, expert parallel 4 |
| Attention backend | `dsv4` |
| Expert kernels | `flashinfer_mxfp4` |
| Speculative decode | DSpark, block size 5 |
| Requests at the same time | 8 (CUDA graphs for decode up to 8) |
| Context | 409,600 tokens for each request |
| KV cache | up to 4,200,000 tokens. The server made 1,829,888 tokens with a memory fraction of 0.85. |
| Prompt chunk | 2,048 tokens |
| Engram tables | `OFFLOAD_MODE=ram`: the server maps the two table files and locks them in host memory |
| Container memory limit | 240 GiB, memory lock not limited, capability `IPC_LOCK` |
| Parsers | tool calls `deepseekv41`, reasoning `deepseek-v41` |
| Address | `127.0.0.1:8010`, with an API key |
| Served model name | `deepseek-v4.1-flash` |

## The host

The host is the host of the GLM-5.3-Flash paper, with more memory:

- Four NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs with 96 GB each, at a power limit of 250 W, on PCIe with no NVLink.
- An AMD Threadripper PRO 7975WX CPU and 377 GiB of RAM (DDR5 at 5,200 MT/s, ECC).
- NVIDIA driver 615.71.09, Linux kernel 7.0.0-38, Docker 29.8.2, IOMMU in pass-through mode.

## How the test tools reach the server

The tools of the GLM-5.3-Flash paper send requests with the model name `glm-5.3-flash`, with the thinking switch of
GLM, and with no API key. We did not change the tools. A small adapter
([`tools/glm_api_adapter.py`](../tools/glm_api_adapter.py)) listens on `127.0.0.1:8020` and sends each request to the
server. The adapter does these changes and no other changes:

1. It adds the API key of the server.
2. It changes the model name `glm-5.3-flash` to `deepseek-v4.1-flash`.
3. It changes `chat_template_kwargs.enable_thinking` to `chat_template_kwargs.thinking`. GLM thinks if a request does
   not say otherwise. Thus a chat request with no thinking switch gets `thinking: true`.
4. It adds the name `glm-5.3-flash` to the model list of `GET /v1/models`.

`reasoning_effort` does not change. DeepSeek accepts `low`, `high`, `xhigh` and `max`. Answers and streams go
through the adapter with no change.

## Known differences

- SGLang makes the prompt with its own copy of the DeepSeek-V4.1 encoding. That copy is older than the encoding of
  the 1017 release. The 1017 encoding changes how a message with only tool results starts a turn, and how a tool
  result with content blocks is shown.
- The reference code of the 1017 release rounds the input of the projection `wo_a` to FP8 and back, and it has a
  fix of the indexer cache. SGLang computes `wo_a` in BF16 (the start log shows the conversion of its weight) with no
  such rounding, and our image has neither change.
- The locked tables did not stay fully in memory. At the end of the test, a part of the 189.1 GiB was not resident,
  and the server read those rows from the disk when it needed them. The results show the measured values.
