# The DeepSeek-V4.1-Flash-1017 server on Karmic Kraken

This file gives the server that we measured. The tests ran on 17 October 2026. Appendix F of the paper uses the
same weights on SGLang: [`deepseek-v4.1-flash-1017-sglang.md`](deepseek-v4.1-flash-1017-sglang.md).

## The weights

| Item | Value |
|---|---|
| Model | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), MIT license |
| Revision | `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Files | 83 files, 510,311,045,295 bytes. We compared each file with the SHA-256 (LFS files) or the Git blob ID (other files) that the Hugging Face API gives for this revision. |
| Precision | As published: the dense layers in FP8, the routed experts in FP4, the embeddings and the vision encoder in BF16 |
| Engram tables | Shards 47 and 48, 189.1 GiB, for layers 1 and 14 |

## The engine

The engine is Karmic Kraken, the vLLM fork of [Local Inference Lab](https://github.com/local-inference-lab), in its
image for Blackwell GPUs:

- Image: `ghcr.io/local-inference-lab/vllm@sha256:f5f9efe1b10ec7b853a495e66b7dbf15ff54f62741e444dea0b44e16736a10a9`,
  the tag `karmic-kraken-beta-20261016.2`. On 17 October 2026, the tag `karmic-kraken-beta` also had this digest.
- The vLLM fork at commit `6d66eed3`.
- The launcher of the image with the profile `ds41-flash` and the hardware profile `rtx-pro-6000-pcie`.

## The start command

The launcher makes the vLLM command from the profile and the options. These are the options that we gave. The
directory `raw/deepseek-v4.1-flash-1017/` contains the full command and the environment that the launcher made
(`launcher-config-16-slots.json` and `launcher-config-8-slots.json`).

```bash
docker run --rm --init --name ds41 --gpus all --ipc=host --shm-size=32g \
  --memory=265g --memory-swap=265g --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  --security-opt seccomp=<a profile that allows io_uring> \
  -p 127.0.0.1:8010:8000 \
  --mount type=bind,src=<weights-dir>,dst=/models/DeepSeek-V4.1-Flash-1017,readonly \
  --mount type=bind,src=<cache-dir>,dst=/cache \
  --env VLLM_API_KEY=<key> --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  ghcr.io/local-inference-lab/vllm@sha256:f5f9efe1b10ec7b853a495e66b7dbf15ff54f62741e444dea0b44e16736a10a9 \
  --profile ds41-flash --hardware rtx-pro-6000-pcie \
  --checkpoint original --model /models/DeepSeek-V4.1-Flash-1017 \
  --served-model-name deepseek-v4.1-flash --tensor-parallel-size 4 --max-num-seqs 16 \
  --gpu-memory-utilization 0.93 --engram-table-memory ram \
  --override-generation-config '{"temperature":1.0,"top_p":1.0}' --cache-mode vram \
  --env B12X_COMPILE_MEMORY_CACHE_SIZE=256 \
  --host 0.0.0.0 --port 8000
```

- For the quality run and the second round, `--max-num-seqs 8`.
- The guide of Local Inference Lab uses `--security-opt seccomp=unconfined`. We used the default profile of Docker with the
  `io_uring` calls added.
- `B12X_COMPILE_MEMORY_CACHE_SIZE=256` (the default is 1024) did not change the memory during the tuning. It is not
  necessary.
- Keep the directory for `/cache` between starts. It holds the results of the tuning of the kernels (Appendix A of
  the paper).

## The settings that the launcher resolved

| Setting | Value |
|---|---|
| Parallelism | 4 GPUs, tensor parallel 4 |
| Kernels | B12X for attention, the experts, and the dense layers |
| Speculative decode | DSpark, 7 draft tokens, adaptive verification, probabilistic draft sampling |
| Requests at the same time | 16, or 8 |
| Context | 1,048,576 tokens for each request (`--max-model-len -1`, the context of the model) |
| Batched tokens | 4,096 |
| KV cache | FP8, all GPU memory that the weights and the CUDA graphs do not use (0.93 of each GPU) |
| CUDA graphs | full and piecewise, up to 128 tokens in one step |
| Engram tables | in host memory (`table_memory: ram`) |
| Prefix cache | on, in GPU memory only (`--cache-mode vram`) |
| All-reduce | the PCIe all-reduce of B12X, one-shot up to 96 KB |
| Chat template, parsers | `deepseek_v41` for the tokenizer mode, the tool calls, and the reasoning |
| Thinking default | on, effort `high` (the profile default) |
| Sampler defaults | temperature 1.0, top_p 1.0 (our change; the profile default is top_p 0.95) |

## The host

The host is the host of the GLM-5.3-Flash paper, with more memory:

- Four NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs with 96 GB each, at a power limit of 250 W, on PCIe with no NVLink.
- An AMD Threadripper PRO 7975WX CPU and 377 GiB of RAM (DDR5 at 5,200 MT/s, ECC).
- NVIDIA driver 615.71.09, Linux kernel 7.0.0-38, Docker 29.8.2, IOMMU in pass-through mode.

## How the test tools reach the server

The tools of the GLM-5.3-Flash paper send requests with the model name `glm-5.3-flash`, with the thinking switch of
GLM, and with no API key. We did not change the tools. A small adapter
([`tools/glm_api_adapter.py`](../tools/glm_api_adapter.py)) listens on `127.0.0.1:8020` and sends each request to the
server. Appendix B of the paper gives each change that the adapter makes.

## Known differences

- The release notes of Karmic Kraken do not name the 1017 release, and Local Inference Lab had not tested it on 17 October 2026.
- The reference code of the 1017 release rounds the input of the projection `wo_a` to FP8 and back, and it has a
  fix of the indexer cache. We did not examine whether Karmic Kraken has these two changes.
