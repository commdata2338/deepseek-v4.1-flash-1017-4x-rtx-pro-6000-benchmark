"""Direct GPU-to-GPU copies between every ordered pair of GPUs: are they correct, and how fast?

A cross-device tensor copy in PyTorch enables CUDA peer access for the pair and then copies device to device, so the
source GPU writes straight into the destination GPU's memory over PCIe. That is the transfer an IOMMU in translated
mode refuses (AMD-Vi logs IO_PAGE_FAULT) until the host boots with iommu=pt. Run it with the GPUs idle, inside an
image with CUDA PyTorch:

  docker run --rm --gpus all --network none -v <this dir>:/tools:ro --entrypoint python <image> /tools/p2p_test.py

Prints one JSON document. Every pair must report "match": true; the caller also checks the kernel log for faults.
"""
import json
import sys
import time

import torch

SIZE = (int(sys.argv[1]) if len(sys.argv) > 1 else 1024) * 2**20  # bytes per buffer, default 1 GiB
REPEATS = 4


def digest(tensor):
    """Two order-sensitive sums, computed on the tensor's own GPU."""
    index = torch.arange(tensor.numel(), device=tensor.device, dtype=torch.int64)
    return int(tensor.sum().item()), int((tensor * (index % 1009 + 1)).sum().item())


def main():
    count = torch.cuda.device_count()
    out = {"torch": torch.__version__, "buffer_mib": SIZE // 2**20,
           "devices": [torch.cuda.get_device_name(i) for i in range(count)], "pairs": []}
    sources = {}
    for a in range(count):
        generator = torch.Generator(device=f"cuda:{a}")
        generator.manual_seed(1234 + a)
        x = torch.randint(0, 2**20, (SIZE // 8,), dtype=torch.int64, device=f"cuda:{a}", generator=generator)
        sources[a] = (x, digest(x))
    for a in range(count):
        for b in range(count):
            if a == b:
                continue
            x, want = sources[a]
            row = {"src": a, "dst": b, "peer_access": bool(torch.cuda.can_device_access_peer(b, a))}
            y = x.to(f"cuda:{b}")  # first copy: enables peer access for the pair
            torch.cuda.synchronize(a)
            torch.cuda.synchronize(b)
            first = digest(y) == want
            y.zero_()
            started = time.perf_counter()
            for _ in range(REPEATS):
                y.copy_(x)
            torch.cuda.synchronize(a)
            torch.cuda.synchronize(b)
            seconds = time.perf_counter() - started
            row.update(match=bool(first and digest(y) == want), gib_per_s=round(REPEATS * SIZE / seconds / 2**30, 2))
            out["pairs"].append(row)
            print(json.dumps(row), file=sys.stderr, flush=True)
            del y
    out["all_match"] = all(row["match"] for row in out["pairs"])
    out["min_gib_per_s"] = min(row["gib_per_s"] for row in out["pairs"])
    out["max_gib_per_s"] = max(row["gib_per_s"] for row in out["pairs"])
    print(json.dumps(out, indent=1))
    return 0 if out["all_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
