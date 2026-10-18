"""Two ways to move a tensor from one GPU to another, timed on the same data.

  direct  the source GPU writes into the destination GPU's memory over PCIe (CUDA peer access)
  staged  the tensor is copied to a pinned buffer in host memory, then from there to the destination GPU

The staged path is what a framework falls back to when peer access is off. It is a simple stand-in, not NCCL's
shared-memory transport itself, but the route is the same: two transfers, with host memory in the middle.

  docker run --rm --gpus '"device=0,1,2"' --network none -v <this dir>:/probes:ro --entrypoint python <image> \
      /probes/gpu_copy_paths.py 256

Prints one JSON document. Every copy is checked against the source before it is timed.
"""
import json
import sys
import time

import torch

SIZE = (int(sys.argv[1]) if len(sys.argv) > 1 else 256) * 2**20  # bytes
REPEATS = 8


def timed(step, devices):
    for device in devices:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(REPEATS):
        step()
    for device in devices:
        torch.cuda.synchronize(device)
    return round(REPEATS * SIZE / (time.perf_counter() - started) / 2**30, 2)


def main():
    count = torch.cuda.device_count()
    out = {"torch": torch.__version__, "buffer_mib": SIZE // 2**20, "repeats": REPEATS, "pairs": []}
    host = torch.empty(SIZE // 8, dtype=torch.int64).pin_memory()
    for a, b in [(0, i) for i in range(1, count)]:
        x = torch.randint(0, 2**40, (SIZE // 8,), dtype=torch.int64, device=f"cuda:{a}")
        want = int(x.sum().item())
        y = x.to(f"cuda:{b}")  # enables peer access for the pair
        assert int(y.sum().item()) == want, "direct copy differs from the source"
        direct = timed(lambda: y.copy_(x), (a, b))
        y.zero_()

        def staged():
            host.copy_(x)
            y.copy_(host)

        staged()
        assert int(y.sum().item()) == want, "staged copy differs from the source"
        row = {"src": a, "dst": b, "peer_access": bool(torch.cuda.can_device_access_peer(b, a)),
               "direct_gib_per_s": direct, "staged_gib_per_s": timed(staged, (a, b))}
        out["pairs"].append(row)
        print(json.dumps(row), file=sys.stderr, flush=True)
        del x, y
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
