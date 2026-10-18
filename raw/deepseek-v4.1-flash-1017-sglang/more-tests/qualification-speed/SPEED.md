Effective prefill includes scheduling/queueing through the last first token. Decode counts
cumulative server token deltas inside the common overlap of every stream; under 2 s is flagged.
Distinct random prefixes; forced output length is a speed workload, not a quality score.

| Input target | C | Prefill tok/s | Total decode tok/s | Decode/request | TTFT s | Shared s | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5000 | 1 | 6712.07 | 162.96 | 162.96 | 0.74 | 25.13 | measured |
| 5000 | 4 | 7535.64 | 481.15 | 119.98 | 1.98 | 32.71 | measured |
| 5000 | 8 | 7515.80 | 742.40 | 93.61 | 3.30 | 42.41 | measured |
| 100000 | 1 | 7807.83 | 158.97 | 158.97 | 12.81 | 25.76 | measured |
| 100000 | 4 | 7935.79 | 486.81 | 121.46 | 31.91 | 32.67 | measured |
| 100000 | 8 | 7864.34 | 714.35 | 90.48 | 57.89 | 44.34 | measured |
