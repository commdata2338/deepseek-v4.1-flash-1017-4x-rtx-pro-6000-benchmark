Effective prefill includes scheduling/queueing through the last first token. Decode counts
cumulative server token deltas inside the common overlap of every stream; under 2 s is flagged.
Distinct random prefixes; forced output length is a speed workload, not a quality score.

| Input target | C | Prefill tok/s | Total decode tok/s | Decode/request | TTFT s | Shared s | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5000 | 1 | 7293.86 | 156.63 | 156.63 | 0.69 | 10.85 | measured |
| 5000 | 2 | 7591.15 | 291.70 | 145.85 | 1.19 | 11.46 | measured |
| 5000 | 3 | 7532.36 | 406.81 | 138.46 | 1.58 | 12.26 | measured |
| 56000 | 1 | 7851.03 | 158.90 | 158.90 | 7.13 | 10.69 | measured |
| 56000 | 2 | 7987.46 | 298.95 | 149.48 | 10.75 | 10.96 | measured |
| 56000 | 3 | 7881.93 | 409.32 | 137.26 | 14.49 | 12.22 | measured |
| 100000 | 1 | 7810.19 | 157.32 | 157.32 | 12.80 | 10.80 | measured |
| 100000 | 2 | 7765.65 | 287.83 | 143.91 | 19.52 | 11.32 | measured |
| 100000 | 3 | 7863.52 | 387.46 | 129.36 | 25.76 | 13.00 | measured |
