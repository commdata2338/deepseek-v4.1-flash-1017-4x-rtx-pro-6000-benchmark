Effective prefill includes scheduling/queueing through the last first token. Decode counts
cumulative server token deltas inside the common overlap of every stream; under 2 s is flagged.
Distinct random prefixes; forced output length is a speed workload, not a quality score.

| Input target | C | Prefill tok/s | Total decode tok/s | Decode/request | TTFT s | Shared s | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5000 | 1 | 9524.68 | 410.51 | 410.51 | 0.52 | 9.98 | measured |
| 5000 | 4 | 9695.92 | 813.26 | 197.94 | 1.12 | 17.72 | measured |
| 5000 | 8 | 7460.85 | 1149.57 | 145.46 | 2.74 | 24.36 | measured |
| 100000 | 1 | 14068.61 | 400.90 | 400.90 | 7.11 | 10.21 | measured |
| 100000 | 4 | 6780.50 | n/a | n/a | 32.74 | 0.00 | no_overlap |
| 100000 | 8 | 6161.98 | n/a | n/a | 68.36 | 0.00 | no_overlap |
