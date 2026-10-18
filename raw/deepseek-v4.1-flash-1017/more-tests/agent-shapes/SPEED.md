Effective prefill includes scheduling/queueing through the last first token. Decode counts
cumulative server token deltas inside the common overlap of every stream; under 2 s is flagged.
Distinct random prefixes; forced output length is a speed workload, not a quality score.

| Input target | C | Prefill tok/s | Total decode tok/s | Decode/request | TTFT s | Shared s | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5000 | 1 | 9855.85 | 376.43 | 376.43 | 0.51 | 4.51 | measured |
| 5000 | 2 | 12853.60 | 593.04 | 296.52 | 0.67 | 5.65 | measured |
| 5000 | 3 | 10773.45 | 736.05 | 247.28 | 0.81 | 6.55 | measured |
| 56000 | 1 | 15209.12 | 366.78 | 366.78 | 3.68 | 4.63 | measured |
| 56000 | 2 | 9466.25 | n/a | n/a | 7.85 | 0.00 | no_overlap |
| 56000 | 3 | 8563.63 | n/a | n/a | 11.62 | 0.00 | no_overlap |
| 100000 | 1 | 14312.06 | 350.25 | 350.25 | 6.99 | 4.85 | measured |
| 100000 | 2 | 10776.24 | n/a | n/a | 12.85 | 0.00 | no_overlap |
| 100000 | 3 | 9794.74 | n/a | n/a | 18.93 | 0.00 | no_overlap |
