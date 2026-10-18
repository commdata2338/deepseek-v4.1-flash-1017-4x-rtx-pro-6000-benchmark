Effective prefill includes scheduling/queueing through the last first token. Decode counts
cumulative server token deltas inside the common overlap of every stream; under 2 s is flagged.
Distinct random prefixes; forced output length is a speed workload, not a quality score.

| Input target | C | Prefill tok/s | Total decode tok/s | Decode/request | TTFT s | Shared s | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5000 | 1 | 10133.96 | 260.99 | 260.99 | 0.49 | 6.51 | measured |
| 5000 | 2 | 10021.05 | 364.98 | 182.49 | 0.74 | 8.81 | measured |
| 5000 | 3 | 6673.29 | 496.88 | 160.72 | 1.10 | 8.29 | measured |
| 56000 | 1 | 11566.28 | 226.58 | 226.58 | 4.84 | 7.50 | measured |
| 56000 | 2 | 7219.30 | 349.59 | 174.79 | 10.19 | 1.64 | short_window |
| 56000 | 3 | 6129.08 | n/a | n/a | 15.69 | 0.00 | no_overlap |
| 100000 | 1 | 10911.91 | 234.17 | 234.17 | 9.16 | 7.26 | measured |
| 100000 | 2 | 7702.10 | n/a | n/a | 17.81 | 0.00 | no_overlap |
| 100000 | 3 | 7360.33 | n/a | n/a | 25.41 | 0.00 | no_overlap |
