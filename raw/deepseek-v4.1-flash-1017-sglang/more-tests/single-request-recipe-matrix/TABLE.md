# Measured inference matrix

Effective prefill includes queuing and mixed decode work until the last request reaches its first token. Decode is the median per-request rate over emitted tokens 129–641, including serving stalls. Concurrency is requested burst size; client overlap records how many streams had emitted tokens and remained unfinished. Synthetic repeated notes, unique prefixes, 8,192-token output budget, ignore_eos=True; this is not a quality test.

| Input tokens/request | Requested C | Effective prefill tok/s | Median TTFT s | Median decode tok/s/request | Total shared decode tok/s | Shared decode tok/s/request | Shared seconds | Peak client overlap | Success |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 3012.68 | 0.17 | 146.62 | 169.67 | 169.67 | 48.28 | 1 | 1/1 |
| 8192 | 1 | 7525.75 | 1.09 | 155.43 | 171.36 | 171.36 | 47.80 | 1 | 1/1 |
| 65536 | 1 | 8014.63 | 8.18 | 155.58 | 160.44 | 160.44 | 51.05 | 1 | 1/1 |
