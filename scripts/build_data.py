"""Make every CSV file of data/ from raw/ of this repository.

  python3 scripts/build_data.py

The servers: DeepSeek-V4.1-Flash-1017 on Karmic Kraken (the vLLM fork of Local Inference Lab) with 16 slots and with
8 slots (two rounds), all measured on 17 October 2026; the same weights on SGLang earlier that day (Appendix F);
and GLM-5.3-Flash on Jovian Judgement r38, the best configuration of the GLM-5.3-Flash paper (its raw files at
version 1.5, copied with no change). The files keep the column names of the GLM paper's data/ where a test is the
same. No number is typed in by hand.
"""
import csv
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW, DATA = ROOT / "raw", ROOT / "data"
KK, SG = RAW / "deepseek-v4.1-flash-1017", RAW / "deepseek-v4.1-flash-1017-sglang"
DS16 = ("deepseek_1017_kk_16", "DeepSeek-V4.1-Flash-1017, Karmic Kraken", "16 slots, Engram tables in RAM")
DS8 = ("deepseek_1017_kk_8", "DeepSeek-V4.1-Flash-1017, Karmic Kraken", "8 slots, Engram tables in RAM, round 1")
DS8B = ("deepseek_1017_kk_8_round2", "DeepSeek-V4.1-Flash-1017, Karmic Kraken", "8 slots, Engram tables in RAM, round 2")
DSSG = ("deepseek_1017_sglang", "DeepSeek-V4.1-Flash-1017, SGLang",
        "the community recipe, 8 slots, Engram tables in RAM (Appendix F)")
GLM16 = ("glm_r38_16_aligned", "GLM-5.3-Flash, Jovian Judgement r38",
         "16 slots, GPU links on, thinking-switch template, checkpoint policy aligned (GLM paper, session 6)")
GLM8 = ("glm_r38_8", "GLM-5.3-Flash, Jovian Judgement r38", "8 slots, GPU links on, stock template (GLM paper, session 6)")
SUITE = {DS16: KK / "16-slots", DS8: KK / "8-slots", DS8B: KK / "8-slots-round2", DSSG: SG,
         GLM16: RAW / "glm-5.3-flash-r38/16-slots-policy-aligned", GLM8: RAW / "glm-5.3-flash-r38/8-slots"}
TYPES = ("prose", "code", "structured", "json")
WAVES = (("greedy.log", "greedy_1k", 0.0, 1.0, "off", 1, "off"),
         ("longctx-sampled.log", "long_context_sampled", 1.0, 1.0, "off", 1, "on, effort max"),
         ("longctx-sampled-p95.log", "long_context_sampled", 1.0, 0.95, "off", 1, "on, effort max"),
         ("longctx-p95-topk20.log", "long_context_sampled", 1.0, 0.95, "20", 1, "on, effort max"),
         ("longctx-p95-topk-off.log", "long_context_sampled", 1.0, 0.95, "off", 2, "on, effort max"),
         ("longctx-p95-topk20-pass2.log", "long_context_sampled", 1.0, 0.95, "20", 2, "on, effort max"))
SYSTEM = (("longctx-system-p95-topk20.log", 1.0, 0.95, "20", 1),
          ("longctx-system-p95-topk-off.log", 1.0, 0.95, "off", 1))
BURST = {DS16: KK / "cold-requests/burst-deepseek-1017-kk-16.jsonl", DS8: KK / "cold-requests/burst-deepseek-1017-kk.jsonl",
         DSSG: SG / "cold-requests/burst-deepseek-1017-ram-d1017.jsonl",
         GLM8: RAW / "glm-5.3-flash-r38/cold-requests/burst-jovian-r38-8slot.jsonl",
         GLM16: RAW / "glm-5.3-flash-r38/cold-requests/burst-jovian-r38-16slot-policy-aligned.jsonl"}
QUALITY = {DS8: RAW / "quality-pi/deepseek-v4.1-flash-1017", DSSG: RAW / "quality-pi/deepseek-v4.1-flash-1017-sglang",
           GLM8: RAW / "quality-pi/glm-5.3-flash-r38"}
PAIRS = ((GLM8, DS8), (DSSG, DS8), (GLM8, DSSG))
MORE = {"Karmic Kraken": KK / "more-tests", "SGLang": SG / "more-tests"}
WEIGHTS_DRIVE = "nvme1n1"  # the NVMe drive of the test host with the weights, which include the Engram tables
CDT = timezone(timedelta(hours=-5))  # the local time of the test host on 17 October 2026, as in the window monitor


def write(name, header, rows):
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / name, "w", newline="") as handle:
        out = csv.writer(handle, lineterminator="\n")
        out.writerow(header)
        out.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def wave_rows(path):
    """The JSON lines that longctx.py prints, one for each number of concurrent requests."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.startswith("{") and '"concurrency"' in line]


def wilson(k, n):
    z = 1.959964
    centre = (k + z * z / 2) / (n + z * z)
    half = z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return round(100 * (centre - half), 1), round(100 * (centre + half), 1)


def mcnemar(b, c):
    """Exact two-sided McNemar test: binomial test of b against b + c at p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def speed():
    decode, prefill, waves, reuse, integrity = [], [], [], [], []
    for (run, engine, config), directory in SUITE.items():
        for kind in TYPES:
            path = directory / f"sparkdash-decode-{kind}.json"
            if not path.is_file():
                continue
            for r in json.loads(path.read_text())["results"]:
                decode.append([run, engine, config, kind, r["concurrency"], r["streamsOk"], r["streamsFailed"],
                               r["aggregateDecodeTps"], r["medianDecodeTps"], r["minDecodeTps"], r["maxDecodeTps"],
                               r["medianTtftMs"]])
        path = directory / "sparkdash-prefill.json"
        for r in json.loads(path.read_text())["results"] if path.is_file() else []:
            prefill.append([run, engine, config, r["targetTokens"], r["promptTokens"], r["prefillTps"],
                            round(r["ttftMs"] / 1000, 3)])
        for log, probe, temperature, top_p, top_k, number, thinking in WAVES:
            for r in wave_rows(directory / log):
                waves.append([run, engine, config, probe, temperature, top_p, top_k, number, thinking, r["concurrency"],
                              r.get("context_tokens"), r.get("output_tokens"), r.get("aggregate_decode_tps"),
                              r.get("median_decode_tps"), r.get("median_ttft_s"), r.get("status")])
        for log, temperature, top_p, top_k, number in SYSTEM:
            for r in wave_rows(directory / log):
                waves.append([run, engine, config, "long_context_sampled_system_message", temperature, top_p, top_k,
                              number, "on, effort max", r["concurrency"], r.get("context_tokens"),
                              r.get("output_tokens"), r.get("aggregate_decode_tps"), r.get("median_decode_tps"),
                              r.get("median_ttft_s"), r.get("status")])
        for r in lines(directory / "warm-chat.jsonl"):
            reuse.append([run, engine, config, r["size"], r["round"], r["case"], r["prompt_tokens"], r["cached_tokens"],
                          r["ttft_s"]])
        for r in lines(directory / "integrity.jsonl"):
            integrity.append([run, engine, config, r["target_tokens"], r["prompt_tokens"], r["ok"], r["seconds"]])
    write("decode_sparkdash.csv", ["run", "engine", "configuration", "output_type", "concurrency", "streams_ok",
                                   "streams_failed", "total_tok_s", "median_stream_tok_s", "min_stream_tok_s",
                                   "max_stream_tok_s", "median_ttft_ms"], decode)
    write("prefill_sparkdash.csv", ["run", "engine", "configuration", "target_tokens", "prompt_tokens", "prefill_tok_s",
                                    "ttft_s"], prefill)
    write("concurrent_waves.csv", ["run", "engine", "configuration", "probe", "temperature", "top_p", "top_k", "pass",
                                   "thinking", "concurrency", "context_tokens", "output_tokens", "total_tok_s",
                                   "median_stream_tok_s", "median_ttft_s", "status"], waves)
    write("chat_reuse_ttft.csv", ["run", "engine", "configuration", "nominal_size", "round", "case", "prompt_tokens",
                                  "cached_tokens_reported", "ttft_s"], reuse)
    write("integrity.csv", ["run", "engine", "configuration", "target_tokens", "prompt_tokens", "passphrase_recalled",
                            "seconds"], integrity)


def reuse_checks():
    rows = []
    for (run, engine, config), directory in SUITE.items():
        for name, place in (("cache-check.jsonl", "user message"), ("cache-check-system.jsonl", "system message")):
            for number, r in enumerate(lines(directory / name), 1):
                rows.append([run, engine, config, r.get("context_in", place), number, r["step"],
                             r["shared_context_tokens"], r["prompt_tokens"], r["completion_tokens"], r["seconds"]])
    write("prefix_reuse_check.csv", ["run", "engine", "configuration", "shared_context_in", "step_number", "step",
                                     "shared_context_tokens", "prompt_tokens", "output_tokens",
                                     "seconds_to_the_full_reply"], rows)


def cold_burst():
    rows = []
    for (run, engine, config), path in BURST.items():
        for r in lines(path):
            usage = r.get("usage") or {}
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            rows.append([run, engine, config, r["name"], r["status"], usage.get("prompt_tokens"), cached,
                         usage.get("completion_tokens"), None if r.get("ttft_s") is None else round(r["ttft_s"], 3),
                         round(r["elapsed_s"], 3)])
    write("cold_burst.csv", ["run", "engine", "configuration", "request", "status", "prompt_tokens",
                             "cached_tokens_reported", "completion_tokens", "ttft_s", "elapsed_s"], rows)


def quality():
    names = {"humaneval": "HumanEval+", "mbpp": "MBPP+"}
    tasks, summary, paired, calls, grades = [], [], [], [], {}
    for (run, engine, config), directory in QUALITY.items():
        graded = {g["task_id"]: g for g in json.loads((directory / "graded.json").read_text())["tasks"]}
        records = lines(directory / "records.jsonl")
        assert {r["task_id"] for r in records} == set(graded), f"{run}: records and grades differ"
        grades[run] = graded
        first = {r["task_id"]: r["prompt_tokens"] for r in lines(directory / "first_calls.jsonl")}
        for r in records:
            g = graded[r["task_id"]]
            tasks.append([run, engine, config, names[g["dataset"]], r["task_id"], g["base_pass"], g["plus_pass"],
                          r["status"], r["limit_hit"], r["solution_present"], round(r["wall_seconds"], 3),
                          r["model_calls"], first.get(r["task_id"]), r["total_input_tokens"], r["output_tokens"]])
        for dataset in ("humaneval", "mbpp", "both"):
            rows = [g for g in graded.values() if dataset == "both" or g["dataset"] == dataset]
            n, base, plus = len(rows), sum(g["base_pass"] for g in rows), sum(g["plus_pass"] for g in rows)
            summary.append([run, engine, config, names.get(dataset, "both"), n, base, plus, round(100 * base / n, 1),
                            *wilson(base, n), round(100 * plus / n, 1), *wilson(plus, n)])
        timing = json.loads((directory / "timing.json").read_text())
        seconds = sorted(r["seconds"] for r in lines(directory / "relay.jsonl")
                         if r["path"].endswith("/chat/completions") and r.get("status") == 200
                         and timing["first_task_start"] <= r["time"] <= timing["last_task_end"])
        middle = (seconds[(len(seconds) - 1) // 2] + seconds[len(seconds) // 2]) / 2
        calls.append([run, engine, config, len(seconds), round(middle, 2), round(sum(seconds) / len(seconds), 2),
                      round(seconds[int(0.9 * len(seconds))], 2), round(sum(seconds)),
                      round((timing["last_task_end"] - timing["first_task_start"]) / 60, 1)])
    for (a, *_), (b, *_) in PAIRS:
        assert grades[a].keys() == grades[b].keys(), "a paired comparison needs the same tasks"
        for dataset in ("humaneval", "mbpp", "both"):
            ids = [t for t, g in grades[a].items() if dataset == "both" or g["dataset"] == dataset]
            for tests in ("base", "plus"):
                pa = [grades[a][t][f"{tests}_pass"] for t in ids]
                pb = [grades[b][t][f"{tests}_pass"] for t in ids]
                both = sum(x and y for x, y in zip(pa, pb))
                only_a = sum(x and not y for x, y in zip(pa, pb))
                only_b = sum(y and not x for x, y in zip(pa, pb))
                n = len(ids)
                difference = 100 * (only_b - only_a) / n
                error = 100 * math.sqrt(only_a + only_b - (only_b - only_a) ** 2 / n) / n
                paired.append([a, b, names.get(dataset, "both"), "base tests" if tests == "base" else "all tests", n,
                               both, only_a, only_b, n - both - only_a - only_b, round(mcnemar(only_a, only_b), 4),
                               round(difference, 4), round(difference - 1.959964 * error, 4),
                               round(difference + 1.959964 * error, 4)])
    write("quality_pi_tasks.csv", ["run", "engine", "configuration", "data_set", "task_id", "base_tests_pass",
                                   "all_tests_pass", "agent_status", "time_limit_hit", "solution_file_present",
                                   "wall_seconds", "model_calls", "first_prompt_tokens", "input_tokens",
                                   "output_tokens"], tasks)
    write("quality_pi_summary.csv", ["run", "engine", "configuration", "data_set", "tasks", "base_tests_pass",
                                     "all_tests_pass", "base_tests_percent", "base_tests_ci95_low",
                                     "base_tests_ci95_high", "all_tests_percent", "all_tests_ci95_low",
                                     "all_tests_ci95_high"], summary)
    write("quality_pi_paired.csv", ["run_a", "run_b", "data_set", "tests", "tasks", "both_pass", "only_a_passes",
                                    "only_b_passes", "neither_passes", "exact_mcnemar_p",
                                    "b_minus_a_percentage_points", "difference_ci95_low", "difference_ci95_high"],
           paired)
    write("quality_pi_calls.csv", ["run", "engine", "configuration", "model_calls", "median_call_s", "mean_call_s",
                                   "p90_call_s", "sum_of_call_s", "run_minutes"], calls)


def more_tests():
    """Appendix D (Karmic Kraken) and Appendix F (SGLang): the tests of the benchmark procedure for local models."""
    checks = []
    for engine, more in MORE.items():
        for folder in sorted(more.glob("qualification-*")) + [more / "soak-60-minutes"]:
            if not (folder / "summary.json").is_file():
                continue
            d = json.loads((folder / "summary.json").read_text())
            rows = d.get("checks", [])
            if d.get("part") == "soak":
                checks.append([engine, "soak, 60 minutes", "soak", sum(c["status"] == "passed" for c in rows), len(rows),
                               round((d["finished_at"] - d["started_at"]) / 60, 1), d["status"]])
                continue
            for c in rows:
                checks.append([engine, c["name"], d.get("part"), int(c["status"] == "passed"), 1, round(c["seconds"], 2),
                               c["status"]])
    write("more_qualification.csv", ["engine", "check", "part", "passed", "total", "seconds_or_minutes", "status"], checks)
    speedrows = []
    for label, folder in (("DeepSeek-V4.1-Flash-1017, Karmic Kraken", KK / "more-tests/qualification-speed"),
                          ("DeepSeek-V4.1-Flash-1017, Karmic Kraken", KK / "more-tests/agent-shapes"),
                          ("DeepSeek-V4.1-Flash-1017, SGLang", SG / "more-tests/qualification-speed"),
                          ("DeepSeek-V4.1-Flash-1017, SGLang", SG / "more-tests/agent-shapes-deepseek-v4.1-flash-1017"),
                          ("GLM-5.3-Flash r38, live unit", SG / "more-tests/agent-shapes-glm-5.3-flash")):
        test = ("qualification speed, 4,096 tokens" if folder.name == "qualification-speed"
                else "agent-shaped, 1,700 tokens")
        for path in sorted(folder.glob("speed-*-summary.json")):
            d = json.loads(path.read_text())
            speedrows.append([test, label, d["input_target"], d["concurrency"], d.get("aggregate_decode_tps"),
                              d.get("median_decode_tps"), d.get("effective_prefill_tps"), d.get("median_ttft_s"),
                              d.get("status")])
    write("more_speed.csv", ["test", "server", "input_tokens", "concurrency", "total_tok_s", "median_stream_tok_s",
                             "effective_prefill_tok_s", "median_ttft_s", "status"], speedrows)
    talk = []
    for label, folder in (("DeepSeek-V4.1-Flash-1017, Karmic Kraken", KK / "more-tests/agent-shapes"),
                          ("DeepSeek-V4.1-Flash-1017, SGLang", SG / "more-tests/agent-shapes-deepseek-v4.1-flash-1017"),
                          ("GLM-5.3-Flash r38, live unit", SG / "more-tests/agent-shapes-glm-5.3-flash")):
        for path in sorted(folder.glob("talk-*-summary.json")):
            d = json.loads(path.read_text())
            talk.append([label, d["talk_prompt_tokens"], d["talk_output_tokens"], round(d["alone_ttft_s"], 2),
                         round(d["alone_elapsed_s"], 2), round(d["under_load_ttft_s"], 2),
                         round(d["under_load_elapsed_s"], 2)])
    write("chat_reply_under_load.csv", ["server", "prompt_tokens", "output_tokens", "alone_ttft_s", "alone_reply_s",
                                        "under_load_ttft_s", "under_load_reply_s"], talk)
    table = (SG / "more-tests/single-request-recipe-matrix/TABLE.md").read_text()
    single = []
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 8 and re.fullmatch(r"\d+", cells[0]):
            single.append([int(cells[0]), int(cells[1]), float(cells[2]), float(cells[3]), float(cells[4]),
                           float(cells[5])])
    write("single_request_recipe_matrix.csv", ["input_tokens", "concurrency", "effective_prefill_tok_s",
                                               "median_ttft_s", "median_decode_tok_s_tokens_129_641",
                                               "total_shared_decode_tok_s"], single)
    residency = lines(SG / "engram-residency.jsonl")
    write("engram_residency.csv", ["time", "label", "tables_gib", "resident_gib", "resident_share",
                                          "host_mlocked_gib", "host_mem_available_gib"],
          [[r["time"], r["label"], r["tables_gib"], r["resident_gib"], r["resident_share"], r["mlocked_host_gib"],
            r["mem_available_gib"]] for r in residency])
    accept = []
    facts = json.loads((SG / "server-log-facts.json").read_text())
    values = [d["accept_len"] for d in facts["decode_lines"]]
    accept.append(["DeepSeek-V4.1-Flash-1017, SGLang", "8 slots", "decode batch lines", len(values),
                   round(sum(values) / len(values), 2), min(values), max(values), None, None])
    for slots, name in (("8 slots", "server-log-facts-8-slots.json"), ("16 slots", "server-log-facts-16-slots.json")):
        facts = json.loads((KK / name).read_text())
        values = [s["mean_acceptance_length"] for s in facts["spec"]]
        accept.append(["DeepSeek-V4.1-Flash-1017, Karmic Kraken", slots, "speculative-decoding lines (10 s)",
                       len(values), round(sum(values) / len(values), 2), min(values), max(values),
                       facts["accepted_tokens"], facts["drafted_tokens"]])
    write("decode_log_summary.csv", ["server", "slots", "source", "lines", "mean_accept_length", "min_accept_length",
                                     "max_accept_length", "accepted_draft_tokens", "drafted_tokens"], accept)
    probes = []
    for engine, path in (("Karmic Kraken", KK / "more-tests/api-feature-probes.json"),
                         ("SGLang", SG / "more-tests/api-feature-probes.json")):
        probes += [[engine, p["probe"], p["http"], p["seconds"], p["verdict"]] for p in json.loads(path.read_text())]
    write("api_feature_probes.csv", ["engine", "probe", "http_status", "seconds", "verdict"], probes)
    batch = json.loads((KK / "batch-schema-check.json").read_text())
    write("batch_schema_check.csv", ["kind", "sent", "ok", "rounds", "timeout_s", "slowest_s"],
          [[kind, v["sent"], v["ok"], batch["rounds"], batch["timeout_s"],
            max(r["seconds"] for r in batch["rows"] if r["kind"] == kind)] for kind, v in batch["by_kind"].items()])


def server_facts():
    """The start facts of the Karmic Kraken server from its start log (metadata lines only) and its launcher."""
    rows = []
    patterns = {"kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
                "max_concurrency_at_max_len": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
                "max_model_len": r"Using max model len (\d+)",
                "engram_mapped_host_gib": r"Engram mapped-host RAM: ([\d.]+) GiB",
                "engram_ram_available_gib": r"Engram mapped-host RAM: [\d.]+ GiB packed tables across TP\d+, ([\d.]+) GiB",
                "engram_ram_reserve_gib": r"Engram mapped-host RAM: .* GiB available, (\d+) GiB reserve",
                "model_weights_gpu_gib": r"Model loading took ([\d.]+) GiB",
                "engine_init_s": r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s"}
    for slots, name, config in (("16", "server-log-facts-16-slots.json", "launcher-config-16-slots.json"),
                                ("8", "server-log-facts-8-slots.json", "launcher-config-8-slots.json")):
        facts = json.loads((KK / name).read_text())
        text = "\n".join(facts["start_milestones"])
        values = {}
        for key, pattern in patterns.items():
            found = re.findall(pattern, text)
            values[key] = found[-1].replace(",", "") if found else None
        argv = json.loads((KK / config).read_text())["argv"]
        option = lambda flag: argv[argv.index(flag) + 1] if flag in argv else None  # noqa: E731
        rows.append([slots, option("--max-num-seqs"), option("--max-num-batched-tokens"),
                     option("--gpu-memory-utilization"), option("--kv-cache-dtype"), values["max_model_len"],
                     values["kv_cache_tokens"], values["max_concurrency_at_max_len"], values["engram_mapped_host_gib"],
                     values["model_weights_gpu_gib"], values["engine_init_s"], values["engram_ram_available_gib"],
                     values["engram_ram_reserve_gib"]])
    write("server_facts.csv", ["slots", "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization",
                               "kv_cache_dtype", "max_model_len", "kv_cache_tokens", "max_concurrency_at_max_len",
                               "engram_mapped_host_gib", "model_weights_gpu_gib_per_gpu", "engine_init_s",
                               "engram_ram_check_available_gib", "engram_ram_check_reserve_gib"], rows)


def clock(epoch):
    return datetime.fromtimestamp(epoch, CDT).strftime("%H:%M:%S")


def engram_kk():
    """Appendix A: the Engram tables in host memory on Karmic Kraken with 8 slots. The reads of the drive with the
    weights in two qualification runs (samples of 5 s), one check of the resident shared memory of each worker during
    the long-context test, and the locked memory of the host in the window monitor at the same times."""
    with open(KK / "window-monitor.csv", newline="") as handle:
        monitor = [(r["time"], float(r["mlocked_gib"])) for r in csv.DictReader(handle)]

    def mlocked(start, end):  # the highest value of the monitor rows from 5 s before to 5 s after
        values = [value for t, value in monitor if clock(start - 5) <= t <= clock(end + 5)]
        return max(values) if values else None

    facts = json.loads((KK / "server-log-facts-8-slots.json").read_text())
    budget = re.search(r"Engram mapped-host RAM: ([\d.]+) GiB packed tables across TP(\d+)",
                       "\n".join(facts["start_milestones"]))
    tables, ranks = float(budget.group(1)), int(budget.group(2))   # each rank has the same padded share
    rows = []
    for folder, label in (("qualification-integrity", "qualification, integrity (Appendix D)"),
                          ("qualification-speed", "qualification, speed (Appendix D)")):
        samples = [(r["time"], r["nvme"][WEIGHTS_DRIVE]["sectors_read"])
                   for r in lines(KK / "more-tests" / folder / "resources.jsonl") if WEIGHTS_DRIVE in r.get("nvme", {})]
        (first, start_sectors), (last, end_sectors) = samples[0], samples[-1]
        rates = [(b - a) * 512 / 1e6 / (u - t) for (t, a), (u, b) in zip(samples, samples[1:]) if u > t]
        read_mb = (end_sectors - start_sectors) * 512 / 1e6
        rows.append([clock(first), label, round(last - first, 1), WEIGHTS_DRIVE, round(read_mb, 1),
                     round(read_mb / (last - first), 3), round(max(rates), 3), None, None, None, tables,
                     round(tables / ranks, 2), mlocked(first, last)])
    check = json.loads((KK / "engram-residency-check.json").read_text())
    read = re.search(r"(\w+) read ([\d.]+) MB/s over ([\d.]+) s", check["output"])
    shared = [float(x) for x in re.findall(r"RssShmem ([\d.]+) GiB", check["output"])]
    rows.append([clock(check["started_epoch"]), "long-context test, top_p 0.95 and top_k 20, 8 slots",
                 float(read.group(3)), read.group(1), None, float(read.group(2)), None, len(shared), min(shared),
                 max(shared), tables, round(tables / ranks, 2), mlocked(check["started_epoch"], check["finished_epoch"])])
    write("engram_ram_kk.csv", ["start", "point_of_the_test", "seconds", "drive", "drive_read_mb", "drive_mean_mb_s",
                                "drive_highest_5s_mb_s", "workers_checked", "worker_resident_shared_gib_min",
                                "worker_resident_shared_gib_max", "tables_gib", "tables_share_each_worker_gib",
                                "host_mlocked_gib_highest"], rows)


def main():
    speed()
    reuse_checks()
    cold_burst()
    quality()
    more_tests()
    server_facts()
    engram_kk()


if __name__ == "__main__":
    main()
