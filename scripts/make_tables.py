"""Print each table of the paper from data/*.csv, or check that README.md contains them.

  python3 scripts/make_tables.py               # print every table, each after a line "<!-- table: NAME -->"
  python3 scripts/make_tables.py NAME          # print one table
  python3 scripts/make_tables.py --check README.md
  python3 scripts/make_tables.py --fill README.md

The check finds each "<!-- table: NAME -->" line in the README and compares the table below it with the table that
this script makes. It ignores the padding of the cells. Change a number by measuring again, never by hand.
Where a run has two rounds or two passes of a test, a table gives their mean.
"""
import csv
import re
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DS16, DS8, DS8B, DSSG = "deepseek_1017_kk_16", "deepseek_1017_kk_8", "deepseek_1017_kk_8_round2", "deepseek_1017_sglang"
GLM16, GLM8 = "glm_r38_16_aligned", "glm_r38_8"
NAMES = {DS16: "DeepSeek, 16 slots", DS8: "DeepSeek, 8 slots", DS8B: "DeepSeek, 8 slots, round 2",
         DSSG: "DeepSeek on SGLang, 8 slots", GLM16: "GLM r38, 16 slots", GLM8: "GLM r38, 8 slots"}
GROUPS = {DS8: (DS8, DS8B)}  # the runs whose values a table averages
KIND = {"prose": "prose", "code": "code", "structured": "count task", "json": "json"}  # sparkDash "structured" counts to 200
NO_TEST, NO_VALUE = "no test", "no value"


def rows(name):
    with open(DATA / name, newline="") as handle:
        return list(csv.DictReader(handle))


def num(value, digits=0):
    if value in (None, "", "None"):
        return NO_VALUE
    x = float(value)
    return f"{x:,.{digits}f}"


def table(header, body):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines)


def ratio(a, b):
    if a in (None, "", "None") or b in (None, "", "None") or float(b) == 0:
        return NO_VALUE
    return f"{float(a) / float(b):.2f}"


def mean(values):
    values = [float(v) for v in values if v not in (None, "", "None")]
    return sum(values) / len(values) if values else None


def pick(data, run, *key, field):
    """The mean of a field over the runs of a group (or one run), for one key; None if no run has it."""
    return mean(data[(member, *key)][field] for member in GROUPS.get(run, (run,)) if (member, *key) in data)


def decode_body():
    data = {(r["run"], r["output_type"], int(r["concurrency"])): r for r in rows("decode_sparkdash.csv")}
    body = []
    for kind in ("prose", "code", "structured", "json"):
        for c in (1, 8, 16):
            d16, d8, g = (pick(data, run, kind, c, field="total_tok_s") for run in (DS16, DS8, GLM16))
            body.append([KIND[kind], c, num(d16), num(d8), num(g), ratio(d16, g)])
    return table(["Output type", "Requests", "DeepSeek 16 slots, tok/s", "DeepSeek 8 slots, tok/s", "GLM r38, tok/s",
                  "DeepSeek 16 slots / GLM"], body)


def decode_full():
    body = [[NAMES[r["run"]], KIND[r["output_type"]], r["concurrency"], r["streams_ok"], num(r["total_tok_s"], 1),
             num(r["median_stream_tok_s"], 1), num(r["median_ttft_ms"])] for r in rows("decode_sparkdash.csv")
            if r["run"] != GLM8]
    return table(["Inference stack", "Output type", "Requests", "Streams", "Total tok/s", "Median stream tok/s",
                  "Median first token, ms"], body)


def prefill_body():
    data = {(r["run"], int(r["target_tokens"])): r for r in rows("prefill_sparkdash.csv")}
    body = []
    for size in sorted({k[1] for k in data}):
        row = [f"{size:,}"]
        for run in (DS16, DS8, GLM16):
            row += [num(pick(data, run, size, field="prefill_tok_s")), num(pick(data, run, size, field="ttft_s"), 2)]
        body.append(row)
    return table(["Prompt, tokens", "DeepSeek 16 slots, tok/s", "first token s", "DeepSeek 8 slots, tok/s",
                  "first token s", "GLM r38, tok/s", "first token s"], body)


def waves_table(probe, settings, servers=(DS16, DS8, GLM16)):
    data = {}
    for r in rows("concurrent_waves.csv"):
        if r["probe"] == probe and r["total_tok_s"] not in ("", "None"):
            data.setdefault((r["run"], r["top_p"], r["top_k"], int(r["concurrency"])), []).append(r)
    body = []
    for top_p, top_k in settings:
        for c in sorted({k[3] for k in data if (k[1], k[2]) == (top_p, top_k)}):
            row = [f"top_p {top_p}, top_k {top_k}", c]
            for run in servers:
                found = [r for member in GROUPS.get(run, (run,)) for r in data.get((member, top_p, top_k, c), [])]
                row += [NO_TEST if not found else num(mean(r["total_tok_s"] for r in found)),
                        NO_TEST if not found else num(mean(r["median_ttft_s"] for r in found), 2)]
            body.append(row)
    header = ["Sampler", "Requests"]
    for run in servers:
        header += [f"{NAMES[run]}: total tok/s", "first token s"]
    return table(header, body)


def longctx_body():
    return waves_table("long_context_sampled", (("1.0", "off"), ("0.95", "off"), ("0.95", "20")))


def system_body():
    return waves_table("long_context_sampled_system_message", (("0.95", "20"), ("0.95", "off")),
                       servers=(DS16, DS8, GLM16, GLM8))


def greedy_body():
    return waves_table("greedy_1k", (("1.0", "off"),))


def warm_chat():
    """The mean of the two rounds of each request; the prompt tokens of the first round."""
    data, order = {}, []
    for r in rows("chat_reuse_ttft.csv"):
        data.setdefault((r["run"], r["nominal_size"], r["case"]), []).append(r)
        if r["case"] not in order:
            order.append(r["case"])
    body = []
    for size in ("9000", "56000"):
        for case in order:
            row = [f"{int(size):,}", case]
            for run in (DS16, GLM16):
                found = data.get((run, size, case), [])
                if not found:
                    row += [NO_TEST, NO_TEST]
                    continue
                row += [f"{int(found[0]['prompt_tokens']):,}",
                        num(sum(float(f["ttft_s"]) for f in found) / len(found), 2)]
            body.append(row)
    return table(["Nominal size", "Request", "DeepSeek, prompt tokens", "DeepSeek, first token s",
                  "GLM r38, prompt tokens", "GLM r38, first token s"], body)


def reuse_check():
    data = rows("prefix_reuse_check.csv")
    body = []
    for place in ("user message", "system message"):
        steps = []
        for r in data:
            if r["shared_context_in"] == place and r["step"] not in steps:
                steps.append(r["step"])
        for step in steps:
            row = [place, step]
            for run in (DS16, DS8, GLM8, GLM16):
                found = [x for member in GROUPS.get(run, (run,)) for x in data
                         if x["run"] == member and x["shared_context_in"] == place and x["step"] == step]
                row.append(NO_TEST if not found else num(mean(x["seconds_to_the_full_reply"] for x in found), 2))
            body.append(row)
    return table(["Shared context in", "Step", "DeepSeek 16 slots, s", "DeepSeek 8 slots, s", "GLM r38 8 slots, s",
                  "GLM r38 16 slots, s"], body)


def cold_burst():
    data = rows("cold_burst.csv")
    body = []
    for run in (DS16, DS8, GLM16, GLM8, DSSG):
        mine = [r for r in data if r["run"] == run]
        first = [float(r["ttft_s"]) for r in mine if r["ttft_s"]]
        body.append([NAMES[run], len(mine), f"{int(mine[0]['prompt_tokens']):,}", num(min(first), 2), num(max(first), 2),
                     num(max(float(r["elapsed_s"]) for r in mine), 2)])
    return table(["Inference stack", "Requests", "Prompt tokens", "First token, earliest s", "First token, latest s",
                  "Last reply ends, s"], body)


def integrity():
    data = {(r["run"], int(r["target_tokens"])): r for r in rows("integrity.csv")}
    body = []
    for size in sorted({k[1] for k in data}):
        row = [f"{size:,}"]
        for run in (DS16, DS8, GLM16):
            found = [data[(m, size)] for m in GROUPS.get(run, (run,)) if (m, size) in data]
            row += [NO_TEST if not found else ("yes" if all(r["passphrase_recalled"] == "True" for r in found) else "no"),
                    NO_TEST if not found else num(mean(r["seconds"] for r in found), 2)]
        body.append(row)
    return table(["Prompt, tokens", "DeepSeek 16 slots, recalled", "s", "DeepSeek 8 slots, recalled", "s",
                  "GLM r38, recalled", "s"], body)


def quality_body():
    data = {(r["run"], r["data_set"]): r for r in rows("quality_pi_summary.csv")}
    body = []
    for ds in ("HumanEval+", "MBPP+", "both"):
        row = [ds, data[(DS8, ds)]["tasks"]]
        for run in (DS8, GLM8, DSSG):
            r = data[(run, ds)]
            row += [f"{r['all_tests_pass']} ({r['all_tests_percent']}%)", f"{r['all_tests_ci95_low']} to {r['all_tests_ci95_high']}"]
        body.append(row)
    return table(["Data set", "Tasks", "DeepSeek, all tests pass", "95% interval", "GLM r38, all tests pass",
                  "95% interval", "DeepSeek on SGLang, all tests pass", "95% interval"], body)


def quality_paired():
    body = [[f"{NAMES[r['run_a']]} (a), {NAMES[r['run_b']]} (b)", r["data_set"], r["tests"], r["both_pass"],
             r["only_a_passes"], r["only_b_passes"], r["exact_mcnemar_p"], num(r["b_minus_a_percentage_points"], 1),
             f"{num(r['difference_ci95_low'], 1)} to {num(r['difference_ci95_high'], 1)}"]
            for r in rows("quality_pi_paired.csv")]
    return table(["Runs", "Data set", "Tests", "Both pass", "Only a", "Only b", "Exact McNemar p", "b minus a, points",
                  "95% interval"], body)


def quality_calls():
    body = [[NAMES[r["run"]], f"{int(r['model_calls']):,}", r["median_call_s"], r["mean_call_s"], r["p90_call_s"],
             r["run_minutes"]] for r in rows("quality_pi_calls.csv")]
    return table(["Inference stack", "Model calls", "Median call s", "Mean call s", "90th percentile s", "Run, minutes"], body)


def server_facts():
    body = [[r["slots"], r["max_num_batched_tokens"], r["gpu_memory_utilization"], r["kv_cache_dtype"],
             f"{int(r['max_model_len']):,}", f"{int(r['kv_cache_tokens']):,}", r["max_concurrency_at_max_len"],
             r["engram_mapped_host_gib"], r["model_weights_gpu_gib_per_gpu"], num(r["engine_init_s"])]
            for r in rows("server_facts.csv")]
    return table(["Slots", "Batched tokens", "GPU memory share", "KV cache type", "Context, tokens", "KV cache, tokens",
                  "Full-length requests that fit", "Engram tables in host RAM, GiB", "Weights per GPU, GiB",
                  "Engine start, s"], body)


def acceptance():
    body = [[r["server"], r["slots"], r["source"], r["lines"], r["mean_accept_length"], r["min_accept_length"],
             r["max_accept_length"],
             NO_VALUE if not r["drafted_tokens"] else f"{100 * int(r['accepted_draft_tokens']) / int(r['drafted_tokens']):.1f}%"]
            for r in rows("decode_log_summary.csv")]
    return table(["Inference stack", "Slots", "Log lines", "Lines", "Mean tokens per step", "Lowest", "Highest",
                  "Draft tokens accepted"], body)


def batch_schema():
    names = {"plain": "long chat, streamed", "schema": "strict JSON schema", "tool_required": "tool_choice required"}
    body = [[names[r["kind"]], r["sent"], r["ok"], num(r["slowest_s"], 1)] for r in rows("batch_schema_check.csv")]
    return table(["Request", "Sent", "Correct and complete", "Slowest, s"], body)


def variation():
    """Round 2 against round 1 of the 8-slot server, for the tests of the speed suite."""
    data = {(r["run"], r["output_type"], int(r["concurrency"])): r for r in rows("decode_sparkdash.csv")}
    body = []
    for kind in ("prose", "code", "structured", "json"):
        for c in (1, 8):
            a, b = data.get((DS8, kind, c)), data.get((DS8B, kind, c))
            if a and b:
                body.append([f"decode, {KIND[kind]}", c, num(a["total_tok_s"]), num(b["total_tok_s"]),
                             f"{100 * (float(b['total_tok_s']) / float(a['total_tok_s']) - 1):+.1f}%"])
    waves = {}
    for r in rows("concurrent_waves.csv"):
        if r["total_tok_s"] not in ("", "None") and r["pass"] == "1":
            waves[(r["run"], r["probe"], r["top_p"], r["top_k"], int(r["concurrency"]))] = r
    for probe, top_p, top_k, label in (("greedy_1k", "1.0", "off", "short greedy answers"),
                                       ("long_context_sampled", "1.0", "off", "long context, top_p 1.0"),
                                       ("long_context_sampled", "0.95", "off", "long context, top_p 0.95")):
        for c in (1, 4, 8):
            a, b = waves.get((DS8, probe, top_p, top_k, c)), waves.get((DS8B, probe, top_p, top_k, c))
            if a and b:
                body.append([label, c, num(a["total_tok_s"]), num(b["total_tok_s"]),
                             f"{100 * (float(b['total_tok_s']) / float(a['total_tok_s']) - 1):+.1f}%"])
    return table(["Test", "Requests", "Round 1, tok/s", "Round 2, tok/s", "Difference"], body)


def sglang_compare():
    data = {(r["run"], r["output_type"], int(r["concurrency"])): r for r in rows("decode_sparkdash.csv")}
    body = []
    for kind in ("prose", "code", "structured", "json"):
        for c in (1, 8):
            k, s = pick(data, DS8, kind, c, field="total_tok_s"), pick(data, DSSG, kind, c, field="total_tok_s")
            body.append([f"decode, {KIND[kind]}", c, num(k), num(s), ratio(k, s)])
    prefill = {(r["run"], int(r["target_tokens"])): r for r in rows("prefill_sparkdash.csv")}
    for size in (8192, 65536, 262144):
        k, s = pick(prefill, DS8, size, field="prefill_tok_s"), pick(prefill, DSSG, size, field="prefill_tok_s")
        body.append([f"cold prefill, {size:,} tokens", 1, num(k), num(s), ratio(k, s)])
    return table(["Test", "Requests", "Karmic Kraken, tok/s", "SGLang, tok/s", "Karmic Kraken / SGLang"], body)


def single_request():
    body = [[f"{int(r['input_tokens']):,}", r["concurrency"], num(r["effective_prefill_tok_s"]), r["median_ttft_s"],
             num(r["median_decode_tok_s_tokens_129_641"], 1), num(r["total_shared_decode_tok_s"], 1)]
            for r in rows("single_request_recipe_matrix.csv")]
    return table(["Prompt, tokens", "Requests", "Prefill tok/s", "First token s", "Decode tok/s, tokens 129 to 641",
                  "Decode tok/s, whole reply"], body)


def more_qualification():
    body = [[r["engine"], r["check"], r["part"], f"{r['passed']} of {r['total']}", r["seconds_or_minutes"], r["status"]]
            for r in rows("more_qualification.csv")]
    return table(["Engine", "Check", "Part", "Passed", "Seconds (soak: minutes)", "Result"], body)


def more_speed():
    body = [[r["test"], r["server"], f"{int(r['input_tokens']):,}", r["concurrency"], num(r["total_tok_s"]),
             num(r["median_stream_tok_s"]), num(r["median_ttft_s"], 2)] for r in rows("more_speed.csv")]
    return table(["Test", "Inference stack", "Prompt, tokens", "Requests", "Total tok/s", "Median stream tok/s",
                  "First token s"], body)


def chat_reply():
    body = [[r["server"], f"{int(r['prompt_tokens']):,}", r["output_tokens"], num(r["alone_ttft_s"], 2),
             num(r["alone_reply_s"], 2), num(r["under_load_ttft_s"], 2), num(r["under_load_reply_s"], 2)]
            for r in rows("chat_reply_under_load.csv")]
    return table(["Inference stack", "Prompt, tokens", "Output tokens", "Alone: first token s", "Alone: reply s",
                  "Under load: first token s", "Under load: reply s"], body)


def engram():
    body = [[r["time"], r["label"], r["tables_gib"], r["resident_gib"], f"{100 * float(r['resident_share']):.1f}%",
             r["host_mlocked_gib"]] for r in rows("engram_residency.csv")]
    return table(["Time", "Point of the test", "Tables, GiB", "Resident, GiB", "Resident share", "Locked on the host, GiB"],
                 body)


def probes():
    body = [[r["engine"], r["probe"], r["http_status"], r["seconds"], r["verdict"]] for r in rows("api_feature_probes.csv")]
    return table(["Engine", "Probe", "HTTP status", "Seconds", "Result"], body)


TABLES = {name: function for name, function in globals().items()
          if callable(function) and name not in ("rows", "num", "table", "ratio", "mean", "pick", "waves_table", "check",
                                                 "fill", "normal", "main", "Path")
          and not name.startswith("_") and function.__module__ == __name__}


def normal(text):
    return [re.sub(r"\s*\|\s*", "|", line.strip()) for line in text.strip().splitlines()]


def check(readme):
    text = Path(readme).read_text()
    failures, found = 0, 0
    for match in re.finditer(r"<!-- table: (\w+) -->\n((?:\|.*\n?)+)", text):
        found += 1
        name, block = match.group(1), match.group(2)
        if name not in TABLES:
            print(f"unknown table {name}")
            failures += 1
        elif normal(block) != normal(TABLES[name]()):
            print(f"table {name} differs from data/")
            failures += 1
    print(f"{found} tables checked, {failures} differ")
    return 1 if failures else 0


def fill(readme):
    """Write the current table below each "<!-- table: NAME -->" line of the README (the old table, if any, goes)."""
    text = Path(readme).read_text()
    def replace(match):
        return f"<!-- table: {match.group(1)} -->\n{TABLES[match.group(1)]()}\n"
    text = re.sub(r"<!-- table: (\w+) -->\n(?:\|.*\n?)*", replace, text)
    Path(readme).write_text(text)
    return check(readme)


def main():
    if sys.argv[1:2] == ["--check"]:
        return check(sys.argv[2])
    if sys.argv[1:2] == ["--fill"]:
        return fill(sys.argv[2])
    for name in sys.argv[1:] or TABLES:
        print(f"<!-- table: {name} -->\n{TABLES[name]()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
